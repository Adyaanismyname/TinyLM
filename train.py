"""
Trains the GPT model in model.py on token-id data prepared by prepare_data.py
(or, for small ad-hoc runs, data.py's in-memory TokenStream), using plain
next-token prediction.

How training actually works, in order:

  1. Take a chunk of token ids, say [5, 9, 2, 7, 1].
     Input  x = [5, 9, 2, 7]   (everything but the last token)
     Target y = [9, 2, 7, 1]   (the same sequence, shifted one to the left)
     So at every position, the "label" is simply the next token in the text.
     This is why no manual labeling is needed -- the text labels itself.
     (Instruction-tuning targets additionally have header positions set to
     IGNORE_INDEX by data.py's EpochBatcher, so only "Story:" tokens count.)

  2. Forward pass: the model turns x into logits of shape
     (batch, seq_len, vocab_size) -- a predicted distribution over the next
     token at every position -- and model.forward() compares that against y
     with cross-entropy loss (see model.py).

  3. Backward pass: loss.backward() uses autograd to compute the gradient of
     the loss with respect to every weight in the model -- i.e. "which
     direction should each weight move to make this prediction less wrong."

  4. Optimizer step: AdamW nudges every weight a small distance in that
     direction (scaled by the learning rate, which here follows a linear
     warmup then a cosine decay -- see lr_at_step). Gradients are clipped to
     a maximum norm first, so one unusually large batch can't throw a
     step-8 model's weights somewhere training never recovers from.
     optimizer.zero_grad() clears old gradients first since PyTorch
     accumulates them by default.

  Repeat that loop many times over the data.py-sampled batches, and the
  model gradually gets better at predicting next tokens -- which, at
  generation time, is exactly what lets it produce coherent text.

What changed from the original version, and why every change is here:
  - Seeding (seed_everything) -- so "same seed -> same run" is actually
    true, which every paired comparison in the paper (LoRA rank vs. rank,
    scaling size vs. size) depends on.
  - A warmup+cosine LR schedule and weight decay applied only to 2D+
    weights (not biases/LayerNorm gains) -- the standard modern recipe;
    the old constant-LR/blanket-decay setup wasn't wrong, just not what a
    reviewer would expect "the training recipe" to mean today.
  - Gradient clipping -- cheap insurance against one bad batch, previously
    absent.
  - Timing excludes evaluation -- the original loop's `elapsed` included
    every in-loop eval pass, which is why the paper's own LoRA-vs-full-FT
    timing numbers turned out to be confounded (more eval calls for a
    longer run, more time counted that has nothing to do with training).
  - EpochBatcher (data.py) instead of ad hoc uniform-random sampling -- see
    data.py's module docstring for why that matters for paired comparisons.
"""

import json
import math
import os
import platform
import random
import subprocess
import time

import numpy as np
import torch

from data import EpochBatcher, TokenStream, stream_from_array

# ---- hyperparameters (train.py's own standalone run; experiment scripts
# override all of these with their own configs) --------------------------
VOCAB_SIZE = 1000
BLOCK_SIZE = 128       # how many tokens of context the model sees at once
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8
DROPOUT = 0.1

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WARMUP_STEPS = 100
MAX_STEPS = 3000
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EVAL_INTERVAL = 200
EVAL_BATCHES = 20
SEED = 0
# -----------------------------------------------------------------------


def seed_everything(seed):
    """Seeds every RNG this codebase's training path touches. Doesn't
    guarantee bit-identical results across different hardware/backends (CUDA
    kernels aren't all deterministic even with a fixed seed) but does make
    reruns on the *same* device reproducible, and is a prerequisite for the
    paired-batch guarantee data.py's EpochBatcher provides -- the model init
    itself needs to match too, not just the data order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device):
    """Blocks until all queued GPU work finishes, so a wall-clock timer read
    right after this call doesn't undercount work that's still running
    asynchronously in the background."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def device_description(device):
    """A human-readable device string for run metadata -- e.g. the actual
    GPU model name, not just 'cuda', since Experiment 4/5's whole point is
    comparing across specific hardware."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        return f"{platform.processor() or platform.machine()} (MPS)"
    return platform.processor() or platform.machine()


def flatten(token_id_lists):
    """
    Concatenates every story's token ids into one long 1D tensor. Kept for
    small/legacy in-memory use (wrap the result with data.stream_from_array
    to get a seeded, epoch-based TokenStream); prepare_data.py's memmap .bin
    files are the path for anything larger than fits comfortably in RAM.
    """
    flat = [tid for story in token_id_lists for tid in story]
    return torch.tensor(flat, dtype=torch.long)


def lr_at_step(step, base_lr, warmup_steps, max_steps, min_lr_ratio=0.1):
    """Linear warmup for `warmup_steps`, then cosine decay from base_lr down
    to `min_lr_ratio * base_lr` by `max_steps`. warmup_steps=0 disables
    warmup (constant-then-decay); the original train.py used a constant LR
    throughout, which is still available by passing warmup_steps=max_steps
    (schedule never leaves the flat warmup plateau) -- but the default here
    is the modern warmup+cosine recipe."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * decay)


def build_optimizer(model, learning_rate, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)):
    """
    AdamW with weight decay applied only to 2D+ parameters (linear/embedding
    weight matrices), not to 1D ones (biases, LayerNorm gains) -- decaying a
    LayerNorm scale or a bias term toward zero has no principled
    regularization justification and is the standard modern-recipe
    exclusion (see e.g. nanoGPT). betas=(0.9, 0.95) rather than PyTorch's
    default 0.999 second-moment decay -- the more common choice for
    language-model training, adapting the per-parameter step size a bit
    faster.
    """
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas)


def get_batch(data, block_size, batch_size, device):
    """
    Backward-compatible single-batch helper for callers that just want one
    batch from an in-memory tensor without setting up an EpochBatcher (e.g.
    a quick script or notebook). train_model itself uses EpochBatcher
    directly for the seeded, no-repeats-per-epoch guarantee; this wraps the
    same underlying TokenStream machinery with an unseeded one-off draw.
    """
    stream = data if isinstance(data, TokenStream) else stream_from_array(data, block_size)
    batcher = EpochBatcher(stream, batch_size, seed=random.randint(0, 2**31 - 1), device=device)
    return batcher.next_batch()


@torch.no_grad()
def estimate_loss(model, train_batcher, val_batcher, num_batches):
    """
    Averages loss over several batches instead of a single one, so the
    reported number isn't just noise from one lucky/unlucky sample. Uses
    model.eval() to disable dropout while measuring. Draws from the same
    EpochBatchers training uses (rather than a separate seeded draw), so
    repeated eval calls don't perturb the training data order.
    """
    model.eval()
    losses = {}
    for name, batcher in [("train", train_batcher), ("val", val_batcher)]:
        batch_losses = torch.zeros(num_batches)
        for i in range(num_batches):
            x, y = batcher.next_batch()
            _, loss, _ = model(x, y)
            batch_losses[i] = loss.item()
        losses[name] = batch_losses.mean().item()
    model.train()
    return losses


def train_model(
    model, train_data, val_data, device, *,
    block_size, batch_size, learning_rate, max_steps, eval_interval, eval_batches,
    warmup_steps=0, weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, seed=SEED,
    log_prefix="",
):
    """
    The core training loop, factored out of main() so scaling/PEFT
    experiments can reuse the exact same mechanics on differently-sized or
    differently-frozen models. Only optimizes parameters with
    requires_grad=True, so it transparently supports LoRA (see lora.py) --
    everything else is frozen and simply doesn't get an optimizer state.

    train_data/val_data accept anything data.py can turn into a TokenStream:
    a TokenStream itself (e.g. from data.stream_from_bin, for real memmap
    data with an instruction-tuning loss mask already applied), a 1D
    tensor/array (wrapped automatically), or an EpochBatcher directly (if
    the caller wants to control its seed/epoch state itself, e.g. to hand
    two arms of a comparison the *same* batcher instance).

    Returns a history list of {step, train_loss, val_loss, elapsed, lr}
    dicts. `elapsed` is training time only -- evaluation passes are timed
    separately and excluded, since folding them in previously confounded
    the very train-time comparisons (e.g. LoRA vs. full fine-tuning) this
    codebase exists to make.
    """
    def to_batcher(data, base_seed):
        if isinstance(data, EpochBatcher):
            return data
        stream = data if isinstance(data, TokenStream) else stream_from_array(data, block_size)
        return EpochBatcher(stream, batch_size, seed=base_seed, device=device)

    train_batcher = to_batcher(train_data, seed)
    val_batcher = to_batcher(val_data, seed + 1)  # different seed: val order need not match train's

    model.train()
    optimizer = build_optimizer(model, learning_rate, weight_decay=weight_decay)

    history = []
    elapsed = 0.0

    for step in range(1, max_steps + 1):
        lr = lr_at_step(step - 1, learning_rate, warmup_steps, max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = train_batcher.next_batch()

        step_start = time.time()
        _, loss, _ = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
        optimizer.step()
        sync_device(device)
        elapsed += time.time() - step_start

        if step % eval_interval == 0 or step == 1:
            losses = estimate_loss(model, train_batcher, val_batcher, eval_batches)
            history.append({
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "elapsed": elapsed,
                "lr": lr,
            })
            print(
                f"{log_prefix}step {step:5d} | train loss {losses['train']:.4f} "
                f"| val loss {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.1f}s elapsed (train only)"
            )

    return history


def save_run_json(path, *, config, seed, device, history, final_metrics, extra=None):
    """
    Writes one JSON per run: exactly what's needed to reproduce it (config,
    seed, git commit) and exactly what it measured (learning curve, final
    metrics) -- so every number in the paper traces back to a file instead
    of a hand-copied table. run_experiments.py/bench_*.py call this for
    every run they produce; results/report.md and every figure are
    generated by reading these files, never by hand-editing a table.
    """
    payload = {
        "config": config,
        "seed": seed,
        "git_commit": git_commit_hash(),
        "device": device_description(device) if isinstance(device, torch.device) else str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "history": history,
        "final_metrics": final_metrics,
    }
    if extra:
        payload.update(extra)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    from dataset import load_tinystories_texts
    from BPE import BPETokenizer, count_words, train_bpe
    from metrics import perplexity
    from model import GPT

    seed_everything(SEED)
    device = get_device()
    print(f"using device: {device} ({device_description(device)})")

    print("building a small tokenizer + dataset for a standalone run...")
    train_texts = load_tinystories_texts("train", num_examples=8000)
    val_texts = load_tinystories_texts("validation", num_examples=2000)

    base_chars, merges = train_bpe(count_words(train_texts), vocab_size=VOCAB_SIZE - 2)
    tokenizer = BPETokenizer(base_chars, merges)

    train_ids = [tokenizer.encode(t, add_eos=True) for t in train_texts]
    val_ids = [tokenizer.encode(t, add_eos=True) for t in val_texts]
    train_data = flatten(train_ids)
    val_data = flatten(val_ids)
    print(f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    model = GPT(
        vocab_size=len(tokenizer.vocab),
        block_size=BLOCK_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {num_params:,}")

    history = train_model(
        model, train_data, val_data, device,
        block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS, max_steps=MAX_STEPS, eval_interval=EVAL_INTERVAL,
        eval_batches=EVAL_BATCHES, seed=SEED,
    )

    val_loss, val_ppl = perplexity(model, val_data, BLOCK_SIZE, BATCH_SIZE, device)
    print(f"final val loss: {val_loss:.4f} | val perplexity: {val_ppl:.2f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "config": {
            "block_size": BLOCK_SIZE,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
    }
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(checkpoint, "checkpoints/checkpoint.pt")
    print("saved checkpoints/checkpoint.pt")

    save_run_json(
        "results/runs/train_standalone.json",
        config={
            "block_size": BLOCK_SIZE, "embed_dim": EMBED_DIM, "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS, "dropout": DROPOUT, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "warmup_steps": WARMUP_STEPS, "max_steps": MAX_STEPS,
            "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "vocab_size": len(tokenizer.vocab),
        },
        seed=SEED, device=device, history=history,
        final_metrics={"val_loss": val_loss, "val_perplexity": val_ppl, "params": num_params},
    )

    # Sanity-check generation: start from a single token and let the model
    # ramble, just to see it's producing tokenizer-decodable text.
    prompt_ids = torch.tensor([[train_data[0].item()]], device=device)
    generated = model.generate(prompt_ids, max_new_tokens=200, temperature=0.8, top_k=50, eos_id=tokenizer.eos_id)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
