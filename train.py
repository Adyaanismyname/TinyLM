"""
Trains the GPT model in model.py on the TinyStories data prepared by
dataset.py, using plain next-token prediction.

How training actually works, in order:

  1. Take a chunk of token ids, say [5, 9, 2, 7, 1].
     Input  x = [5, 9, 2, 7]   (everything but the last token)
     Target y = [9, 2, 7, 1]   (the same sequence, shifted one to the left)
     So at every position, the "label" is simply the next token in the text.
     This is why no manual labeling is needed — the text labels itself.

  2. Forward pass: the model turns x into logits of shape
     (batch, seq_len, vocab_size) — a predicted distribution over the next
     token at every position — and model.forward() compares that against y
     with cross-entropy loss (see model.py).

  3. Backward pass: loss.backward() uses autograd to compute the gradient of
     the loss with respect to every weight in the model — i.e. "which
     direction should each weight move to make this prediction less wrong."

  4. Optimizer step: AdamW nudges every weight a small distance in that
     direction (scaled by the learning rate). optimizer.zero_grad() clears
     old gradients first since PyTorch accumulates them by default.

  Repeat that loop thousands of times over randomly sampled chunks, and the
  model gradually gets better at predicting next tokens — which, at
  generation time, is exactly what lets it produce coherent text.
"""

import os
import time
import torch

from dataset import build_dataset
from model import GPT

# ---- hyperparameters -------------------------------------------------
VOCAB_SIZE = 1000
BLOCK_SIZE = 128       # how many tokens of context the model sees at once
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8
DROPOUT = 0.1

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_STEPS = 3000
EVAL_INTERVAL = 200
EVAL_BATCHES = 20
# -----------------------------------------------------------------------


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def flatten(token_id_lists):
    """
    Concatenates every story's token ids into one long 1D tensor. Training
    then samples random windows from this stream rather than padding each
    story to a fixed length — simpler, and no wasted compute on padding.
    """
    flat = [tid for story in token_id_lists for tid in story]
    return torch.tensor(flat, dtype=torch.long)


def get_batch(data, block_size, batch_size, device):
    """
    Samples `batch_size` random windows of length `block_size + 1` from the
    token stream, then splits each into an input chunk and a target chunk
    shifted by one position (see module docstring).
    """
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in starts])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, num_batches):
    """
    Averages loss over several random batches instead of a single one, so
    the reported number isn't just noise from one lucky/unlucky sample.
    Uses model.eval() to disable dropout while measuring.
    """
    model.eval()
    losses = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        batch_losses = torch.zeros(num_batches)
        for i in range(num_batches):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss, _ = model(x, y)
            batch_losses[i] = loss.item()
        losses[name] = batch_losses.mean().item()
    model.train()
    return losses


def train_model(
    model, train_data, val_data, device, *,
    block_size, batch_size, learning_rate, max_steps, eval_interval, eval_batches,
    log_prefix="",
):
    """
    The core training loop, factored out of main() so scaling/PEFT
    experiments can reuse the exact same mechanics on differently-sized or
    differently-frozen models. Only optimizes parameters with
    requires_grad=True, so it transparently supports LoRA (see lora.py) --
    everything else is frozen and simply doesn't get an optimizer state.

    Returns a history list of {step, train_loss, val_loss, elapsed} dicts.
    """
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # AdamW = Adam with a small weight-decay penalty pulling weights toward
    # zero, which fights overfitting. It's the standard optimizer for
    # transformers because it adapts its effective step size per-parameter,
    # which handles the very different gradient scales that show up across
    # embeddings, attention, and feed-forward layers.
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    history = []
    start_time = time.time()

    for step in range(1, max_steps + 1):
        x, y = get_batch(train_data, block_size, batch_size, device)

        _, loss, _ = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == 1:
            losses = estimate_loss(
                model, train_data, val_data, block_size, batch_size, device, eval_batches
            )
            elapsed = time.time() - start_time
            history.append({
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "elapsed": elapsed,
            })
            print(
                f"{log_prefix}step {step:5d} | train loss {losses['train']:.4f} "
                f"| val loss {losses['val']:.4f} | {elapsed:.1f}s elapsed"
            )

    return history


def main():
    device = get_device()
    print(f"using device: {device}")

    print("building dataset (loading TinyStories + training BPE tokenizer)...")
    tokenizer, train_token_ids, val_token_ids = build_dataset(vocab_size=VOCAB_SIZE)

    train_data = flatten(train_token_ids)
    val_data = flatten(val_token_ids)
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

    train_model(
        model, train_data, val_data, device,
        block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS, eval_interval=EVAL_INTERVAL, eval_batches=EVAL_BATCHES,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": tokenizer.vocab,
        "merges": tokenizer.merges,
        "unk_token": tokenizer.unk_token,
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

    # Sanity-check generation: start from a single token and let the model
    # ramble, just to see it's producing tokenizer-decodable text.
    prompt_ids = torch.tensor([[train_data[0].item()]], device=device)
    generated = model.generate(prompt_ids, max_new_tokens=200, temperature=0.8, top_k=50)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
