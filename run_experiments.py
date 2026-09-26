"""
Experiment driver for the TinyLLM paper's three experiments, run against the
.bin files prepare_data.py writes.

    python3 run_experiments.py e1 [--quick]   # does more data help? (scaling the *data*)
    python3 run_experiments.py e2 [--quick]   # does more capacity help? (scaling the *model*)
    python3 run_experiments.py e3 [--quick]   # LoRA vs. full fine-tuning, same final architecture

Every run --quick uses a drastically reduced config (fewer steps, fewer
seeds, fewer model sizes) so the entire pipeline -- data loading, LR
selection, multi-seed training, evaluation, JSON output -- can be smoke-
tested in well under a minute before committing to the real multi-hour run.
Always run --quick first on a machine you haven't run this on before.

Every individual run's config, seed, git commit, device, learning curve,
and final metrics get written to results/runs/<experiment>/<run_name>.json
via train.save_run_json -- these files are the only source of truth,
figures and results/report.md are generated from them (see analysis.py /
generate_figures.py), never hand-edited.
"""

import argparse
import copy
import json
import os
import time

import torch

from data import TokenStream, stream_from_bin
from lora import add_lora, merge_lora, trainable_parameters
from metrics import perplexity, words_constraint_rate
from model import GPT
from train import (
    build_optimizer,
    device_description,
    get_device,
    save_run_json,
    seed_everything,
    sync_device,
    train_model,
)

DATA_DIR = "data"
RESULTS_DIR = "results/runs"
CHECKPOINTS_DIR = "checkpoints"

BLOCK_SIZE = 512
DROPOUT = 0.1
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01


def log(msg):
    print(f"[run_experiments] {msg}", flush=True)


def load_meta():
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        return json.load(f)


def load_tokenizer():
    from BPE import BPETokenizer
    return BPETokenizer.load(os.path.join(DATA_DIR, "tokenizer.json"))


def bin_path(name):
    return os.path.join(DATA_DIR, f"{name}.bin")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def steps_for_token_budget(token_budget, batch_size, block_size):
    return max(1, token_budget // (batch_size * block_size))


def select_best_lr(build_model, train_stream, dev_stream, lrs, *, batch_size, block_size,
                    steps, eval_interval, eval_batches, device, seed, log_prefix=""):
    """
    Trains a short run per candidate LR (all starting from the *same* model
    init and seeing the *same* batches, since train_model's seed controls
    both), and returns (best_lr, results) where results maps lr -> final
    dev loss. Shared by Experiments 2 and 3 -- both need "pick a learning
    rate that's fair to this arm" rather than reusing one global LR that
    happens to suit only one of them (the original paper's Experiment 2 used
    one LR for both LoRA and full fine-tuning, which is exactly the kind of
    confound this replaces).
    """
    results = {}
    for lr in lrs:
        seed_everything(seed)  # identical init AND identical batch draw for every candidate LR
        model = build_model()
        history = train_model(
            model, train_stream, dev_stream, device,
            block_size=block_size, batch_size=batch_size, learning_rate=lr,
            warmup_steps=max(1, steps // 10), max_steps=steps, eval_interval=eval_interval,
            eval_batches=eval_batches, weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP,
            seed=seed, log_prefix=f"{log_prefix}[lr={lr:.0e}] ",
        )
        results[lr] = history[-1]["val_loss"]
    best_lr = min(results, key=results.get)
    log(f"{log_prefix}LR sweep: {results} -> best {best_lr:.0e}")
    return best_lr, results


# =========================================================================
# Experiment 1: does more data help? (fixed token budget, varying how much
# of it is *unique* data vs. repeats of the same data)
# =========================================================================

E1_MODEL_CONFIG = dict(num_layers=8, num_heads=8, embed_dim=128, block_size=BLOCK_SIZE, dropout=DROPOUT)
E1_DATA_SIZES = [2_000_000, 8_000_000, 32_000_000, 100_000_000]  # unique tokens available to train on
E1_TOKEN_BUDGET = 100_000_000  # tokens actually trained on, regardless of how much is unique
E1_SEEDS = [0, 1, 2]
E1_BATCH_SIZE = 32
E1_LEARNING_RATE = 3e-4
E1_EVAL_INTERVAL_FRACTION = 20  # ~20 eval points per run


def run_e1(quick=False):
    device = get_device()
    log(f"Experiment 1 (data scaling) on {device_description(device)}")

    tokenizer = load_tokenizer()
    full_train = stream_from_bin(bin_path("tinystories_train"), block_size=BLOCK_SIZE)
    test_stream = stream_from_bin(bin_path("tinystories_test"), block_size=BLOCK_SIZE)

    data_sizes = [50_000, 200_000] if quick else E1_DATA_SIZES
    seeds = [0] if quick else E1_SEEDS
    token_budget = 200_000 if quick else E1_TOKEN_BUDGET
    batch_size = 8 if quick else E1_BATCH_SIZE

    steps = steps_for_token_budget(token_budget, batch_size, BLOCK_SIZE)
    eval_interval = max(1, steps // E1_EVAL_INTERVAL_FRACTION)
    log(f"token budget {token_budget:,} -> {steps} steps at batch_size={batch_size}, block_size={BLOCK_SIZE}")

    for data_size in data_sizes:
        data_size = min(data_size, len(full_train.data) - 1)
        train_stream = TokenStream(full_train.data[:data_size], block_size=BLOCK_SIZE)

        for seed in seeds:
            run_name = f"data{data_size}_seed{seed}"
            log(f"=== {run_name} ===")
            seed_everything(seed)
            model = GPT(vocab_size=len(tokenizer.vocab), **E1_MODEL_CONFIG).to(device)

            history = train_model(
                model, train_stream, test_stream, device,
                block_size=BLOCK_SIZE, batch_size=batch_size, learning_rate=E1_LEARNING_RATE,
                warmup_steps=max(1, steps // 20), max_steps=steps, eval_interval=eval_interval,
                eval_batches=10, weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, seed=seed,
                log_prefix=f"[{run_name}] ",
            )

            val_loss, val_ppl = perplexity(model, test_stream.data[:2_000_000], BLOCK_SIZE, batch_size, device)
            log(f"{run_name}: final val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

            save_run_json(
                os.path.join(RESULTS_DIR, "e1", f"{run_name}.json"),
                config={**E1_MODEL_CONFIG, "unique_data_tokens": data_size, "token_budget": token_budget,
                        "batch_size": batch_size, "learning_rate": E1_LEARNING_RATE, "max_steps": steps,
                        "vocab_size": len(tokenizer.vocab)},
                seed=seed, device=device, history=history,
                final_metrics={"val_loss": val_loss, "val_perplexity": val_ppl, "params": count_params(model)},
            )


# =========================================================================
# Experiment 2: does more capacity help, over a wide range? Model family:
# embed_dim = 16 * num_layers, num_heads = num_layers (head_dim fixed at 16).
# =========================================================================

E2_LAYER_COUNTS = [2, 4, 6, 8, 10, 12, 14]
E2_LR_GRID = [3e-4, 1e-3, 3e-3]
E2_SEEDS = [0, 1, 2]
E2_TOKEN_BUDGET = 200_000_000  # one pass over this many tokens, same for every size
E2_BATCH_SIZE = 32
E2_LR_SWEEP_TOKEN_BUDGET = 2_000_000  # short runs just to rank candidate LRs


def e2_model_config(num_layers):
    return dict(num_layers=num_layers, num_heads=num_layers, embed_dim=16 * num_layers,
                block_size=BLOCK_SIZE, dropout=DROPOUT)


def run_e2(quick=False):
    device = get_device()
    log(f"Experiment 2 (model scaling) on {device_description(device)}")

    tokenizer = load_tokenizer()
    train_stream = stream_from_bin(bin_path("tinystories_train"), block_size=BLOCK_SIZE)
    dev_stream = stream_from_bin(bin_path("tinystories_dev"), block_size=BLOCK_SIZE)
    test_stream = stream_from_bin(bin_path("tinystories_test"), block_size=BLOCK_SIZE)

    layer_counts = [2, 4, 6] if quick else E2_LAYER_COUNTS
    seeds = [0] if quick else E2_SEEDS
    batch_size = 8 if quick else E2_BATCH_SIZE
    token_budget = 100_000 if quick else E2_TOKEN_BUDGET
    lr_sweep_budget = 20_000 if quick else E2_LR_SWEEP_TOKEN_BUDGET
    lr_grid = [3e-4, 3e-3] if quick else E2_LR_GRID

    steps = steps_for_token_budget(token_budget, batch_size, BLOCK_SIZE)
    lr_sweep_steps = max(1, steps_for_token_budget(lr_sweep_budget, batch_size, BLOCK_SIZE))
    eval_interval = max(1, steps // 20)

    for num_layers in layer_counts:
        cfg = e2_model_config(num_layers)
        vocab_size = len(tokenizer.vocab)

        def build_model(cfg=cfg, vocab_size=vocab_size):
            return GPT(vocab_size=vocab_size, **cfg).to(device)

        probe = build_model()
        params = count_params(probe)
        del probe
        log(f"--- layers={num_layers} -> {params:,} params ---")

        best_lr, lr_results = select_best_lr(
            build_model, train_stream, dev_stream, lr_grid,
            batch_size=batch_size, block_size=BLOCK_SIZE, steps=lr_sweep_steps,
            eval_interval=max(1, lr_sweep_steps // 5), eval_batches=5, device=device, seed=0,
            log_prefix=f"[L{num_layers} lr-sweep] ",
        )

        for seed in seeds:
            run_name = f"layers{num_layers}_seed{seed}"
            log(f"=== {run_name} (lr={best_lr:.0e}) ===")
            seed_everything(seed)
            model = build_model()

            history = train_model(
                model, train_stream, dev_stream, device,
                block_size=BLOCK_SIZE, batch_size=batch_size, learning_rate=best_lr,
                warmup_steps=max(1, steps // 20), max_steps=steps, eval_interval=eval_interval,
                eval_batches=10, weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, seed=seed,
                log_prefix=f"[{run_name}] ",
            )

            val_loss, val_ppl = perplexity(model, test_stream.data[:2_000_000], BLOCK_SIZE, batch_size, device)
            log(f"{run_name}: final test val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

            if num_layers == 8 and seed == 0:
                os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
                ckpt_path = os.path.join(CHECKPOINTS_DIR, "e2_base_layers8_seed0.pt")
                torch.save({
                    "model_state_dict": model.state_dict(), "tokenizer": tokenizer.to_dict(),
                    "config": cfg,
                }, ckpt_path)
                log(f"saved base checkpoint for Experiment 3: {ckpt_path}")

            save_run_json(
                os.path.join(RESULTS_DIR, "e2", f"{run_name}.json"),
                config={**cfg, "batch_size": batch_size, "learning_rate": best_lr, "max_steps": steps,
                        "lr_sweep": lr_results, "vocab_size": vocab_size},
                seed=seed, device=device, history=history,
                final_metrics={"val_loss": val_loss, "val_perplexity": val_ppl, "params": params},
            )


# =========================================================================
# Experiment 3: LoRA vs. full fine-tuning, adapting a pretrained TinyStories
# model to TinyStoriesInstruct, with merge_lora making the final LoRA model
# architecturally identical (same param count) to the fully fine-tuned one.
# =========================================================================

E3_LORA_RANKS = [1, 2, 4, 8, 16, 32, 64]
E3_LORA_LR_GRID = [3e-4, 1e-3, 3e-3, 1e-2]
E3_FULL_LR_GRID = [3e-5, 1e-4, 3e-4, 1e-3]
E3_SEEDS = [0, 1, 2, 3, 4]
E3_TOKEN_BUDGET = 20_000_000
E3_LR_SWEEP_TOKEN_BUDGET = 1_000_000
E3_BATCH_SIZE = 32


def load_base_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    from BPE import BPETokenizer
    tokenizer = BPETokenizer.from_dict(checkpoint["tokenizer"])
    model = GPT(vocab_size=len(tokenizer.vocab), **checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, tokenizer


def build_e3_variant(base_model, mode, r=None):
    """mode: 'frozen' (no fine-tuning baseline), 'lora' (wrap + train A/B,
    then merge_lora at the end so the saved/evaluated model has the same
    architecture as 'full'), or 'full' (every parameter trainable)."""
    model = copy.deepcopy(base_model)
    if mode == "frozen":
        for p in model.parameters():
            p.requires_grad = False
    elif mode == "lora":
        add_lora(model, r=r, alpha=r * 2)
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown mode: {mode}")
    return model


def run_e3(quick=False, base_checkpoint=None):
    device = get_device()
    log(f"Experiment 3 (LoRA vs. full fine-tuning) on {device_description(device)}")

    if base_checkpoint is None:
        default_ckpt = os.path.join(CHECKPOINTS_DIR, "e2_base_layers8_seed0.pt")
        if os.path.exists(default_ckpt):
            base_checkpoint = default_ckpt
        elif quick:
            base_checkpoint = None  # trained fresh below, quick mode only
        else:
            raise FileNotFoundError(
                f"no base checkpoint given and {default_ckpt} doesn't exist yet -- "
                f"run `run_experiments.py e2` first (it saves this checkpoint), or pass --base-checkpoint"
            )

    if base_checkpoint is not None:
        base_model, tokenizer = load_base_checkpoint(base_checkpoint, device)
        log(f"loaded base checkpoint from {base_checkpoint} ({count_params(base_model):,} params)")
    else:
        # --quick with no checkpoint available yet: train a tiny base model
        # from scratch just so the rest of the pipeline can be exercised.
        tokenizer = load_tokenizer()
        pre_stream = stream_from_bin(bin_path("tinystories_train"), block_size=BLOCK_SIZE)
        seed_everything(0)
        base_model = GPT(vocab_size=len(tokenizer.vocab), num_layers=2, num_heads=2,
                          embed_dim=32, block_size=BLOCK_SIZE, dropout=DROPOUT).to(device)
        train_model(base_model, pre_stream, pre_stream, device, block_size=BLOCK_SIZE,
                    batch_size=8, learning_rate=3e-3, max_steps=20, eval_interval=10,
                    eval_batches=2, seed=0, log_prefix="[quick base pretrain] ")

    instruct_train = stream_from_bin(bin_path("instruct_train"), block_size=BLOCK_SIZE,
                                      mask_path=bin_path("instruct_train_mask"))
    instruct_dev = stream_from_bin(bin_path("instruct_dev"), block_size=BLOCK_SIZE,
                                    mask_path=bin_path("instruct_dev_mask"))
    instruct_test = stream_from_bin(bin_path("instruct_test"), block_size=BLOCK_SIZE,
                                     mask_path=bin_path("instruct_test_mask"))
    tinystories_test = stream_from_bin(bin_path("tinystories_test"), block_size=BLOCK_SIZE)

    with open(os.path.join(DATA_DIR, "words_eval_prompts.json")) as f:
        words_prompts = json.load(f)
    if quick:
        words_prompts = words_prompts[:5]

    ranks = [1, 8, 64] if quick else E3_LORA_RANKS
    seeds = [0] if quick else E3_SEEDS
    batch_size = 8 if quick else E3_BATCH_SIZE
    token_budget = 20_000 if quick else E3_TOKEN_BUDGET
    lr_sweep_budget = 4_000 if quick else E3_LR_SWEEP_TOKEN_BUDGET
    lora_lr_grid = [3e-4, 1e-2] if quick else E3_LORA_LR_GRID
    full_lr_grid = [3e-5, 1e-3] if quick else E3_FULL_LR_GRID

    steps = steps_for_token_budget(token_budget, batch_size, BLOCK_SIZE)
    lr_sweep_steps = max(1, steps_for_token_budget(lr_sweep_budget, batch_size, BLOCK_SIZE))
    eval_interval = max(1, steps // 10)

    baseline_ppl_before = {}
    for name, stream in [("tinystories", tinystories_test), ("instruct", instruct_test)]:
        loss, ppl = perplexity(base_model, stream.data[:1_000_000], BLOCK_SIZE, batch_size, device,
                                mask=stream.mask[:1_000_000] if stream.mask is not None else None)
        baseline_ppl_before[name] = {"loss": loss, "perplexity": ppl}
    log(f"base model (before fine-tuning): {baseline_ppl_before}")

    arms = [{"name": "baseline_frozen", "mode": "frozen", "r": None}]
    arms += [{"name": f"lora_r{r}", "mode": "lora", "r": r} for r in ranks]
    arms += [{"name": "full_finetune", "mode": "full", "r": None}]

    for arm in arms:
        log(f"--- {arm['name']} ---")

        if arm["mode"] == "frozen":
            best_lr, lr_results, steps_this_arm = None, {}, 0
        else:
            lr_grid = lora_lr_grid if arm["mode"] == "lora" else full_lr_grid

            def build_variant(arm=arm):
                return build_e3_variant(base_model, arm["mode"], arm.get("r"))

            best_lr, lr_results = select_best_lr(
                build_variant, instruct_train, instruct_dev, lr_grid,
                batch_size=batch_size, block_size=BLOCK_SIZE, steps=lr_sweep_steps,
                eval_interval=max(1, lr_sweep_steps // 5), eval_batches=5, device=device, seed=0,
                log_prefix=f"[{arm['name']} lr-sweep] ",
            )
            steps_this_arm = steps

        for seed in seeds:
            run_name = f"{arm['name']}_seed{seed}"
            seed_everything(seed)

            if arm["mode"] == "frozen":
                model = build_e3_variant(base_model, "frozen")
                history = []
                train_time = 0.0
            else:
                model = build_e3_variant(base_model, arm["mode"], arm.get("r"))
                start = time.time()
                history = train_model(
                    model, instruct_train, instruct_dev, device,
                    block_size=BLOCK_SIZE, batch_size=batch_size, learning_rate=best_lr,
                    warmup_steps=max(1, steps_this_arm // 10), max_steps=steps_this_arm,
                    eval_interval=eval_interval, eval_batches=10, weight_decay=WEIGHT_DECAY,
                    grad_clip=GRAD_CLIP, seed=seed, log_prefix=f"[{run_name}] ",
                )
                sync_device(device)
                train_time = time.time() - start

                if arm["mode"] == "lora":
                    model = merge_lora(model)  # same architecture/param count as 'full' from here on

            trainable, total = trainable_parameters(build_e3_variant(base_model, arm["mode"], arm.get("r")))

            instruct_loss, instruct_ppl = perplexity(
                model, instruct_test.data[:1_000_000], BLOCK_SIZE, batch_size, device,
                mask=instruct_test.mask[:1_000_000],
            )
            forget_loss, forget_ppl = perplexity(
                model, tinystories_test.data[:1_000_000], BLOCK_SIZE, batch_size, device,
            )
            words_rate = words_constraint_rate(model, tokenizer, words_prompts, device,
                                                max_new_tokens=100 if quick else 150)

            log(f"{run_name}: instruct_ppl={instruct_ppl:.2f} forget_ppl={forget_ppl:.2f} "
                f"words_rate={words_rate:.3f} train_time={train_time:.1f}s")

            save_run_json(
                os.path.join(RESULTS_DIR, "e3", f"{run_name}.json"),
                config={"mode": arm["mode"], "r": arm.get("r"), "batch_size": batch_size,
                        "learning_rate": best_lr, "max_steps": steps_this_arm, "lr_sweep": lr_results,
                        "block_size": BLOCK_SIZE},
                seed=seed, device=device, history=history,
                final_metrics={
                    "trainable_params": trainable, "total_params": total,
                    "trainable_pct": 100 * trainable / total,
                    "instruct_val_loss": instruct_loss, "instruct_val_perplexity": instruct_ppl,
                    "tinystories_forget_val_loss": forget_loss, "tinystories_forget_val_perplexity": forget_ppl,
                    "words_constraint_rate": words_rate, "train_time_sec": train_time,
                    "baseline_before_finetuning": baseline_ppl_before,
                },
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["e1", "e2", "e3"])
    parser.add_argument("--quick", action="store_true", help="Drastically reduced config for pipeline smoke-testing.")
    parser.add_argument("--base-checkpoint", default=None, help="(e3 only) path to a pretrained checkpoint.")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(DATA_DIR, "meta.json")):
        raise FileNotFoundError(f"{DATA_DIR}/meta.json not found -- run prepare_data.py first")

    if args.experiment == "e1":
        run_e1(quick=args.quick)
    elif args.experiment == "e2":
        run_e2(quick=args.quick)
    elif args.experiment == "e3":
        run_e3(quick=args.quick, base_checkpoint=args.base_checkpoint)


if __name__ == "__main__":
    main()
