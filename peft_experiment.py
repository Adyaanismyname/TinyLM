"""
PEFT/LoRA vs full fine-tuning, matched by trainable parameter count -- and
both measured against a "do nothing" baseline (the pretrained checkpoint,
un-fine-tuned) plus the unigram-frequency floor from metrics.py.

Loads a pretrained base checkpoint (produced by train.py), then for each
variant below, fine-tunes a fresh copy of the base model for the same
number of steps on the same data and reports:

  - trainable parameters (and what fraction of the full model that is)
  - final train/val loss and perplexity
  - wall-clock training time

The trainable-param column plus the baseline row are what make this
apples-to-apples: "does LoRA at rank r actually move the needle over just
using the pretrained model as-is, and how does that compare to spending
100% of the parameters on full fine-tuning?"

Run standalone (needs checkpoint.pt from train.py first):

    python3 peft_experiment.py
"""

import copy
import json
import os
import time

import torch

from lora import add_lora, trainable_parameters
from metrics import load_checkpoint, load_split_token_ids, perplexity, unigram_baseline_perplexity
from train import flatten, train_model

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_STEPS = 500
EVAL_INTERVAL = 100
EVAL_BATCHES = 20
NUM_TRAIN_EXAMPLES = 3000
NUM_VAL_EXAMPLES = 500

# "baseline_pretrained" is the checkpoint exactly as train.py left it --
# zero additional training, zero trainable params -- so every other row's
# improvement can be read as "what did spending these params actually buy".
VARIANTS = [
    {"name": "baseline_pretrained", "mode": "frozen"},
    {"name": "lora_r4", "mode": "lora", "r": 4},
    {"name": "lora_r8", "mode": "lora", "r": 8},
    {"name": "lora_r16", "mode": "lora", "r": 16},
    {"name": "full_finetune", "mode": "full"},
]


def build_variant(base_model, mode, r=None):
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


def run():
    base_model, tokenizer, device = load_checkpoint()
    block_size = base_model.block_size

    print("loading + encoding TinyStories train/val splits with the checkpoint's tokenizer...")
    train_token_ids, _ = load_split_token_ids(tokenizer, "train", NUM_TRAIN_EXAMPLES)
    val_token_ids, _ = load_split_token_ids(tokenizer, "validation", NUM_VAL_EXAMPLES)
    train_data = flatten(train_token_ids)
    val_data = flatten(val_token_ids)

    unigram_ppl = unigram_baseline_perplexity(train_token_ids, len(tokenizer.vocab))
    print(f"unigram frequency baseline perplexity: {unigram_ppl:.2f}  (weakest possible reference point)")

    results = []
    for variant in VARIANTS:
        print(f"\n=== {variant['name']} ===")
        model = build_variant(base_model, variant["mode"], variant.get("r")).to(device)
        trainable, total = trainable_parameters(model)
        print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        if variant["mode"] == "frozen":
            train_time = 0.0
        else:
            start = time.time()
            train_model(
                model, train_data, val_data, device,
                block_size=block_size, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                max_steps=MAX_STEPS, eval_interval=EVAL_INTERVAL, eval_batches=EVAL_BATCHES,
                log_prefix=f"[{variant['name']}] ",
            )
            train_time = time.time() - start

        val_loss, val_ppl = perplexity(model, val_data, block_size, BATCH_SIZE, device)

        results.append({
            "name": variant["name"],
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100 * trainable / total,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
            "train_time_sec": train_time,
        })

    print("\n=== PEFT comparison (unigram baseline ppl: {:.2f}) ===".format(unigram_ppl))
    print(f"{'variant':>20} | {'trainable':>10} | {'%':>6} | {'val_loss':>9} | {'val_ppl':>8} | {'train_s':>8}")
    for r in results:
        print(
            f"{r['name']:>20} | {r['trainable_params']:>10,} | {r['trainable_pct']:>5.2f}% | "
            f"{r['val_loss']:>9.4f} | {r['val_perplexity']:>8.2f} | {r['train_time_sec']:>8.1f}"
        )

    os.makedirs("results", exist_ok=True)
    with open("results/peft_results.json", "w") as f:
        json.dump({"unigram_baseline_perplexity": unigram_ppl, "variants": results}, f, indent=2)
    print("\nsaved results/peft_results.json")


if __name__ == "__main__":
    run()
