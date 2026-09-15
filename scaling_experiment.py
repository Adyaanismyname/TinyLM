"""
Scaling experiment: trains several GPT configs of increasing parameter
count on the *same* tokenizer/data (so param count is the only thing that
varies), and records final validation loss/perplexity for each -- the raw
numbers behind a "does more parameters actually help" scaling curve. Also
reports the unigram-frequency baseline perplexity as a reference floor every
config should be well below.

Run standalone:

    python3 scaling_experiment.py
"""

import json
import os
import time

from dataset import build_dataset
from model import GPT
from train import get_device, flatten, train_model
from metrics import perplexity, unigram_baseline_perplexity

VOCAB_SIZE = 1000
BLOCK_SIZE = 128
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_STEPS = 2000       # kept modest so the whole sweep finishes in reasonable time
EVAL_INTERVAL = 500
EVAL_BATCHES = 20
DROPOUT = 0.1

# Depth/width grow together, from far below train.py's default down to
# roughly it, so param count increases geometrically across the sweep.
CONFIGS = [
    {"name": "tiny", "embed_dim": 32, "num_heads": 2, "num_layers": 2},
    {"name": "small", "embed_dim": 64, "num_heads": 4, "num_layers": 4},
    {"name": "medium", "embed_dim": 96, "num_heads": 4, "num_layers": 6},
    {"name": "base", "embed_dim": 128, "num_heads": 8, "num_layers": 8},
    {"name": "large", "embed_dim": 192, "num_heads": 8, "num_layers": 10},
]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run():
    device = get_device()
    print(f"using device: {device}")

    print("building dataset once (shared tokenizer/data across every config)...")
    tokenizer, train_token_ids, val_token_ids = build_dataset(vocab_size=VOCAB_SIZE)
    train_data = flatten(train_token_ids)
    val_data = flatten(val_token_ids)
    print(f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    unigram_ppl = unigram_baseline_perplexity(train_token_ids, len(tokenizer.vocab))
    print(f"unigram frequency baseline perplexity: {unigram_ppl:.2f}  (every config below should beat this)")

    results = []
    for cfg in CONFIGS:
        print(
            f"\n=== {cfg['name']} (embed_dim={cfg['embed_dim']}, "
            f"heads={cfg['num_heads']}, layers={cfg['num_layers']}) ==="
        )

        model = GPT(
            vocab_size=len(tokenizer.vocab),
            block_size=BLOCK_SIZE,
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            dropout=DROPOUT,
        ).to(device)

        num_params = count_params(model)
        print(f"parameters: {num_params:,}")

        start = time.time()
        train_model(
            model, train_data, val_data, device,
            block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
            max_steps=MAX_STEPS, eval_interval=EVAL_INTERVAL, eval_batches=EVAL_BATCHES,
            log_prefix=f"[{cfg['name']}] ",
        )
        train_time = time.time() - start

        val_loss, val_ppl = perplexity(model, val_data, BLOCK_SIZE, BATCH_SIZE, device)

        results.append({
            "name": cfg["name"],
            "params": num_params,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
            "train_time_sec": train_time,
        })

    print(f"\n=== scaling results (unigram baseline ppl: {unigram_ppl:.2f}) ===")
    print(f"{'name':>8} | {'params':>10} | {'val_loss':>9} | {'val_ppl':>8} | {'train_s':>8}")
    for r in results:
        print(
            f"{r['name']:>8} | {r['params']:>10,} | {r['val_loss']:>9.4f} | "
            f"{r['val_perplexity']:>8.2f} | {r['train_time_sec']:>8.1f}"
        )

    os.makedirs("results", exist_ok=True)
    with open("results/scaling_results.json", "w") as f:
        json.dump({"unigram_baseline_perplexity": unigram_ppl, "configs": results}, f, indent=2)
    print("\nsaved results/scaling_results.json")


if __name__ == "__main__":
    run()
