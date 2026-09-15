"""
End-to-end experiment runner for the TinyLLM article.

Builds the dataset/tokenizer once, then:

  STAGE 1 (scaling): trains three model sizes from scratch on identical
    data -- ~1M params (4 layers / 4 heads), ~1.35M (6 layers / 6 heads),
    ~1.7M (8 layers / 8 heads) -- to see how much parameter count alone
    actually buys.

  STAGE 2 (PEFT): takes the largest trained model as a base and compares
    LoRA at a few ranks against full fine-tuning and against doing nothing,
    matched by trainable parameter count (see peft_experiment.py).

  STAGE 3 (latency): benchmarks autoregressive generation with vs without
    the KV cache (see metrics.py) at each model size.

Every number is written to results/report.md (human-readable) and
results/report_data.json (raw numbers, for plotting in the article).

This can take a long time end to end (BPE training + 3 training runs +
PEFT fine-tuning + latency sweeps) -- consider running it in the
background:

    python3 run_experiments.py

Individual checkpoints for each trained model size are also saved to
checkpoints/checkpoint_<name>.pt.
"""

import json
import os
import time
from datetime import datetime

import torch

from dataset import build_dataset
from model import GPT
from train import get_device, flatten, train_model
from metrics import perplexity, topk_accuracy, bits_per_char, unigram_baseline_perplexity, compare_kv_cache
from peft_experiment import build_variant, VARIANTS as PEFT_VARIANTS
from lora import trainable_parameters

# ---- experiment configuration -------------------------------------------
RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"

VOCAB_SIZE = 1000
NUM_TRAIN = 8000  # explicit here (not left to dataset.py's own default) so the
NUM_VAL = 2000    # report is self-documenting about exactly how much data was used
BLOCK_SIZE = 128
DROPOUT = 0.1

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_STEPS = 3000
EVAL_INTERVAL = 200
EVAL_BATCHES = 20

# Each entry's embed_dim is solved for below so the *actual* param count
# lands close to target_params, given the requested layers/heads.
MODEL_CONFIGS = [
    {"name": "small_1M", "num_layers": 4, "num_heads": 4, "target_params": 1_000_000},
    {"name": "medium_1.35M", "num_layers": 6, "num_heads": 6, "target_params": 1_350_000},
    {"name": "large_1.7M", "num_layers": 8, "num_heads": 8, "target_params": 1_700_000},
]

PEFT_BASE_MODEL = "large_1.7M"  # which trained model the PEFT/LoRA comparison adapts
PEFT_MAX_STEPS = 500
PEFT_EVAL_INTERVAL = 100
PEFT_EVAL_BATCHES = 20

LATENCY_PROMPT_LEN = 10
LATENCY_GEN_LENGTHS = (20, 50, 100)
LATENCY_NUM_REPEATS = 3
# --------------------------------------------------------------------------


def resolve_embed_dim(target_params, num_layers, num_heads, vocab_size, block_size, dropout, max_dim=1024):
    """
    Searches multiples of num_heads (so it always divides embed_dim evenly)
    for the one whose actual GPT parameter count is closest to
    target_params. There's no simple closed form once biases/layernorms are
    counted, so this just builds throwaway models and counts directly.
    """
    best_dim, best_params, best_diff = None, None, None
    dim = num_heads
    while dim <= max_dim:
        model = GPT(
            vocab_size=vocab_size, block_size=block_size, embed_dim=dim,
            num_heads=num_heads, num_layers=num_layers, dropout=dropout,
        )
        params = sum(p.numel() for p in model.parameters())
        diff = abs(params - target_params)
        if best_diff is None or diff < best_diff:
            best_dim, best_params, best_diff = dim, params, diff
        elif params > target_params * 1.3:
            # Past the target and getting worse -- every larger candidate
            # from here only overshoots further.
            break
        dim += num_heads
    return best_dim, best_params


def evaluate_model(model, tokenizer, val_data, val_token_ids, device):
    val_loss, val_ppl = perplexity(model, val_data, model.block_size, BATCH_SIZE, device)
    acc = topk_accuracy(model, val_data, model.block_size, BATCH_SIZE, device, ks=(1, 5))
    decoded_texts = [tokenizer.decode(ids) for ids in val_token_ids]
    bpc = bits_per_char(val_loss, val_token_ids, decoded_texts)
    return {
        "val_loss": val_loss,
        "val_perplexity": val_ppl,
        "bits_per_char": bpc,
        "top1_accuracy": acc[1],
        "top5_accuracy": acc[5],
    }


def save_checkpoint(model, tokenizer, name):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": tokenizer.vocab,
        "merges": tokenizer.merges,
        "unk_token": tokenizer.unk_token,
        "config": {
            "block_size": model.block_size,
            "embed_dim": model.token_embedding.embedding_dim,
            "num_heads": model.blocks[0].attn.num_heads,
            "num_layers": len(model.blocks),
            "dropout": DROPOUT,
        },
    }
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINTS_DIR, f"checkpoint_{name}.pt")
    torch.save(checkpoint, path)
    return path


def run_scaling(tokenizer, train_data, val_data, val_token_ids, device):
    print("\n" + "=" * 70)
    print("STAGE 1: scaling -- training each model size from scratch")
    print("=" * 70)

    results = []
    models = {}
    for cfg in MODEL_CONFIGS:
        embed_dim, resolved_params = resolve_embed_dim(
            cfg["target_params"], cfg["num_layers"], cfg["num_heads"], len(tokenizer.vocab), BLOCK_SIZE, DROPOUT
        )
        print(
            f"\n--- {cfg['name']}: embed_dim={embed_dim}, layers={cfg['num_layers']}, "
            f"heads={cfg['num_heads']} -> {resolved_params:,} params (target {cfg['target_params']:,}) ---"
        )

        model = GPT(
            vocab_size=len(tokenizer.vocab), block_size=BLOCK_SIZE, embed_dim=embed_dim,
            num_heads=cfg["num_heads"], num_layers=cfg["num_layers"], dropout=DROPOUT,
        ).to(device)

        start = time.time()
        train_model(
            model, train_data, val_data, device,
            block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
            max_steps=MAX_STEPS, eval_interval=EVAL_INTERVAL, eval_batches=EVAL_BATCHES,
            log_prefix=f"[{cfg['name']}] ",
        )
        train_time = time.time() - start

        metrics_result = evaluate_model(model, tokenizer, val_data, val_token_ids, device)
        checkpoint_path = save_checkpoint(model, tokenizer, cfg["name"])

        results.append({
            "name": cfg["name"], "embed_dim": embed_dim, "num_layers": cfg["num_layers"],
            "num_heads": cfg["num_heads"], "params": resolved_params,
            "train_time_sec": train_time, "checkpoint": checkpoint_path,
            **metrics_result,
        })
        models[cfg["name"]] = model

    return results, models


def run_peft(base_model, train_data, val_data, device):
    print("\n" + "=" * 70)
    print(f"STAGE 2: PEFT/LoRA vs full fine-tune (base model: {PEFT_BASE_MODEL})")
    print("=" * 70)

    results = []
    for variant in PEFT_VARIANTS:
        print(f"\n--- {variant['name']} ---")
        model = build_variant(base_model, variant["mode"], variant.get("r")).to(device)
        trainable, total = trainable_parameters(model)
        print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        if variant["mode"] == "frozen":
            train_time = 0.0
        else:
            start = time.time()
            train_model(
                model, train_data, val_data, device,
                block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                max_steps=PEFT_MAX_STEPS, eval_interval=PEFT_EVAL_INTERVAL, eval_batches=PEFT_EVAL_BATCHES,
                log_prefix=f"[{variant['name']}] ",
            )
            train_time = time.time() - start

        val_loss, val_ppl = perplexity(model, val_data, BLOCK_SIZE, BATCH_SIZE, device)
        results.append({
            "name": variant["name"], "trainable_params": trainable, "total_params": total,
            "trainable_pct": 100 * trainable / total, "val_loss": val_loss,
            "val_perplexity": val_ppl, "train_time_sec": train_time,
        })

    return results


def run_latency(models, device):
    print("\n" + "=" * 70)
    print("STAGE 3: KV-cache generation latency, per model size")
    print("=" * 70)

    results = {}
    for name, model in models.items():
        print(f"\n--- {name} ---")
        results[name] = compare_kv_cache(
            model, device, prompt_len=LATENCY_PROMPT_LEN,
            gen_lengths=LATENCY_GEN_LENGTHS, num_repeats=LATENCY_NUM_REPEATS,
        )

    return results


def collect_metadata(device, tokenizer, train_data, val_data):
    """
    Everything a write-up would otherwise need to reverse-engineer from a
    checkpoint after the fact: exact data/tokenizer sizes, environment, and
    the training/PEFT/latency hyperparameters actually used this run.
    """
    import platform
    import torch

    return {
        "environment": {
            "device": str(device),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "data": {
            "num_train_stories": NUM_TRAIN,
            "num_val_stories": NUM_VAL,
            "train_tokens": len(train_data),
            "val_tokens": len(val_data),
            "vocab_size": len(tokenizer.vocab),
            "num_merges": len(tokenizer.merges),
            "unk_token": tokenizer.unk_token,
        },
        "training": {
            "block_size": BLOCK_SIZE, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "max_steps": MAX_STEPS, "dropout": DROPOUT, "optimizer": "AdamW (PyTorch defaults except lr)",
            "gradient_clipping": None, "random_seed": None,
            "note": "No torch.manual_seed() call exists in the training/experiment path, and get_batch() "
                    "draws unseeded random windows -- every run and every variant sees a different batch "
                    "sequence, and each configuration below is trained exactly once (no repeats).",
        },
        "peft": {
            "max_steps": PEFT_MAX_STEPS, "base_model": PEFT_BASE_MODEL,
            "lora_target_modules": ["attn.qkv_proj", "attn.out_proj", "ff.net[0]", "ff.net[2]"],
            "lora_alpha_formula": "alpha = 2 * r for every rank tested",
            "lora_dropout": 0.0,
            "identical_init": "every variant is a deepcopy() of the same trained base model instance",
        },
        "latency": {
            "prompt_len": LATENCY_PROMPT_LEN, "gen_lengths": LATENCY_GEN_LENGTHS,
            "num_repeats": LATENCY_NUM_REPEATS, "decoding": "sampling (temperature=0.8, top_k=50)",
            "synchronized_timing": "torch.mps.synchronize()/torch.cuda.synchronize() called before "
                                   "both the start and end of the timed region",
            "peak_memory_measured": False,
        },
    }


def write_report(scaling_results, unigram_ppl, peft_results, latency_results, metadata):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, "report_data.json"), "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "metadata": metadata,
            "unigram_baseline_perplexity": unigram_ppl,
            "scaling": scaling_results,
            "peft": {"base_model": PEFT_BASE_MODEL, "variants": peft_results},
            "latency": latency_results,
        }, f, indent=2)

    lines = [
        "# TinyLLM experiment report",
        f"\nGenerated {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        f"Unigram frequency baseline perplexity: {unigram_ppl:.2f} (reference floor -- "
        f"every model below should land well under this).\n",
        "## 0. Experimental controls\n",
        "What keeps the comparisons below apples-to-apples:\n",
        "- **One shared tokenizer/dataset.** `build_dataset()` runs once; the same BPE vocab and the "
        "same train/val token streams are reused for every model size and every PEFT variant, so "
        "tokenization is never a confound between rows.",
        "- **Scaling isolates parameter count.** `num_layers`/`num_heads` are set directly per config; "
        "`embed_dim` is solved (`resolve_embed_dim`) to hit each target parameter count. Learning rate, "
        "batch size, step count, eval cadence, and dropout are identical across all three sizes -- only "
        "capacity varies.",
        "- **Exact, non-sampled evaluation.** Val loss/perplexity/accuracy/bits-per-char are computed over "
        "*every* window of the validation set (`metrics.perplexity` / `topk_accuracy`), not random sampled "
        "batches like the training-time progress printouts -- so these numbers are deterministic for a "
        "given model, not noise from one lucky batch.",
        "- **Two reference floors, not just relative comparisons.** The unigram frequency baseline above is "
        "the weakest reasonable model; `baseline_pretrained` in the PEFT table is the base checkpoint with "
        "*zero* additional training. Every other row's improvement is measured against actually doing "
        "nothing, not just against each other.",
        "- **PEFT rows differ in exactly one variable.** Every variant (`lora_r4/r8/r16`, `full_finetune`) "
        "starts from an identical deep copy of the same pretrained base model, trains on identical data, "
        "for the identical step budget (`PEFT_MAX_STEPS`). The only thing that changes between rows is "
        "which/how many parameters are trainable -- that's what the trainable-param and %-of-total columns "
        "are for.",
        "- **KV cache: verified correctness, not just speed.** `model.py`'s cached and non-cached generation "
        "paths are checked (in `model.py`'s own smoke test) to produce bit-for-bit identical sampled tokens "
        "given the same seed, so the latency numbers below measure pure speed, not a behavior change. Each "
        "timing excludes one warmup generation (lazy setup shouldn't count) and averages "
        f"{LATENCY_NUM_REPEATS} repeats.\n",
        "## 1. Scaling: does more parameters actually help?\n",
        "| model | layers | heads | embed_dim | params | val loss | val ppl | bits/char | top-1 acc | top-5 acc | train time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in scaling_results:
        lines.append(
            f"| {r['name']} | {r['num_layers']} | {r['num_heads']} | {r['embed_dim']} | {r['params']:,} | "
            f"{r['val_loss']:.4f} | {r['val_perplexity']:.2f} | {r['bits_per_char']:.4f} | "
            f"{r['top1_accuracy'] * 100:.2f}% | {r['top5_accuracy'] * 100:.2f}% | {r['train_time_sec']:.1f} |"
        )

    lines += [
        f"\n## 2. PEFT: LoRA vs full fine-tuning (base model: {PEFT_BASE_MODEL})\n",
        "| variant | trainable params | % of total | val loss | val ppl | train time (s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in peft_results:
        lines.append(
            f"| {r['name']} | {r['trainable_params']:,} | {r['trainable_pct']:.2f}% | "
            f"{r['val_loss']:.4f} | {r['val_perplexity']:.2f} | {r['train_time_sec']:.1f} |"
        )

    lines.append("\n## 3. KV-cache generation latency\n")
    for name, rows in latency_results.items():
        lines.append(f"\n### {name}\n")
        lines.append("| gen length | no cache (s) | cached (s) | speedup |")
        lines.append("|---|---|---|---|")
        for row in rows:
            lines.append(
                f"| {row['gen_len']} | {row['no_cache_sec']:.4f} | {row['cache_sec']:.4f} | {row['speedup']:.2f}x |"
            )

    env, data, training, peft_meta, latency_meta = (
        metadata["environment"], metadata["data"], metadata["training"],
        metadata["peft"], metadata["latency"],
    )
    lines += [
        "\n## Appendix: run configuration\n",
        f"- **Environment:** {env['device']} | torch {env['torch_version']} | "
        f"Python {env['python_version']} | {env['platform']}",
        f"- **Data:** {data['num_train_stories']:,} train / {data['num_val_stories']:,} val stories "
        f"-> {data['train_tokens']:,} / {data['val_tokens']:,} tokens | vocab {data['vocab_size']} "
        f"({data['num_merges']} merges + `{data['unk_token']}`)",
        f"- **Training (scaling stage):** block_size={training['block_size']}, "
        f"batch_size={training['batch_size']}, lr={training['learning_rate']}, "
        f"max_steps={training['max_steps']}, dropout={training['dropout']}, "
        f"optimizer={training['optimizer']}, gradient_clipping={training['gradient_clipping']}",
        f"  - {training['note']}",
        f"- **PEFT:** max_steps={peft_meta['max_steps']}, base_model={peft_meta['base_model']}, "
        f"LoRA targets={peft_meta['lora_target_modules']}, {peft_meta['lora_alpha_formula']}, "
        f"lora_dropout={peft_meta['lora_dropout']}, {peft_meta['identical_init']}",
        f"- **Latency:** prompt_len={latency_meta['prompt_len']}, "
        f"gen_lengths={latency_meta['gen_lengths']}, num_repeats={latency_meta['num_repeats']}, "
        f"decoding={latency_meta['decoding']}, {latency_meta['synchronized_timing']}, "
        f"peak_memory_measured={latency_meta['peak_memory_measured']}",
    ]

    with open(os.path.join(RESULTS_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nsaved {RESULTS_DIR}/report.md and {RESULTS_DIR}/report_data.json")


def run():
    device = get_device()
    print(f"using device: {device}")

    print("\nbuilding dataset once (shared tokenizer/data across every stage)...")
    tokenizer, train_token_ids, val_token_ids = build_dataset(
        vocab_size=VOCAB_SIZE, num_train=NUM_TRAIN, num_val=NUM_VAL
    )
    train_data = flatten(train_token_ids)
    val_data = flatten(val_token_ids)
    print(f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    unigram_ppl = unigram_baseline_perplexity(train_token_ids, len(tokenizer.vocab))
    print(f"unigram frequency baseline perplexity: {unigram_ppl:.2f}")

    metadata = collect_metadata(device, tokenizer, train_data, val_data)

    scaling_results, models = run_scaling(tokenizer, train_data, val_data, val_token_ids, device)

    base_model = models[PEFT_BASE_MODEL]
    peft_results = run_peft(base_model, train_data, val_data, device)

    latency_results = run_latency(models, device)

    write_report(scaling_results, unigram_ppl, peft_results, latency_results, metadata)


if __name__ == "__main__":
    run()
