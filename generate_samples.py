"""
Qualitative samples for Experiment 3: what do the fine-tuned models actually
write?

run_experiments.py e3 measures perplexity and the Words-constraint rate but
keeps neither the generated text nor the fine-tuned models, so this script
recreates the seed-0 model of each requested arm -- same base checkpoint,
same learning rate and step count (read from results/runs/e3/<arm>_seed0.json),
same data order -- saves it under checkpoints/, and generates from it.

    python generate_samples.py                      # base + lora_r1/r8/r64 + full_finetune
    python generate_samples.py --arms lora_r16      # just one more arm
    python generate_samples.py --num-prompts 10

Each recreated model's Instruct perplexity is compared with the number stored
in its E3 JSON, so a report built from these samples can say the models are
the ones the E3 table describes. Fine-tuned checkpoints are reused on later
runs (delete them, or pass --retrain, to rebuild).

Writes results/samples/samples.md (readable) and samples.json (raw).
"""

import argparse
import json
import os

import torch

from data import stream_from_bin
from generate import sample
from lora import merge_lora
from metrics import _word_present, load_checkpoint, perplexity
from run_experiments import (
    BLOCK_SIZE, CHECKPOINTS_DIR, DATA_DIR, GRAD_CLIP, RESULTS_DIR, WEIGHT_DECAY,
    bin_path, build_e3_variant, load_base_checkpoint,
)
from train import device_description, get_device, seed_everything, train_model

DEFAULT_BASE_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "e2_base_layers8_seed0.pt")
DEFAULT_ARMS = ["lora_r1", "lora_r8", "lora_r64", "full_finetune"]
OUT_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "samples")

# Free-form story openings for the plain pretrained model.
STORY_PROMPTS = [
    "Once upon a time, there was a little girl named Mia.",
    "Tom and his dog went to the park.",
    "One day, a big bird landed on the window.",
    "",  # empty: the model starts a story on its own
]

TEMPERATURE = 0.8
TOP_K = 50
MAX_NEW_TOKENS = 300


def log(msg):
    print(f"[generate_samples] {msg}", flush=True)


def arm_config(arm):
    """The E3 seed-0 run's own record of how this arm was trained."""
    path = os.path.join(RESULTS_DIR, "e3", f"{arm}_seed0.json")  # RESULTS_DIR is already results/runs
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run `run_experiments.py e3` first (or pick a different --arms)")
    with open(path) as f:
        return json.load(f)


def finetune_arm(base_model, arm, run, device, streams, log_prefix):
    """
    Recreates run_e3's seed-0 training of one arm. Mirrors that code on
    purpose rather than refactoring it, so the numbers in the E3 table keep
    coming from untouched code. Two details matter for reproducing it:
    the eval cadence (eval draws batches from the same batcher that trains,
    so a different eval_interval would change which data each step sees) and
    seeding before the LoRA adapters are created (their init consumes RNG).
    """
    cfg = run["config"]
    steps, batch_size, lr = cfg["max_steps"], cfg["batch_size"], cfg["learning_rate"]

    seed_everything(0)
    model = build_e3_variant(base_model, cfg["mode"], cfg.get("r"))
    train_model(
        model, streams["train"], streams["dev"], device,
        block_size=BLOCK_SIZE, batch_size=batch_size, learning_rate=lr,
        warmup_steps=max(1, steps // 10), max_steps=steps,
        eval_interval=max(1, steps // 10), eval_batches=10,
        weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, seed=0, log_prefix=log_prefix,
    )
    if cfg["mode"] == "lora":
        model = merge_lora(model)  # same architecture as the base / full fine-tune
    return model


def instruct_ppl(model, stream, batch_size, device):
    _, ppl = perplexity(model, stream.data[:1_000_000], BLOCK_SIZE, batch_size, device,
                        mask=stream.mask[:1_000_000])
    return ppl


def get_arm_model(arm, base_model, base_ckpt_config, tokenizer, device, streams, retrain):
    run = arm_config(arm)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"e3_{arm}_seed0.pt")

    if os.path.exists(ckpt_path) and not retrain:
        log(f"{arm}: loading saved model {ckpt_path}")
        model, _, _ = load_checkpoint(ckpt_path)
    else:
        log(f"{arm}: fine-tuning (lr={run['config']['learning_rate']}, steps={run['config']['max_steps']})")
        model = finetune_arm(base_model, arm, run, device, streams, log_prefix=f"[{arm}] ")
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(), "tokenizer": tokenizer.to_dict(),
            "config": base_ckpt_config,
        }, ckpt_path)
        log(f"{arm}: saved {ckpt_path}")

    model.eval()
    ours = instruct_ppl(model, streams["test"], run["config"]["batch_size"], device)
    stored = run["final_metrics"]["instruct_val_perplexity"]
    log(f"{arm}: instruct perplexity {ours:.3f} here vs {stored:.3f} in the E3 JSON")
    return model, {"instruct_ppl_here": ours, "instruct_ppl_e3_json": stored}


def build_report(story_samples, instruct_samples, arms, checks, device):
    lines = [
        "# Sample generations",
        "",
        f"Generated on {device_description(device)}. Sampling: temperature {TEMPERATURE}, top-k {TOP_K}, "
        f"up to {MAX_NEW_TOKENS} new tokens, one sample per prompt. **Samples are unfiltered** -- "
        "nothing was picked or edited, so they include the model's odd word choices.",
        "",
        "Every model is the 1.78M-parameter 8-layer model from Experiment 2 (seed 0), optionally "
        "fine-tuned on TinyStoriesInstruct as in Experiment 3 (seed 0, same learning rate, steps and "
        "data order as the E3 run). Each fine-tuned model reproduces its E3 Instruct perplexity:",
        "",
        "| arm | Instruct ppl (this model) | Instruct ppl (E3 JSON) |",
        "|---|---|---|",
    ]
    for arm in arms:
        c = checks[arm]
        lines.append(f"| {arm} | {c['instruct_ppl_here']:.3f} | {c['instruct_ppl_e3_json']:.3f} |")

    lines += ["", "## 1. The pretrained model continuing story openings", ""]
    for prompt, text, hit_eos in story_samples:
        shown = f'"{prompt}"' if prompt else "*(empty prompt: model starts its own story)*"
        lines += [f"**Prompt:** {shown}", "", f"> {(prompt + text if prompt else text).strip()}", ""]
        if not hit_eos:
            lines += ["*(stopped at the token limit)*", ""]

    lines += [
        "## 2. Following an instruction: header in, story out", "",
        "Each prompt is a TinyStoriesInstruct header (features / words / summary) ending in `Story:`. "
        "A model following the instruction should write a story that uses the listed **Words**. "
        "For each output, the words it actually used are listed (inflections like -s/-ed/-ing count).", "",
    ]
    for i, item in enumerate(instruct_samples, 1):
        lines += [f"### Prompt {i}", "", f"> {item['header']}", "", f"Target words: **{', '.join(item['words'])}**", ""]
        for model_name in ["baseline_frozen"] + arms:
            out = item["outputs"][model_name]
            used = ", ".join(out["words_used"]) or "none"
            label = "no fine-tuning" if model_name == "baseline_frozen" else model_name
            lines += [f"**{label}** -- words used: {used}", "", f"> {out['text'].strip()}", ""]
    return "\n".join(lines) + "\n"


def run(arms, num_prompts, base_checkpoint, retrain, out_dir):
    device = get_device()
    log(f"device: {device_description(device)}")

    base_model, tokenizer = load_base_checkpoint(base_checkpoint, device)
    base_model.eval()
    base_ckpt_config = torch.load(base_checkpoint, map_location="cpu")["config"]

    streams = {
        "train": stream_from_bin(bin_path("instruct_train"), BLOCK_SIZE, mask_path=bin_path("instruct_train_mask")),
        "dev": stream_from_bin(bin_path("instruct_dev"), BLOCK_SIZE, mask_path=bin_path("instruct_dev_mask")),
        "test": stream_from_bin(bin_path("instruct_test"), BLOCK_SIZE, mask_path=bin_path("instruct_test_mask")),
    }
    with open(os.path.join(DATA_DIR, "words_eval_prompts.json")) as f:
        prompts = json.load(f)[:num_prompts]

    models = {"baseline_frozen": base_model}
    checks = {}
    for arm in arms:
        models[arm], checks[arm] = get_arm_model(
            arm, base_model, base_ckpt_config, tokenizer, device, streams, retrain
        )

    log("generating story continuations from the pretrained model...")
    story_samples = []
    for i, prompt in enumerate(STORY_PROMPTS):
        text, hit_eos = sample(base_model, tokenizer, prompt, device, MAX_NEW_TOKENS, TEMPERATURE, TOP_K, seed=100 + i)
        story_samples.append((prompt, text, hit_eos))

    log("generating from every model on the Instruct prompts...")
    instruct_samples = []
    for i, (header, words) in enumerate(prompts):
        outputs = {}
        for name, model in models.items():
            # same seed for every model on a given prompt, so differences come
            # from the models rather than from a different random stream
            text, _ = sample(model, tokenizer, header, device, MAX_NEW_TOKENS, TEMPERATURE, TOP_K, seed=1000 + i)
            outputs[name] = {"text": text, "words_used": [w for w in words if _word_present(w, text.lower())]}
        instruct_samples.append({"header": header, "words": words, "outputs": outputs})

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump({
            "device": device_description(device), "temperature": TEMPERATURE, "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS, "arms": arms, "reproduction_check": checks,
            "story_samples": [{"prompt": p, "text": t, "hit_eos": e} for p, t, e in story_samples],
            "instruct_samples": instruct_samples,
        }, f, indent=2)
    with open(os.path.join(out_dir, "samples.md"), "w", encoding="utf-8") as f:
        f.write(build_report(story_samples, instruct_samples, arms, checks, device))
    log(f"wrote {out_dir}/samples.md and {out_dir}/samples.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--retrain", action="store_true", help="Rebuild fine-tuned models even if saved ones exist.")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    run(args.arms, args.num_prompts, args.base_checkpoint, args.retrain, args.out_dir)
