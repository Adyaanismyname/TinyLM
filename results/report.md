# TinyLLM experiment report

Regenerated 2026-09-26 from `results/runs/{e1,e2,e3,e4,e5}/*.json` (the per-run
files are the source of truth; this file only summarizes them). The paper
(`tinyLmPaper.latex`) is the full write-up. All runs: NVIDIA GeForce RTX 3050
6GB Laptop GPU, PyTorch 2.5.1, block_size 512, batch 32 (E1–E3) or 16 (E4–E5),
AdamW with warmup+cosine, dropout 0.1, vocab 1,024 (word-level BPE).

Data: TinyStories train 904.5M tokens / dev 4.27M / test 9.17M;
TinyStoriesInstruct train 153.5M / dev 2.60M / test 2.48M (loss masked to
story tokens). Test metrics are computed over fixed prefixes: first 2M test
tokens (E1, E2) and first 1M tokens of each test set (E3).

## E1 — data scaling at a fixed 100M-token budget (1.78M model, 3 seeds)

| unique tokens | epochs | test ppl (mean ± 95% CI) |
|---|---|---|
| 2M | 50.0 | 5.29 ± 0.05 |
| 8M | 12.5 | 5.20 ± 0.03 |
| 32M | 3.1 | 5.18 ± 0.08 |
| 100M | 1.0 | 5.19 ± 0.03 |

Nearly all the benefit of fresh data arrives by 8M unique tokens; even 50
epochs over a 2M pool costs only ~2% perplexity.

## E2 — model scaling (200M tokens, one pass, LR swept per size, 3 seeds)

| layers | width | params | best LR | test loss ± CI | ppl |
|---|---|---|---|---|---|
| 2 | 32 | 74,624 | 3e-3 | 2.322 ± 0.025 | 10.20 |
| 4 | 64 | 298,368 | 3e-3 | 1.702 ± 0.086 | 5.48 |
| 6 | 96 | 818,688 | 3e-3 | 1.348 ± 0.002 | 3.85 |
| 8 | 128 | 1,783,040 | 1e-3 | 1.225 ± 0.003 | 3.40 |
| 10 | 160 | 3,338,880 | 3e-3 | 1.089 ± 0.004 | 2.97 |
| 12 | 192 | 5,633,664 | 1e-3 | 1.045 ± 0.004 | 2.84 |
| 14 | 224 | 8,814,848 | 1e-3 | 0.995 ± 0.005 | 2.70 |

Power-law fit on seed means: L = 0.71 + 103.7·N^−0.37 (bootstrap 95% CI for
the exponent: 0.22–0.57).

## E3 — LoRA vs. full fine-tuning on TinyStoriesInstruct (base: E2 8-layer seed 0; 20M tokens, 5 seeds)

| arm | trainable (% of base) | LR | Instruct ppl | TinyStories ppl (forgetting) | Words rate | train time |
|---|---|---|---|---|---|---|
| baseline (frozen) | 0 | – | 3.650 | 3.398 | 0.0% | 0s |
| lora_r1 | 16,384 (0.9%) | 1e-2 | 3.287 | 3.444 | 0.6% | 306s |
| lora_r2 | 32,768 (1.8%) | 1e-2 | 3.268 | 3.447 | 0.6% | 313s |
| lora_r4 | 65,536 (3.7%) | 1e-2 | 3.256 | 3.449 | 0.6% | 313s |
| lora_r8 | 131,072 (7.3%) | 1e-2 | 3.250 | 3.453 | 1.3% | 312s |
| lora_r16 | 262,144 (14.7%) | 3e-3 | 3.230 | 3.444 | 1.9% | 308s |
| lora_r32 | 524,288 (29.4%) | 3e-3 | 3.221 | 3.447 | 1.7% | 317s |
| lora_r64 | 1,048,576 (58.8%) | 1e-3 | 3.211 | 3.442 | 1.8% | 323s |
| full fine-tune | 1,783,040 (100%) | 3e-4 | 3.201 | 3.474 | 1.4% | 254s |

LoRA r16 closes ~94% of the baseline→full-FT perplexity gap with 14.7% of the
trainable parameters; full fine-tuning wins perplexity but forgets the most.
Words-constraint following barely moves for any arm (reference-story ceiling:
92.8%). LoRA is 20–27% slower in wall-clock at this width — see E4. LoRA arms
were merged (`merge_lora`) before evaluation, so final architectures are
identical to full fine-tuning.

## E4 — LoRA vs. full fine-tuning step time vs. width (8 layers, batch 16, 200 timed steps × 5 repeats, random weights/tokens)

| width | params | full | lora_r8 | lora_r64 |
|---|---|---|---|---|
| 64 | 0.50M | 0.108s | 0.121s | 0.127s |
| 128 | 1.78M | 0.218s | 0.238s | 0.248s |
| 256 | 6.71M | 0.765s | 0.742s | 0.765s |
| 512 | 26.0M | 1.046s | 1.157s | 1.277s |
| 1024 | 102.3M | 5.040s | 2.737s (1.84x) | 3.061s (1.65x) |

LoRA is slower below the crossover (~width 512) and clearly faster at width
1024. Repeat-to-repeat thermal drift is visible at width 1024; medians are
reported.

## E5 — KV-cache latency (1.78M model, greedy, 16-token prompt, 10 repeats, random weights/tokens)

Batch 16, median seconds (speedup vs. no cache):

| gen tokens | no cache | concat | prealloc |
|---|---|---|---|
| 20 | 0.140 | 0.139 (1.01x) | 0.162 (0.86x) |
| 50 | 0.342 | 0.257 (1.33x) | 0.320 (1.07x) |
| 100 | 0.588 | 0.622 (0.94x) | 0.722 (0.81x) |
| 250 | 1.911 | 1.795 (1.06x) | 1.437 (1.33x) |
| 490 | 6.348 | 3.428 (1.85x) | 3.390 (1.87x) |

Batch 1 shows no reliable benefit at any length tested. Peak memory at batch
16: no-cache grows 30→108MB with generation length; prealloc constant 93.4MB
(KV tensors: 66MB at 490 tokens).

## Samples

`results/samples/samples.md` has unfiltered generations from the E3 seed-0
models (recreated checkpoints reproduce their stored Instruct perplexities to
three decimals). Headline: fine-tuned arms adopt the story genre and pick up
prompt entities, but explicit word constraints are mostly ignored — matching
the Words-rate numbers above.

## Known limitations of this data

- E1/E2 test metrics cover 2M of 9.17M test tokens; E3 covers 1M per test set.
- E1 logged against the test split during training (nothing selected on it).
- E1 pools are prefixes of the train file, and E1's LR was fixed at 3e-4.
- E4/E5 measure compute with random weights/tokens on one laptop GPU.
- ~5.6% of TinyStories validation stories contain a mangled-quote artifact
  inherited from the source data.
