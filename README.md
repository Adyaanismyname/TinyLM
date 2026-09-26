# TinyLLM

A from-scratch, decoder-only GPT-style language model with a custom BPE tokenizer, trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct). Built to run controlled experiments on questions that are well studied at large model scale but rarely tested at ~1–10M parameters:

- **E1 — Data scaling**: at a fixed token budget, does more *unique* data beat repeating a smaller pool?
- **E2 — Model scaling**: does more capacity help, over two orders of magnitude in parameter count?
- **E3 — PEFT**: how does LoRA compare to full fine-tuning, matched by *final* parameter count, on a real fine-tuning task (adapting to TinyStoriesInstruct)?
- **E4 — Training speed**: at what model width, if any, does LoRA actually train faster than full fine-tuning?
- **E5 — KV caching**: does it speed up generation for models this small, and does a write-in-place cache change the answer?

Everything is implemented from scratch on top of PyTorch — no `transformers`, no existing tokenizer library. Full write-up, methodology, and results: [`results/report.md`](results/report.md).

## What's here

| File | What it is |
|---|---|
| `BPE.py` | From-scratch, word-level byte-pair-encoding tokenizer (training + encode/decode), with a per-word cache and deterministic token IDs |
| `dataset.py` | Loads + preprocesses TinyStories and TinyStoriesInstruct into plain Python structures (no tokenizer dependency) |
| `prepare_data.py` | One-time pipeline: downloads both datasets, trains the tokenizer, writes memmap `.bin` files + `tokenizer.json` under `data/` |
| `data.py` | `TokenStream`/`EpochBatcher` — seeded, epoch-based batch sampling over memmap or in-memory token streams, with optional per-token loss masking |
| `model.py` | The GPT model — causal self-attention (manual or `scaled_dot_product_attention`), pre-norm blocks, weight tying, and three interchangeable generation paths (no cache, growing cache, preallocated cache) |
| `lora.py` | LoRA (`LoRALinear`) plus `merge_lora`, which folds a trained adapter back into plain weights so a LoRA-fine-tuned model has the same architecture as a fully fine-tuned one |
| `train.py` | Core training loop (`train_model`) — seeding, warmup+cosine LR schedule, gradient clipping — reused by every experiment script |
| `metrics.py` | Evaluation: perplexity/BPC (optionally loss-masked), top-k accuracy, the Words-constraint-rate metric, generation latency, and peak-memory helpers |
| `run_experiments.py` | Entry points for E1/E2/E3 (`python3 run_experiments.py e1\|e2\|e3 [--quick]`) |
| `bench_train.py` | E4: standalone LoRA-vs-full-fine-tuning training-speed benchmark |
| `bench_kvcache.py` | E5: standalone KV-cache latency/memory benchmark across cache implementations |
| `generate.py` | Generate text from any saved checkpoint (`python generate.py --checkpoint ... --prompt "..."`) |
| `generate_samples.py` | Recreates E3's seed-0 fine-tuned models and writes side-by-side sample stories to `results/samples/` |
| `analysis.py` | Aggregates per-run JSONs into means + 95% CIs and fits E2's power law |
| `generate_figures.py` | Renders the figures in `results/figs/` from `results/runs/` |
| `tests/` | pytest suite covering the tokenizer, dataloader, model, LoRA merging, training, metrics, and each pipeline script end-to-end at small scale |

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Tested on PyTorch 2.5 with CUDA (RTX 3050) and on Apple Silicon (MPS); also works on CPU.

## Usage

Run the test suite (fast — small synthetic/tiny-real-data cases, not the real experiments):

```bash
python3 -m pytest tests/
```

Prepare the data once (downloads TinyStories + TinyStoriesInstruct, trains the tokenizer, writes `data/*.bin`):

```bash
python3 prepare_data.py
```

This is a long-running job (the full TinyStories train split alone encodes to ~900M tokens) — run it in the background. Use `--tinystories-train-limit N` etc. for a fast, small-scale smoke test.

Run an experiment. Always try `--quick` first on a machine you haven't run this on before — it exercises the whole pipeline (LR selection, multi-seed training, evaluation, JSON output) in well under a minute:

```bash
python3 run_experiments.py e1 --quick   # then, for real: python3 run_experiments.py e1
python3 run_experiments.py e2 --quick
python3 run_experiments.py e3 --quick   # needs e2's layers=8 checkpoint, or trains a tiny one itself in --quick
```

Run the training-speed and KV-cache benchmarks (do this separately on every machine/backend you want compared — that's the point of E4/E5):

```bash
python3 bench_train.py --quick     # then: python3 bench_train.py
python3 bench_kvcache.py --quick   # then: python3 bench_kvcache.py
```

Regenerate the figures from whatever's in `results/runs/`:

```bash
python3 generate_figures.py
```

See what the models actually write. `run_experiments.py` reports numbers only (and saves just the 8-layer E2 base model), so `generate_samples.py` rebuilds the E3 seed-0 models — same learning rate, steps and data order, checked against the stored perplexities — and generates from each on the same prompts:

```bash
python3 generate.py --checkpoint checkpoints/e2_base_layers8_seed0.pt --prompt "Once upon a time" -n 3
python3 generate_samples.py        # ~20 min: re-fine-tunes 4 arms, writes results/samples/samples.md
```

Evaluate a trained checkpoint:

```bash
python3 metrics.py                 # full report
python3 metrics.py --kv-cache      # KV-cache latency comparison only
```

## Results

The full results, aggregated from `results/runs/*/*.json` (means + 95% CIs over seeds, E2 power-law fit), are in [`results/report.md`](results/report.md); the paper-length write-up is [`tinyLmPaper.latex`](tinyLmPaper.latex). Headlines:

- **E1 (data):** at a fixed 100M-token budget, unique data past ~8M tokens (12.5 epochs of repetition) buys nothing measurable; even 50 epochs of repetition costs only ~2% perplexity.
- **E2 (model):** test loss falls monotonically from 75K to 8.8M parameters, well fit by $L = 0.71 + 103.7\,N^{-0.37}$.
- **E3 (LoRA vs. full FT):** LoRA r16 closes ~94% of the perplexity gap with 14.7% of the trainable parameters and forgets less; no arm learns explicit word-constraint following (≤1.9% vs. a 92.8% reference ceiling); LoRA trains ~20–27% slower at this width.
- **E4 (speed crossover):** LoRA is slower per step below ~width 512 and 1.84x faster at width 1024 (102M params).
- **E5 (KV cache):** no reliable speedup ≤100 generated tokens or at batch 1; 1.85–1.87x at batch 16 with 490 tokens, where a preallocated cache also caps peak memory.

## License

MIT — see [LICENSE](LICENSE).
