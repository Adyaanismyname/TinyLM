# TinyLLM experiment report

Generated 2026-09-15 19:30. Unigram frequency baseline perplexity: 589.63 (reference floor -- every model below should land well under this).

## 0. Experimental controls

What keeps the comparisons below apples-to-apples:

- **One shared tokenizer/dataset.** `build_dataset()` runs once; the same BPE vocab and the same train/val token streams are reused for every model size and every PEFT variant, so tokenization is never a confound between rows.
- **Scaling isolates parameter count.** `num_layers`/`num_heads` are set directly per config; `embed_dim` is solved (`resolve_embed_dim`) to hit each target parameter count. Learning rate, batch size, step count, eval cadence, and dropout are identical across all three sizes -- only capacity varies.
- **Exact, non-sampled evaluation.** Val loss/perplexity/accuracy/bits-per-char are computed over *every* window of the validation set (`metrics.perplexity` / `topk_accuracy`), not random sampled batches like the training-time progress printouts -- so these numbers are deterministic for a given model, not noise from one lucky batch.
- **Two reference floors, not just relative comparisons.** The unigram frequency baseline above is the weakest reasonable model; `baseline_pretrained` in the PEFT table is the base checkpoint with *zero* additional training. Every other row's improvement is measured against actually doing nothing, not just against each other.
- **PEFT rows differ in exactly one variable.** Every variant (`lora_r4/r8/r16`, `full_finetune`) starts from an identical deep copy of the same pretrained base model, trains on identical data, for the identical step budget (`PEFT_MAX_STEPS`). The only thing that changes between rows is which/how many parameters are trainable -- that's what the trainable-param and %-of-total columns are for.
- **KV cache: verified correctness, not just speed.** `model.py`'s cached and non-cached generation paths are checked (in `model.py`'s own smoke test) to produce bit-for-bit identical sampled tokens given the same seed, so the latency numbers below measure pure speed, not a behavior change. Each timing excludes one warmup generation (lazy setup shouldn't count) and averages 3 repeats.

## 1. Scaling: does more parameters actually help?

| model | layers | heads | embed_dim | params | val loss | val ppl | bits/char | top-1 acc | top-5 acc | train time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| small_1M | 4 | 4 | 132 | 992,376 | 3.2508 | 25.81 | 1.3337 | 29.04% | 55.52% | 265.6 |
| medium_1.35M | 6 | 6 | 126 | 1,295,280 | 3.1576 | 23.51 | 1.2954 | 30.82% | 57.37% | 334.0 |
| large_1.7M | 8 | 8 | 128 | 1,730,816 | 3.0827 | 21.82 | 1.2647 | 32.07% | 58.79% | 407.3 |

## 2. PEFT: LoRA vs full fine-tuning (base model: large_1.7M)

| variant | trainable params | % of total | val loss | val ppl | train time (s) |
|---|---|---|---|---|---|
| baseline_pretrained | 0 | 0.00% | 3.0827 | 21.82 | 0.0 |
| lora_r4 | 65,536 | 3.65% | 3.0352 | 20.80 | 123.9 |
| lora_r8 | 131,072 | 7.04% | 3.0288 | 20.67 | 116.8 |
| lora_r16 | 262,144 | 13.15% | 3.0194 | 20.48 | 111.1 |
| full_finetune | 1,730,816 | 100.00% | 2.9738 | 19.57 | 81.8 |

## 3. KV-cache generation latency


### small_1M

| gen length | no cache (s) | cached (s) | speedup |
|---|---|---|---|
| 20 | 0.3736 | 0.4409 | 0.85x |
| 50 | 1.0332 | 1.0784 | 0.96x |
| 100 | 1.8274 | 1.8835 | 0.97x |

### medium_1.35M

| gen length | no cache (s) | cached (s) | speedup |
|---|---|---|---|
| 20 | 0.4483 | 0.4188 | 1.07x |
| 50 | 0.9998 | 1.2627 | 0.79x |
| 100 | 2.7093 | 2.8394 | 0.95x |

### large_1.7M

| gen length | no cache (s) | cached (s) | speedup |
|---|---|---|---|
| 20 | 0.5985 | 0.6179 | 0.97x |
| 50 | 1.7828 | 1.6845 | 1.06x |
| 100 | 3.0879 | 2.5904 | 1.19x |

## Appendix: answers to the write-up's [TO ADD] items

Pulled by inspecting this run's saved checkpoint (`checkpoints/checkpoint_small_1M.pt`, whose tokenizer is shared across every experiment) and this machine's environment. Organized by the section each `\toadd{}` appears in.

**Author / affiliation** -- not something this can answer; fill in directly.

**Dataset (Section: Dataset)**
- Stories used: 8,000 training stories, 2,000 validation stories (`dataset.build_dataset(num_train=8000, num_val=2000)`), taken as a fixed prefix of TinyStories' `train`/`validation` splits (not shuffled).
- Full TinyStories corpus for reference: ~2.12M train stories / ~21.99K validation stories -- this run used ~0.38% of train and ~9.1% of validation.
- Preprocessing (`dataset.preprocess_text`): (1) Unicode NFKC normalization, (2) drop any character that is neither printable nor whitespace, (3) collapse repeated whitespace to a single space, (4) strip leading/trailing whitespace. No truncation, length filtering, or deduplication.

**Tokenization (Section: Tokenization)**
- Implementation: from-scratch, pure-Python BPE (`BPE.py`) -- not a library (no `tokenizers`/`sentencepiece` dependency).
- Exact counts (read from the checkpoint): 89 starting single-character tokens -> 910 learned merges -> 999 learned tokens + 1 reserved `<unk>` token = 1000 total vocab.
- Whitespace/punctuation: no special-casing -- every character, including spaces and punctuation, starts as its own token and is merged purely by pair frequency like any other character.
- Unknown tokens: a `<unk>` token is added after training; `encode()` falls back to it for any token absent from the learned vocab.

**Training setup (Section: Training setup)**
- Context length (block_size): 128
- Batch size: 32
- Optimizer: `torch.optim.AdamW`, PyTorch defaults except `lr` -- betas=(0.9, 0.999), eps=1e-8, **weight_decay=0.01** (never overridden in `train_model()`)
- Learning rate: constant 3e-4, no scheduler
- Training steps: 3,000 steps per model (Experiment 1); step-based, not epoch-based -- each step samples a fresh random block_size window via `get_batch`
- Dropout: 0.1
- Gradient clipping: none (`train_model` never calls `clip_grad_norm_`)
- Random seed: **none set anywhere in the real training path.** No `torch.manual_seed()` call exists outside `model.py`'s own unrelated standalone smoke test. Every run, including this one, used whatever the ambient global RNG state was.

**Hardware and timing methodology (Section: Hardware and timing methodology)**
- Hardware: Apple M1 Pro (MacBook Pro), 16 GB RAM
- Backend: PyTorch MPS (`torch.backends.mps.is_available() == True`)
- Software: PyTorch 2.4.0, Python 3.12.4, macOS 27.0 (build 26A428)
- Synchronization: yes -- `metrics.generation_latency()` calls `torch.mps.synchronize()` right after the untimed warmup generation and again right after the timed repeats, before either clock read, so queued-but-unfinished MPS work never leaks across the timer boundary.
- Prompt/context length for the latency benchmark: 10 tokens (`LATENCY_PROMPT_LEN`), generating 20/50/100 tokens on top -- all well inside the models' block_size=128.
- Decoding strategy: **sampling**, not greedy -- temperature=0.8, top_k=50 (same defaults used everywhere generation happens in this codebase). The prompt itself is a random token sequence, not real text, since the benchmark measures compute cost, not output quality.

**Fair-comparison controls (Section: Fair-comparison controls)**
- `PEFT_MAX_STEPS` (fine-tuning step budget for every Experiment 2 variant, baseline included): 500

**Experiment 2 / LoRA config (Section: Experiment 2 Setup)**
- LoRA target modules: **all four linear layers per transformer block** -- both attention projections (`qkv_proj`, `out_proj`) and both feed-forward linears (`ff.net[0]`, `ff.net[2]`) -- not only query/value.
- LoRA alpha: alpha = 2 x r for every rank tested (r=4->alpha=8, r=8->alpha=16, r=16->alpha=32), so the scaling factor alpha/r is a constant 2.0 across all three ranks -- only capacity (r) varies between them, not the update's magnitude scaling.
- LoRA dropout: 0.0
- Base weights frozen: confirmed by construction -- `add_lora()` sets `requires_grad=False` on every base parameter before wrapping, and `train_model()` only ever optimizes parameters with `requires_grad=True`, so frozen base weights never receive an update in any LoRA variant.
- Optimizer/LR for Experiment 2: identical to Experiment 1 -- same `AdamW`, same constant `LEARNING_RATE=3e-4`.
- Identical initialization: yes -- every variant (`full_finetune` included) is a `copy.deepcopy()` of the exact same trained base-model instance, so all variants start from bit-identical weights.
- Identical data ordering: **no** -- `get_batch()` draws random windows via unseeded `torch.randint` on every call, so no two variants (or two runs) see the same sequence of training batches. Combined with the no-seed point above, the small differences between LoRA ranks include some irreducible run-to-run sampling noise that this run does not quantify.

**Experiment 2 training-time note (Section: Results, "Training time")**
- Independent runs per training-time figure: **1** (single run, no repeats, no seed control). The rank-vs-time ordering (123.9s > 116.8s > 111.1s) is a single-sample measurement per rank -- suggestive, not confirmed stable.

**Experiment 3 (Section: Setup / Limitations)**
- Prompt/context length: 10 tokens (same as above)
- Decoding: sampling (temperature=0.8, top_k=50), not greedy
- Peak memory: **not measured** -- no `torch.mps.current_allocated_memory()` (or equivalent) call exists anywhere in this codebase. This is a real gap, not a value to look up; it needs new instrumentation to answer.

**Limitations section**
- "exact number of stories/documents used relative to the full corpus": see Dataset above (8,000 / ~2.12M train stories; 2,000 / ~21.99K validation stories).
- "exact hardware and PyTorch/MPS version": see Hardware above.
- "confirmation of how many random seeds were used per configuration": zero seeds set, one run per configuration everywhere in this report. Latency's "3 repeats" average multiple generations *within the same run*, not multiple independently-seeded runs.
- "exact implementation details" for BPE: see Tokenization above -- from-scratch, pure-Python, greedy highest-pair-frequency merging (`get_pair_counts` + `max(..., key=...)`), no whitespace/punctuation special-casing, `<unk>` fallback added post hoc.
