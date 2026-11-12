# Camera-ready re-analysis (no retraining)

Everything here was computed from the existing token files, run JSONs and
saved checkpoints, in the original environment (Windows conda env `pytorch`:
Python 3.11.13, PyTorch 2.5.1, RTX 3050 6GB). No model was retrained.

Reproduce from the repo root, in this order:

```
python -B camera_ready/contamination.py     # ~70 min, CPU (n-gram scan of 1.06B tokens)
python -B camera_ready/memorization.py      # ~3 min, GPU
python -B camera_ready/clean_eval.py        # ~5 min, GPU
python -B camera_ready/e1_e2_impact.py      # seconds
python -B camera_ready/words_matched.py     # ~35 min, GPU (150-token protocol)
python -B camera_ready/words_matched.py --budget 0   # supplementary: fill the context
python -B camera_ready/runtime_checks.py    # ~1 min, GPU
python -B camera_ready/bench_mem_check.py   # ~25 min, GPU must be otherwise idle
```

Outputs are in `camera_ready/out/`.

Sanity checks that passed:
* The batch-order reconstruction matches the real `data.EpochBatcher`
  exactly (checked for E1, E2 and E3 configurations).
* Every recomputed "all" perplexity reproduces the stored JSON value. The
  largest difference is 1.6e-5 nats.
* Regenerating the seed-0 Words outputs reproduces the stored hit counts
  exactly: 0, 3, 8, 8 and 7.

---

## 1. Test-set overlap

### 1a. What overlaps

Duplicates were identified by exact token-sequence hashes of whole stories.
Near-duplicates are stories that share at least one 100-token (or 50-token)
span with either training file.

| Evaluated set | Stories | Exact copy in TinyStories train | Exact copy in Instruct train | Shares a 100-token span | Shares a 50-token span |
|---|---|---|---|---|---|
| E1/E2: first 2M TinyStories test tokens | 5,161 | 31.4% of stories / 33.1% of tokens | 12.6% / 13.5% | 43.0% / 46.5% | 53.2% / 55.9% |
| E3 forgetting: first 1M TinyStories test tokens | 2,543 | 33.3% / 35.1% | 13.3% / 13.4% | 39.2% / 42.2% | 49.1% / 51.7% |
| E3: first 1M Instruct test tokens | 2,066 | 45.8% / 44.3% | 0 / 0 | 70.3% / 72.3% | 80.4% / 82.0% |

The overlap comes from the released dataset itself, not from this project's
pipeline:
* 15% of TinyStories training stories duplicate another training story.
* Every duplicated TinyStories test story has exactly one copy in the
  training file.
* The Instruct stories that share spans without being exact copies are
  mostly TinyStories stories with a summary appended (see 3b).

### 1b. Which copies each model actually trained on

This is reconstructed from the seeded batch order. A story counts as
*exposed* if any of its copy's tokens were training targets, and as *≥1
pass* if the copy received the equivalent of at least one full pass.
Percentages are of scored test tokens.

| Run | Any exposure | ≥1 pass |
|---|---|---|
| E1, 2M / 8M / 32M pools (all seeds) | **0** | 0 |
| E1, 100M pool (seeds 0/1/2) | 0.85% (50 stories) | 0.78–0.85% |
| E2, seed 0/1/2 (identical for all 7 sizes within a seed) | 12.7 / 12.7 / 12.5% | 3.0 / 2.9 / 2.7% |
| E3 base (E2 L8 seed 0) on the Instruct test | 16.7% | 4.3% |
| E3 base on the TinyStories test (first 1M) | 13.8% | 3.0% |
| E3 fine-tuning, seeds 0–4, on the Instruct test | 0 | 0 |
| E3 fine-tuning, seeds 0–4, on the TinyStories test (via Instruct copies) | 2.5–3.6% | 0.3–0.5% |

### 1c. Does training on a copy lower that story's loss?

This uses the 8-layer seed-0 model on the whole test sets. Within the
duplicated stories, whether a story was trained on was decided by the random
window shuffle, so this is a natural experiment. It is stratified to stories
with a single copy. The table shows the loss difference (exposed minus
unexposed, nats per token) with a story-bootstrap 95% CI.

| Test set | ≥1 pass vs. unexposed | Any exposure vs. unexposed |
|---|---|---|
| TinyStories (597 vs. 4,065 stories) | +0.003 [−0.013, +0.020] | +0.004 [−0.005, +0.013] |
| Instruct (179 vs. 1,341 stories) | −0.031 [−0.060, −0.001] | −0.014 [−0.030, +0.004] |

There is no detectable benefit on TinyStories. On Instruct stories there is a
small benefit of at most a few hundredths of a nat.

### 1d. Re-evaluation of the saved models on clean subsets (seed 0)

The subsets used below:
* **not exposed:** drops only the stories whose copy the evaluated model
  trained on.
* **no exact copy:** drops every story with a copy in either training file.
* **strict:** additionally drops every story that shares a 100-token (or
  50-token) span with either training file.

**E2 8-layer seed-0 model, E1/E2 test window (PPL):**

| all | not exposed | no exact copy | strict-100 | strict-50 |
|---|---|---|---|---|
| 3.399 | 3.382 | 3.354 | 3.357 | 3.475 |

The subsets differ because of their composition, not because of
memorization:
* Duplicated stories are *harder* than average.
* The strict-50 subset has 10× more stories with the garbled `â€` sequence
  (7.1% vs. 0.7%), and those stories cost +0.126 nats/token.

**E3 PPL-gain recovery (seed 0):**

| Subset (Instruct test) | r=1 | r=8 | r=64 | Base PPL → full-FT PPL |
|---|---|---|---|---|
| all (reproduces stored) | 81.1% | 89.0% | 97.6% | 3.650 → 3.202 |
| not exposed (1,721 stories) | 81.3% | 89.2% | 97.5% | 3.716 → 3.211 |
| no exact copy (1,119) | 81.6% | 89.5% | 97.2% | 3.835 → 3.177 |
| strict-100 (613) | 77.1% | 86.3% | 96.5% | 3.337 → 2.843 |
| strict-50 (406) | 73.8% | 84.5% | 96.1% | 3.336 → 2.863 |

**E3 forgetting (TinyStories PPL increase over the base model, seed 0):**

| Subset | r=1 | r=8 | r=64 | Full FT |
|---|---|---|---|---|
| all | +0.047 | +0.054 | +0.044 | +0.077 |
| not exposed | +0.047 | +0.054 | +0.043 | +0.077 |
| no exact copy | +0.048 | +0.055 | +0.042 | +0.078 |
| strict-100 | +0.049 | +0.056 | +0.044 | +0.081 |
| strict-50 | +0.055 | +0.064 | +0.054 | +0.101 |

Removing the stories the models trained on changes recovery by under 0.3
points, and the forgetting ranking is unchanged on every subset. Low-rank
recovery does depend on *which* stories are scored: r=1 ranges from 74% to
82%. r=16, the abstract's 94%, was not saved; it falls between r=8 (84–90%)
and r=64 (96–98%) on every subset.

### 1e. Experiments 1 and 2 (models not saved)

These use the least favourable 95% bound from 1c: up to 0.013 nats for ≥1
pass and 0.005 nats for partial exposure.

* **E1:**
  * The 2M, 8M and 32M pools contain *none* of the evaluated test stories.
  * The 100M pool's worst-case bias is 0.0001 nats (0.01% PPL).
  * E1 is unaffected.
* **E2:**
  * The worst-case bias for the 8-layer model is 0.0009 nats (0.09% PPL).
  * Every size within a seed trained on the same windows, so exposure is
    identical across sizes.
  * Correcting the fit for that bias:

    | Bias assumption | α |
    |---|---|
    | Reported | 0.371 (CI 0.22–0.57) |
    | Bias constant across sizes | 0.371 |
    | Bias ∝ N | 0.374 |
    | Bias ∝ N² (0.022 nats at 8.8M) | 0.388 |

  * Moving α by 0.01 needs 15× the 8-layer worst case at 8.8M (0.0135 nats).

**Exact clean per-size losses need retraining.** E2 models other than L8
seed 0 were not saved. All 21 runs would take about 19 h; L14 seed 0 alone,
which would measure memorization at the largest size, takes about 2.3 h. Not
needed for the conclusions above.

---

## 2. Words: like-for-like comparison

* **Prompt matching:**
  * All 500 prompts were matched to their own reference example.
    `prepare_data.build_words_eval_prompts` is deterministic and was
    reconstructed.
  * The previously unmatched prompt (index 38) contains `’`, which the
    tokenizer maps to `<unk>`. Its reference is test example 301.
  * The models were scored on these same 500 prompts.
* **Matcher:** unchanged (`metrics._word_present`).
* **Reference at 150 tokens:** the first 150 story tokens, the same budget
  the models get, contain all three words in **230/500 = 46.0%** (Wilson 95%
  41.7–50.4). With 300 tokens: 82.4%. Full stories: 92.8%. Mean reference
  length is 404 tokens.

**Model rates, 150 tokens, stored, 5 seeds:**

| Method | Hits per seed (of 500) | Mean |
|---|---|---|
| Baseline | 0 0 0 0 0 | 0.0% |
| r=1 | 3 1 4 2 5 | 0.6% |
| r=2 | 1 4 5 3 1 | 0.56% |
| r=4 | 5 2 4 2 3 | 0.64% |
| r=8 | 8 3 5 3 13 | 1.28% |
| r=16 | 9 7 8 12 11 | 1.88% |
| r=32 | 10 15 6 6 6 | 1.72% |
| r=64 | 8 7 9 10 12 | 1.84% |
| Full FT | 7 8 6 8 5 | 1.36% |

Every method falls far below the matched reference, with McNemar p < 1e-60
for every seed-0 model against the reference.

**Partial compliance (seed-0 outputs, now saved in
`out/words_outputs_seed0_150.json`):**

| | Base | r=1 | r=8 | r=64 | Full FT | Reference@150 |
|---|---|---|---|---|---|---|
| Mean words present (of 3) | 0.35 | 0.48 | 0.68 | 0.82 | 0.76 | 2.29 |
| ≥1 word | 32.2% | 41.0% | 55.2% | 62.6% | 59.6% | 97.8% |
| ≥2 words | 3.2% | 6.8% | 11.4% | 18.0% | 15.4% | 85.2% |
| Word appears only in `Words:` (not in summary) | 4.9% | 7.8% | 13.1% | 17.8% | 13.4% | 68.8% |
| Stopped at EOS within 150 | 62% | 0% | 0% | 0% | 0.2% | – |

The 15 seed-0 successes of r=64 and full FT were audited by hand, and all are genuine matches.

**Supplementary: does the 150-token budget explain the low rates?** This
re-runs seed 0 with the budget filling the 512-token context (about 420 new
tokens; KV-cache generation), and gives the references the same per-prompt
budget.

| | Base | r=1 | r=8 | r=64 | Full FT | Reference |
|---|---|---|---|---|---|---|
| All three words | 0.2% | 0.6% | 1.8% | 4.4% | 3.8% | 88.6% |
| Mean words present (of 3) | 0.41 | 0.57 | 0.85 | 1.10 | 1.05 | 2.87 |
| Stopped at EOS | 93% | 67% | 71% | 63% | 69% | – |

No, it doesn't. Given room, the fine-tuned models usually end their stories,
and still rarely use all three words. The budget explains the reference rate
(46% → 89%) but not the models' rates.


---

## 3. Instruction-following setup (numbers checked on the windows each E3 run actually trained on)

| | Seeds 0–4 |
|---|---|
| Supervised tokens whose 512-token window contains their example's `Words:` field | 26.7–27.0% |
| Training examples with no `Words:` field at all | 32.6% |
| Epochs of the Instruct stream per run | 0.130 |
| Supervised tokens per run | 16.4M (of ~20M input positions) |

**New finding:**
* In 29.9% of Instruct training examples (30.3% of test examples), the
  dataset places `Summary:` *after* the story.
* The parser keeps it as story text, so it is loss-counted (5.7% of scored
  training tokens; 6.1% of scored test tokens).
* The paper's "the training loss is computed only on the story" is therefore
  inaccurate.
* This likely explains the observed failure mode of stories continuing into
  a new `Summary:` header: the models were trained to write one.

---

## 4. Suggested manuscript text (replace or insert)

**Abstract, Words sentence:**
> However, explicit word constraints were rarely satisfied under our
> fine-tuning setup (0.6% to 1.9% of prompts, compared with 46.0% for
> reference stories truncated to the same 150-token length).

**Abstract, KV-cache sentence:**
> … reaches 1.85× (concatenating cache; 1.87× preallocated) at batch size 16
> with 490 tokens.

**Abstract, LoRA speed:**
> LoRA also trains more slowly than full fine-tuning at this width; at the
> largest width tested (1024) it is only 1.11× faster once GPU memory
> pressure is removed.

**Discussion and Conclusion:** remove "at which point the same code ran
1.84× faster" and "its speed advantage appeared only between embedding
widths 512 and 1024". Present only the KV-cache result as a clear
scale-dependent crossover.

**Section III, Data paragraph (replace):**
> We use the Hugging Face releases of TinyStories [1] and TinyStoriesInstruct.
> From the 2,119,719 TinyStories training stories we hold out a random 10,000
> (seed 0; 4.27M tokens) for development and train on the remaining 2,109,719
> (904.5M tokens); the full official validation split (21,990 stories, 9.17M
> tokens) is our test set. For TinyStoriesInstruct we load the first 300,000
> training examples, drop the 7 with an empty story, hold out a random 5,000
> for development (2.60M tokens) and train on the remaining 294,993 (153.5M
> tokens); the test set is the first 5,000 examples of the official
> validation split (4,999 after dropping one boundary fragment; 2.48M
> tokens). Each Instruct example is rendered as a header holding whichever of
> the Features:, Words:, Summary: and Random sentence: fields it has (67% have
> a Words: field), followed by Story:. In about 30% of examples the dataset
> places the Summary after the story; we keep it as part of the story text,
> so it is included in the loss (5.7% of loss-counted training tokens). The
> loss is computed only on text after Story:. We apply NFKC normalization,
> remove non-printable characters and collapse whitespace, and we do not
> deduplicate.

**Section III, new paragraph "Test-set overlap" (insert after Data):**
> TinyStories contains verbatim duplicates across its official splits: 33% of
> the evaluated TinyStories test tokens and 44% of the evaluated Instruct
> test tokens belong to stories that also occur in the TinyStories training
> split. Reconstructing each run's seeded batch order, the E1 pools of 2M–32M
> tokens contain none of these stories and the 100M pool 0.85% of evaluated
> tokens; each E2 run trained on copies of stories covering 12.5–12.7% of
> evaluated tokens (2.7–3.0% for at least one full pass). For the 8-layer
> model, duplicated test stories it trained on have the same loss as those
> it did not (+0.003 nats/token, 95% CI −0.013 to +0.020; Instruct: −0.031,
> CI −0.060 to −0.001). Re-scoring the saved models without the stories they
> trained on changes LoRA's recovery by under 0.3 percentage points and
> leaves the forgetting ranking unchanged; the overlap lowers the 8-layer E2
> test loss by at most 0.0009 nats, which would have to be 15 times larger at
> 8.8M parameters to move the fitted exponent by 0.01.

**Tokenizer:** replace "Because every experiment uses this tokenizer, it
cannot explain differences between conditions" with:
> Tokenization is held fixed across all conditions, so conclusions are
> conditional on it.

**Evaluation and timing:**
* Add "95% confidence intervals use the Student t distribution over seeds".
* Replace "so that thermal or scheduling drift cannot favor any condition"
  with:
  > to reduce bias from thermal or scheduling drift

**Section IV (E1):** delete "they come nowhere near memorizing even heavily
repeated data" and replace with:
> repetition produced only a small held-out perplexity penalty under this
> schedule (memorization was not measured directly).

**Section V (E2):** replace "smoothly and predictably … with no sign of a
floor" with:
> decreases with every increase in size, with no plateau over the evaluated
> range

Add:
> The learning rate selected for four of the seven sizes (3×10⁻³) is the
> largest in the grid.

**Section VI (E3), body:**
* Replace "Reference stories reach 92.8% on this measure" with:
  > Reference stories truncated to the same 150-token budget contain all
  > three words for 46.0% of prompts (92.8% for complete stories).
* Replace "far below the 92.8% ceiling … within noise" with:
  > far below the 46.0% reached by reference stories of the same length. At
  > 1–15 successes per run, the ranking between methods is not resolved.
* Add on learning rates:
  > For ranks 1–8 the selected rate, 10⁻², is the largest in the grid and
  > development loss was still decreasing, so these ranks may be under-tuned.
* Replace "taught the models the style of the task but not how to follow
  explicit constraints" with the Task-3 text below.

**Table II caption:**
> Means over 5 seeds; 95% t-intervals are at most 0.0033 for perplexity and
> 1.05 percentage points for Words. Words: 500 prompts, up to 150 generated
> tokens. Time: wall-clock for 1,220 steps including data loading and 11
> in-loop evaluations (training steps alone: 240 s for full fine-tuning,
> 283–303 s for LoRA).

**Task 3, narrowed conclusion (Section VI and Conclusion):**
> Under our fine-tuning setup, explicit constraint satisfaction remained
> rare: fine-tuned models used all three requested words for 0.6–1.9% of
> prompts, against 46.0% for reference stories of the same length, although
> a word listed only in the Words: field appeared more often after
> fine-tuning (4.9% for the base model, 8–18% after fine-tuning; seed 0). The
> setup gave limited opportunity to learn the constraint: training used fixed
> 512-token windows that ignore example boundaries, so only 26.8% of
> supervised tokens had their example's Words: field in context; 32.6% of
> training examples have no Words: field; and adaptation covered 0.13 epochs
> of the Instruct data. These results do not show that models of this size
> cannot follow such constraints; testing that would require training on
> windows that keep each example's instructions in context.

**Section VII (E4), replace the paragraph after Table III, and replace the width-1024 row with the batch-8 numbers:**
> LoRA (r=8) is slower per step than full fine-tuning at widths 64 and 128 in
> every repeat. At widths 256 and 512 step times switch between two regimes
> about 20% apart from one repeat to the next, larger than the difference
> between methods, so we cannot rank the methods there. At width 1024 with
> batch size 16, full fine-tuning needs 5.7 GiB, more than the GPU memory
> left free by the display, and slows down disproportionately (0.29 s per
> sample, against 0.095 s at batch size 8). Re-timed at batch size 8, where
> neither method is memory-bound, LoRA (r=8) is 1.11× faster per step (0.69
> vs. 0.76 s). Fewer trainable parameters therefore gave at most a modest
> speedup at the largest width we could measure cleanly, and we do not
> locate a crossover width.

**Section VIII (E5):**
* Replace "speedups between 0.81× and 1.33×" with:
  > speedups between 0.67× and 1.42× across both batch sizes and both caches
* Specify "median 1.20×" as:
  > preallocated cache; 1.04× for the concatenating cache
* Replace the memory sentence:
  > At batch size 16 and 490 new tokens, peak memory is 108 MB without a
  > cache, 159 MB with the concatenating cache and 93 MB with the
  > preallocated cache; the preallocated cache's 93 MB is fixed by its
  > 512-token allocation, so it exceeds the no-cache peak for shorter
  > generations (30 MB at 20 tokens).
* In the Discussion and Conclusion, remove "capped memory, which matters more
  than speed" and "where it also limited peak memory".

**Table IV / Sample section:**
* Caption:
  > Samples (temperature 0.8, top-k 50, up to 300 new tokens; the Words
  > metric uses 150). The LoRA r=64 output is the only one of the 20
  > fine-tuned samples (5 prompts × 4 methods) that contains all three words;
  > it is illustrative, not typical.
* Drop "Unfiltered" or "(trimmed)", whichever does not apply.
* Text: replace "every fine-tuned model opens with 'Once upon a time' and
  reuses names … which the base model never does" with:
  > 16 of the 20 fine-tuned samples open with "Once upon a time", and the
  > fine-tuned models reuse names and objects from the summary more
  > consistently than the base model
* Add, after "continue past their natural ending into a new Summary:
  header":
  > a format present in 30% of the training examples

**Limitations:**
* (1) Replace "22% to 40%" with:
  > 22% of the TinyStories test tokens in Experiments 1–2, and 11% of the
  > TinyStories and 40% of the Instruct test tokens in Experiment 3
* (2) End after "no early stopping". Delete "so the results stand".
* (3) Add the LR-grid boundary.
* (4) Replace with:
  > Experiments 4 and 5 use random weights; Experiment 4 varies only width at
  > a fixed depth of 8 layers, and Experiment 5 uses only the 1.78M-parameter
  > shape.
* (5) Replace with:
  > The Words rate uses 500 prompts with one sample each (1–15 successes per
  > run), so differences between methods are not resolved.
* (8) Replace with:
  > Training memory was measured only in the Experiment 4 benchmark: within
  > 0.07 GB across methods at width 128, but 5.7 GiB (full fine-tuning) vs.
  > 4.7–4.8 GiB (LoRA) at width 1024.
* (9, new):
  > TinyStories' official splits share verbatim stories (Section III); the
  > effect on our results is small but these splits are not fully held out.

---

## 5. Runtime checks

* **Tests:** `tests/test_model.py` and `tests/test_lora.py` pass (27/27).
* **Benchmark-scale equivalence** (8 layers × 128 dim, batch 16, 16-token
  prompt, 490 greedy tokens, random weights):
  * No-cache, concatenating and preallocated caches produce identical tokens.
  * Max logit difference is 8.3e-7.
* **E4 per-repeat medians:** in `out/runtime_checks.json`.

| Width | LoRA r=8 faster in (of 25 full-vs-LoRA repeat pairings) | Speedup range |
|---|---|---|
| 64 | 0% | 0.74–0.98 |
| 128 | 20% | 0.86–1.10 |
| 256 | 76% | 0.84–5.58, one anomalous 0.17 s repeat |
| 512 | 56% | 0.85–1.22 |
| 1024 | 100% | 1.15–2.25 |

r=64 at 512 is slower in every pairing.

**E4 peak allocated memory (MiB), full / r=8 / r=64:**

| Width | Full | r=8 | r=64 |
|---|---|---|---|
| 64 | 387 | 388 | 450 |
| 128 | 662 | 650 | 717 |
| 256 | 1239 | 1183 | 1260 |
| 512 | 2581 | 2325 | 2436 |
| 1024 | **5815** | 4795 | 4962 |

The card has 6,144 MiB.

**Width-1024 memory-pressure check** (`out/bench_mem_check.json`):
* Setup: `bench_train.run_one` unchanged; 15 warm-up and 40 timed steps;
  2 interleaved repeats per cell.
* The card had 5,153 of 6,144 MiB free at the start; the Windows display
  holds the rest.

| Batch | Full FT median s/step (peak MiB) | LoRA r=8 median s/step (peak MiB) | LoRA speedup |
|---|---|---|---|
| 16 (as in E4) | 4.68, 4.80 (5,815) | 1.34, 1.34 (4,795) | 3.55× |
| 8 | 0.76, 0.77 (3,701) | 0.68, 0.69 (2,613) | **1.11×** |

* **Full FT at batch 16:** per-sample time is 3.1× worse than at batch 8
  (0.29 vs. 0.095 s). This is the signature of exceeding the free GPU
  memory. On Windows, the driver then spills to shared system memory instead
  of failing.
* **LoRA:** per-sample time is the same at both batch sizes (0.084 vs.
  0.086 s).
* **Conclusion:** the 1.84× at width 1024 is mostly a memory-pressure
  artifact. Without it, LoRA r=8 is 1.11× faster at that width.
* **Run-to-run variation:** the batch-16 speedup itself varies with how much
  memory other programs hold (1.84× originally, 3.55× today), so that row of
  Table III should not be reported.
* **Optional clean replacement table (~1.5 h, benchmark only, no training):**

  ```
  python bench_train.py --batch-size 8 --out-dir camera_ready/out/e4_batch8
  ```

  The `--out-dir` keeps the original E4 JSON intact.

---

## 6. What still needs a person

* Apply the text above.
* Restore the two figures. `fig_e5_kvcache_speedup.png` is unaffected.
  `fig_e2_scaling_law.png` is unchanged unless you retrain.
* Add the INMIC first-page notices:
  * Top right: "2026 28th International Multitopic Conference (INMIC)".
  * Bottom left: "979-8-3195-6694-2/26/$31.00 ©2026 IEEE".
* Check the page limit and layout.
* Run PDF eXpress with conference ID 73848X before 10 Nov.
