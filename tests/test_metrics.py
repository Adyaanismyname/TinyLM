"""
Tests for metrics.py's pure computation pieces: masked perplexity/BPC,
the Words-constraint rate matcher, and the percentile-summary helper. No
checkpoint files or network access needed.
"""

import math

import numpy as np
import torch

from metrics import (
    _word_present,
    bits_per_char,
    percentile_summary,
    perplexity,
    unigram_baseline_perplexity,
    words_constraint_rate,
)
from model import GPT


def test_perplexity_matches_unmasked_reference():
    torch.manual_seed(0)
    model = GPT(vocab_size=50, block_size=16, embed_dim=32, num_heads=4, num_layers=2, dropout=0.0)
    model.eval()
    data = torch.randint(0, 50, (500,))
    device = torch.device("cpu")

    loss, ppl = perplexity(model, data, block_size=16, batch_size=8, device=device)
    assert math.isclose(math.exp(loss), ppl, rel_tol=1e-6)
    assert ppl > 1.0


def test_perplexity_accepts_numpy_array_and_memmap_not_just_tensors():
    """Regression test: data.py's TokenStream.data is a numpy array/memmap
    (uint16), not a torch tensor like train.py's old flatten() -- perplexity
    (and topk_accuracy, which shares _windows) must accept both."""
    torch.manual_seed(0)
    model = GPT(vocab_size=50, block_size=16, embed_dim=32, num_heads=4, num_layers=2, dropout=0.0)
    model.eval()
    device = torch.device("cpu")

    data_tensor = torch.randint(0, 50, (300,))
    loss_tensor, _ = perplexity(model, data_tensor, block_size=16, batch_size=8, device=device)

    data_numpy_uint16 = data_tensor.numpy().astype(np.uint16)
    loss_numpy, _ = perplexity(model, data_numpy_uint16, block_size=16, batch_size=8, device=device)

    assert math.isclose(loss_tensor, loss_numpy, rel_tol=1e-6)


def test_masked_perplexity_excludes_masked_positions():
    """Masking out an entire batch's worth of positions with mask=0 should
    give the same loss as computing perplexity over just the unmasked
    remainder -- mirrors model.py's ignore_index test but at the metrics
    (whole-stream, exact-eval) level."""
    torch.manual_seed(0)
    model = GPT(vocab_size=50, block_size=16, embed_dim=32, num_heads=4, num_layers=2, dropout=0.0)
    model.eval()
    device = torch.device("cpu")

    data = torch.randint(0, 50, (200,))
    mask_all_ones = torch.ones(200, dtype=torch.uint8)
    loss_unmasked, _ = perplexity(model, data, block_size=16, batch_size=8, device=device, mask=mask_all_ones)
    loss_reference, _ = perplexity(model, data, block_size=16, batch_size=8, device=device)
    assert math.isclose(loss_unmasked, loss_reference, rel_tol=1e-6)

    # Mask out everything from index 97 onward -- chosen so the cut lands
    # exactly on a window boundary (window i's targets span
    # [i*16+1, i*16+17); 97 == 6*16+1) rather than mid-window, so no window
    # straddles masked and unmasked targets and the two computations should
    # match exactly.
    cutoff = 97
    mask_cut = mask_all_ones.clone()
    mask_cut[cutoff:] = 0
    loss_masked, _ = perplexity(model, data, block_size=16, batch_size=8, device=device, mask=mask_cut)
    loss_front_only, _ = perplexity(model, data[:cutoff], block_size=16, batch_size=8, device=device)
    assert math.isclose(loss_masked, loss_front_only, rel_tol=1e-4)


def test_bits_per_char_positive_and_scales_with_loss():
    token_ids = [[1, 2, 3], [4, 5]]
    texts = ["abc", "de"]
    low = bits_per_char(1.0, token_ids, texts)
    high = bits_per_char(2.0, token_ids, texts)
    assert 0 < low < high


def test_unigram_baseline_matches_hand_computed_uniform_case():
    # A perfectly uniform token distribution over `vocab_size` symbols
    # should give a unigram baseline perplexity close to vocab_size.
    vocab_size = 20
    token_ids = [[i for i in range(vocab_size)] * 1000]
    ppl = unigram_baseline_perplexity(token_ids, vocab_size)
    assert abs(ppl - vocab_size) < 1.0


def test_word_present_matches_base_form():
    assert _word_present("cat", "the cat sat down")
    assert not _word_present("cat", "the dog sat down")


def test_word_present_matches_common_inflections():
    assert _word_present("cat", "the cats played")
    assert _word_present("play", "she played outside")
    assert _word_present("play", "she is playing now")
    assert _word_present("jump", "he jumps high")


def test_word_present_respects_word_boundaries():
    # "cat" should not match inside "category" or "concatenate"
    assert not _word_present("cat", "a category of things")
    assert not _word_present("cat", "concatenate the strings")


def test_words_constraint_rate_zero_when_words_never_appear():
    """An untrained random model given nonsense words that can't appear in
    its (numeric-decode) output should score at or near 0 -- sanity check
    that the metric doesn't trivially always return 1.0."""
    torch.manual_seed(0)
    from BPE import BPETokenizer, count_words, train_bpe

    texts = ["the cat sat on the mat", "a dog ran in the park"]
    base_chars, merges = train_bpe(count_words(texts), vocab_size=60)
    tokenizer = BPETokenizer(base_chars, merges)

    model = GPT(vocab_size=len(tokenizer.vocab), block_size=32, embed_dim=32, num_heads=4, num_layers=2)
    model.eval()
    device = torch.device("cpu")

    prompts = [("Words: xyzzyzzy, qwrfgh. Story:", ["xyzzyzzy", "qwrfgh"])]
    rate = words_constraint_rate(model, tokenizer, prompts, device, max_new_tokens=20)
    assert rate == 0.0


class _StubTokenizer:
    """Encodes/decodes via raw character codes -- just enough to drive
    words_constraint_rate's prompt-length bookkeeping and text matching
    without a real BPE vocabulary."""
    eos_id = None

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_words_constraint_rate_ignores_words_that_only_appear_in_the_prompt():
    """Regression test for a real bug: the header prompt states the target
    words verbatim ("Words: cat, ... Story:"), and generate() returns
    prompt+continuation concatenated -- so counting the whole sequence made
    the metric trivially 1.0 no matter what the model actually produced.
    Only the newly generated continuation should be checked."""
    class EchoModel:
        def eval(self):
            pass

        def generate(self, prompt_ids, max_new_tokens, temperature, top_k, eos_id):
            return prompt_ids  # no new tokens appended -- empty continuation

    prompts = [("the cat sat. Story:", ["cat"])]  # "cat" appears only in the prompt
    rate = words_constraint_rate(EchoModel(), _StubTokenizer(), prompts, torch.device("cpu"))
    assert rate == 0.0


def test_words_constraint_rate_counts_words_in_the_actual_continuation():
    class ContinuationModel:
        def eval(self):
            pass

        def generate(self, prompt_ids, max_new_tokens, temperature, top_k, eos_id):
            extra = torch.tensor([[ord(c) for c in " a cat appeared"]])
            return torch.cat([prompt_ids, extra], dim=1)

    prompts = [
        ("Story:", ["cat"]),       # continuation contains "cat" -> satisfied
        ("Story:", ["elephant"]),  # continuation doesn't contain "elephant" -> unsatisfied
    ]
    rate = words_constraint_rate(ContinuationModel(), _StubTokenizer(), prompts, torch.device("cpu"))
    assert rate == 0.5


def test_percentile_summary_median_and_iqr():
    samples = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    summary = percentile_summary(samples)
    assert summary["p50"] == 5.0
    assert summary["p25"] < summary["p50"] < summary["p75"]
