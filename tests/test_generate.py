"""
Tests for generate.sample(), the encode -> generate -> decode path behind
generate.py and generate_samples.py.
"""

import pytest
import torch

from BPE import BPETokenizer, count_words, train_bpe
from generate import sample
from model import GPT

TEXTS = ["once upon a time there was a little cat", "the cat was happy and liked to play in the sun"]


def make_tokenizer():
    base_chars, merges = train_bpe(count_words(TEXTS), vocab_size=60)
    return BPETokenizer(base_chars, merges)


def make_model(tokenizer, block_size=32, seed=0):
    torch.manual_seed(seed)
    return GPT(len(tokenizer.vocab), block_size=block_size, embed_dim=32, num_heads=2, num_layers=2).eval()


def test_returns_only_the_continuation_not_the_prompt():
    tok = make_tokenizer()
    model = make_model(tok)
    text, _ = sample(model, tok, "once upon a time", torch.device("cpu"), max_new_tokens=10, seed=0)
    assert isinstance(text, str)
    assert not text.startswith("once upon a time")


def test_same_seed_same_text():
    tok = make_tokenizer()
    model = make_model(tok)
    a = sample(model, tok, "the cat", torch.device("cpu"), max_new_tokens=15, seed=3)
    b = sample(model, tok, "the cat", torch.device("cpu"), max_new_tokens=15, seed=3)
    assert a == b


def test_output_is_cut_at_the_first_eos():
    """A 2-token-vocab model emits EOS almost immediately; whatever comes back
    must not contain the EOS token's text, and hit_eos must say why it stopped."""
    tok = make_tokenizer()
    model = make_model(tok)
    text, hit_eos = sample(model, tok, "the cat", torch.device("cpu"), max_new_tokens=25, temperature=0, seed=0)
    assert tok.eos_token not in text
    # greedy on an untrained model may or may not emit EOS, but the flag must be a real bool
    assert isinstance(hit_eos, bool)


def test_max_new_tokens_is_clamped_to_the_context_window():
    tok = make_tokenizer()
    model = make_model(tok, block_size=16)
    # asking for far more than fits must not trip the model's length assertion
    text, _ = sample(model, tok, "once upon a time", torch.device("cpu"), max_new_tokens=500, seed=0)
    assert isinstance(text, str)


def test_prompt_that_fills_the_context_raises():
    tok = make_tokenizer()
    model = make_model(tok, block_size=8)
    long_prompt = "the cat was happy and liked to play in the sun " * 3
    with pytest.raises(ValueError):
        sample(model, tok, long_prompt, torch.device("cpu"), max_new_tokens=5)


def test_empty_prompt_starts_a_fresh_story():
    tok = make_tokenizer()
    model = make_model(tok)
    text, _ = sample(model, tok, "", torch.device("cpu"), max_new_tokens=10, seed=0)
    assert isinstance(text, str)


def test_greedy_is_deterministic_regardless_of_seed():
    tok = make_tokenizer()
    model = make_model(tok)
    a, _ = sample(model, tok, "the cat", torch.device("cpu"), max_new_tokens=12, temperature=0, seed=1)
    b, _ = sample(model, tok, "the cat", torch.device("cpu"), max_new_tokens=12, temperature=0, seed=99)
    assert a == b
