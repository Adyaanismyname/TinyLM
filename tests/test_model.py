"""
Tests for model.py -- particularly that the three generation paths (no
cache, growing concat-cache, preallocated write-in-place cache) and both
attention implementations (manual, sdpa) all compute the same thing, since
Experiment 5 (bench_kvcache.py) depends on comparing their *speed* meaning
nothing unless their *output* is already known to agree.
"""

import copy

import pytest
import torch

from model import GPT, IGNORE_INDEX, _apply_eos


def make_model(attn_impl="manual", vocab_size=200, block_size=32, embed_dim=64, num_heads=4, num_layers=2, seed=0):
    torch.manual_seed(seed)
    model = GPT(vocab_size, block_size=block_size, embed_dim=embed_dim, num_heads=num_heads,
                num_layers=num_layers, attn_impl=attn_impl)
    model.eval()
    return model


@pytest.mark.parametrize("attn_impl", ["manual", "sdpa"])
def test_forward_shapes_and_loss(attn_impl):
    model = make_model(attn_impl=attn_impl)
    x = torch.randint(0, 200, (3, 10))
    y = torch.randint(0, 200, (3, 10))
    logits, loss, cache = model(x, y)
    assert logits.shape == (3, 10, 200)
    assert loss.ndim == 0
    assert cache is None


@pytest.mark.parametrize("attn_impl", ["manual", "sdpa"])
def test_ignore_index_masking_matches_dropping_the_row(attn_impl):
    model = make_model(attn_impl=attn_impl)
    x = torch.randint(0, 200, (2, 10))
    y = torch.randint(0, 200, (2, 10))

    y_masked = y.clone()
    y_masked[1, :] = IGNORE_INDEX
    _, loss_masked, _ = model(x, y_masked)
    _, loss_row0, _ = model(x[:1], y[:1])
    assert torch.allclose(loss_masked, loss_row0, atol=1e-5)


@pytest.mark.parametrize("attn_impl", ["manual", "sdpa"])
def test_cache_matches_nocache_exactly(attn_impl):
    model = make_model(attn_impl=attn_impl)
    prompt = torch.randint(0, 200, (2, 3))

    torch.manual_seed(1)
    gen_nocache = model.generate(prompt, max_new_tokens=8, use_cache=False)
    torch.manual_seed(1)
    gen_cache = model.generate(prompt, max_new_tokens=8, use_cache=True)
    assert torch.equal(gen_nocache, gen_cache)


@pytest.mark.parametrize("attn_impl", ["manual", "sdpa"])
def test_prealloc_matches_nocache_exactly(attn_impl):
    model = make_model(attn_impl=attn_impl)
    prompt = torch.randint(0, 200, (2, 3))

    torch.manual_seed(1)
    gen_nocache = model.generate(prompt, max_new_tokens=8, use_cache=False)
    torch.manual_seed(1)
    gen_prealloc = model.generate_prealloc(prompt, max_new_tokens=8)
    assert torch.equal(gen_nocache, gen_prealloc)


@pytest.mark.parametrize("attn_impl", ["manual", "sdpa"])
def test_prealloc_prefill_logits_match_concat_cache(attn_impl):
    """Direct, single-step check (rather than a multi-step autoregressive
    chain, which can compound floating-point noise into a different
    discrete sample if it ever occurs): the two caches' prefill logits
    should be numerically identical."""
    model = make_model(attn_impl=attn_impl)
    prompt = torch.randint(0, 200, (2, 5))

    logits_concat, _, _ = model(prompt, use_cache=True)
    cache = model.init_prealloc_cache(2, prompt.device)
    logits_prealloc = model.forward_prealloc(prompt, cache, pos=0)
    assert torch.allclose(logits_concat, logits_prealloc, atol=1e-5)


def test_manual_and_sdpa_agree_on_same_weights():
    """manual and sdpa compute the same softmax(QK^T/sqrt(d))V formula --
    given identical weights and input, their logits should be close (not
    exactly bit-equal, since the fused SDPA kernel may sum in a different
    order)."""
    model_manual = make_model(attn_impl="manual", seed=5)
    model_sdpa = make_model(attn_impl="sdpa", seed=5)
    model_sdpa.load_state_dict(model_manual.state_dict())

    x = torch.randint(0, 200, (2, 10))
    logits_manual, _, _ = model_manual(x)
    logits_sdpa, _, _ = model_sdpa(x)
    assert torch.allclose(logits_manual, logits_sdpa, atol=1e-4)


def test_generate_prealloc_respects_block_size():
    model = make_model(block_size=16)
    prompt = torch.randint(0, 200, (1, 5))
    out = model.generate_prealloc(prompt, max_new_tokens=10)
    assert out.shape[1] <= 16


def test_apply_eos_clamps_finished_rows():
    finished = torch.tensor([False, True])
    next_id = torch.tensor([[7], [7]])
    clamped, new_finished = _apply_eos(next_id, finished, eos_id=99)
    assert clamped.tolist() == [[7], [99]]
    assert new_finished.tolist() == [False, True]


def test_apply_eos_marks_newly_finished():
    finished = torch.tensor([False, False])
    next_id = torch.tensor([[99], [3]])
    _, new_finished = _apply_eos(next_id, finished, eos_id=99)
    assert new_finished.tolist() == [True, False]


def test_eos_stops_generation_early_and_clamps():
    tiny = make_model(vocab_size=2, block_size=64, embed_dim=32, num_heads=2, num_layers=2)
    torch.manual_seed(0)
    start = torch.zeros(8, 1, dtype=torch.long)
    out = tiny.generate(start, max_new_tokens=50, use_cache=True, eos_id=1)

    assert out.shape[1] < 51  # stopped before exhausting the budget
    for row in out:
        eos_positions = (row == 1).nonzero()
        if len(eos_positions) > 0:
            first_eos = eos_positions[0].item()
            assert (row[first_eos:] == 1).all()


def test_lora_wrapped_model_still_matches_across_caches():
    """The cache-equivalence property should survive wrapping the model with
    LoRA too, since Experiment 3's fine-tuned models are generated from."""
    from lora import add_lora

    model = make_model()
    add_lora(model, r=4, alpha=8)
    model.eval()
    prompt = torch.randint(0, 200, (2, 3))

    torch.manual_seed(2)
    gen_nocache = model.generate(prompt, max_new_tokens=6, use_cache=False)
    torch.manual_seed(2)
    gen_cache = model.generate(prompt, max_new_tokens=6, use_cache=True)
    assert torch.equal(gen_nocache, gen_cache)
