"""
Tests for lora.py -- especially merge_lora, which Experiment 3 depends on to
make "LoRA vs. full fine-tuning" a same-architecture, same-parameter-count
comparison rather than comparing a LoRA-wrapped model (extra A/B parameters,
extra modules) against a plain one.
"""

import pytest
import torch

from lora import LoRALinear, add_lora, lora_state_dict, merge_lora, trainable_parameters
from model import GPT


def build_and_wrap(r=4, alpha=8, target_ff=True, seed=0):
    torch.manual_seed(seed)
    model = GPT(vocab_size=50, block_size=16, embed_dim=32, num_heads=4, num_layers=2)
    model.eval()
    add_lora(model, r=r, alpha=alpha, target_ff=target_ff)
    return model


def test_add_lora_zero_init_matches_base_output():
    torch.manual_seed(0)
    model = GPT(vocab_size=50, block_size=16, embed_dim=32, num_heads=4, num_layers=2)
    model.eval()
    x = torch.randint(0, 50, (2, 8))
    logits_before, _, _ = model(x)

    add_lora(model, r=4, alpha=8)
    logits_after, _, _ = model(x)
    assert torch.allclose(logits_before, logits_after)


def test_add_lora_freezes_base_leaves_adapters_trainable():
    model = build_and_wrap()
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert p.requires_grad
        else:
            assert not p.requires_grad


def _train_a_few_steps(model, steps=20, lr=1e-2, seed=1):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        x = torch.randint(0, 50, (4, 8))
        y = torch.randint(0, 50, (4, 8))
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


def test_merge_lora_matches_unmerged_after_training():
    model = build_and_wrap()
    model.train()
    _train_a_few_steps(model)  # moves lora_A/lora_B away from zero-init

    x = torch.randint(0, 50, (3, 8))
    logits_unmerged, _, _ = model(x)

    merged = merge_lora(model)
    logits_merged, _, _ = merged(x)

    assert torch.allclose(logits_unmerged, logits_merged, atol=1e-5)


def test_merge_lora_produces_plain_linear_layers():
    model = build_and_wrap()
    merged = merge_lora(model)
    for block in merged.blocks:
        assert not isinstance(block.attn.qkv_proj, LoRALinear)
        assert not isinstance(block.attn.out_proj, LoRALinear)
        assert not isinstance(block.ff.net[0], LoRALinear)
        assert not isinstance(block.ff.net[2], LoRALinear)


def test_merge_lora_param_count_matches_unwrapped_model_of_same_shape():
    """The whole point: after merging, the model's total parameter count
    should exactly equal a fresh, never-wrapped model built with the same
    architecture -- so 'LoRA at rank r' and 'full fine-tuning' really do
    end up comparing the same-sized model."""
    wrapped = build_and_wrap(r=4, alpha=8)
    merged = merge_lora(wrapped)
    _, merged_total = trainable_parameters(merged)

    plain = GPT(vocab_size=50, block_size=16, embed_dim=32, num_heads=4, num_layers=2)
    _, plain_total = trainable_parameters(plain)

    assert merged_total == plain_total


def test_merge_lora_all_params_trainable_afterward():
    model = build_and_wrap()
    merged = merge_lora(model)
    trainable, total = trainable_parameters(merged)
    assert trainable == total


def test_merge_lora_respects_target_ff_false():
    """merge_lora should be a no-op (return the plain nn.Linear unchanged)
    on layers add_lora never wrapped in the first place."""
    model = build_and_wrap(target_ff=False)
    for block in model.blocks:
        assert not isinstance(block.ff.net[0], LoRALinear)
    merged = merge_lora(model)
    for block in merged.blocks:
        assert not isinstance(block.ff.net[0], LoRALinear)
        assert not isinstance(block.attn.qkv_proj, LoRALinear)


def test_lora_params_match_base_device_and_dtype():
    """Regression test: lora_A/lora_B must be created on the same
    device/dtype as the wrapped base layer, not default CPU float32 --
    otherwise wrapping an already-moved-to-device model (the normal order
    of operations when starting from a loaded checkpoint) crashes the first
    forward pass with a device mismatch."""
    model = GPT(vocab_size=50, block_size=16, embed_dim=32, num_heads=4, num_layers=2)
    model = model.to(dtype=torch.float64)  # stand-in for "a non-default device/dtype"
    add_lora(model, r=4, alpha=8)

    for block in model.blocks:
        assert block.attn.qkv_proj.lora_A.dtype == torch.float64
        assert block.attn.qkv_proj.lora_B.dtype == torch.float64
        assert block.attn.qkv_proj.lora_A.device == block.attn.qkv_proj.base.weight.device

    x = torch.randint(0, 50, (2, 8))
    model.eval()
    logits, _, _ = model(x)  # would raise on a device/dtype mismatch
    assert logits.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_lora_params_match_base_cuda_device():
    model = GPT(vocab_size=50, block_size=16, embed_dim=32, num_heads=4, num_layers=2).to("cuda")
    add_lora(model, r=4, alpha=8)
    model.eval()
    x = torch.randint(0, 50, (2, 8), device="cuda")
    logits, _, _ = model(x)  # would raise "tensors on different devices" before the fix
    assert logits.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_merge_lora_keeps_model_on_cuda():
    """Regression test: merge_lora's replacement nn.Linear must be created
    on the base layer's device, not default CPU -- otherwise a model
    trained on CUDA ends up with CPU-resident layers after merging, and the
    next forward pass crashes with a device mismatch."""
    model = GPT(vocab_size=50, block_size=16, embed_dim=32, num_heads=4, num_layers=2).to("cuda")
    add_lora(model, r=4, alpha=8)
    merged = merge_lora(model)

    for block in merged.blocks:
        assert block.attn.qkv_proj.weight.device.type == "cuda"
        assert block.ff.net[0].weight.device.type == "cuda"

    merged.eval()
    x = torch.randint(0, 50, (2, 8), device="cuda")
    logits, _, _ = merged(x)  # would raise on a device mismatch before the fix
    assert logits.device.type == "cuda"


def test_lora_state_dict_only_adapters():
    model = build_and_wrap()
    adapter = lora_state_dict(model)
    assert len(adapter) > 0
    assert all(name.endswith("lora_A") or name.endswith("lora_B") for name in adapter)
