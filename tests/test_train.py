"""
Tests for train.py -- the LR schedule, param-group weight decay split,
seeding, and (most importantly) that two identically-seeded train_model
calls produce identical loss curves, which is the property every paired
comparison in the paper (LoRA rank vs. rank, model size vs. size) depends on.
"""

import math

import numpy as np
import torch

from data import stream_from_array
from model import GPT
from train import build_optimizer, get_device, lr_at_step, seed_everything, train_model


def make_tiny_model(seed=0):
    torch.manual_seed(seed)
    return GPT(vocab_size=64, block_size=16, embed_dim=32, num_heads=4, num_layers=2, dropout=0.0)


def make_data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 64, size=n).tolist()


def test_lr_warmup_then_decays_to_min():
    base_lr = 1e-3
    warmup, max_steps = 10, 100

    # warmup: linearly increasing, reaching base_lr exactly on the last
    # warmup step (step index warmup - 1)
    lrs = [lr_at_step(s, base_lr, warmup, max_steps) for s in range(warmup)]
    assert lrs == sorted(lrs)
    assert math.isclose(lrs[-1], base_lr, rel_tol=1e-9)
    assert lrs[0] < base_lr

    # right after warmup: at (or essentially at) base_lr
    assert math.isclose(lr_at_step(warmup, base_lr, warmup, max_steps), base_lr, rel_tol=1e-6)

    # at max_steps: decayed to min_lr_ratio * base_lr
    assert math.isclose(lr_at_step(max_steps, base_lr, warmup, max_steps), base_lr * 0.1, rel_tol=1e-6)

    # monotonically non-increasing across the decay phase
    decay_lrs = [lr_at_step(s, base_lr, warmup, max_steps) for s in range(warmup, max_steps + 1)]
    assert all(a >= b - 1e-9 for a, b in zip(decay_lrs, decay_lrs[1:]))


def test_lr_no_warmup_still_decays():
    lrs = [lr_at_step(s, 1e-3, 0, 100) for s in [0, 50, 100]]
    assert lrs[0] > lrs[1] > lrs[2]


def test_build_optimizer_splits_decay_by_ndim():
    model = make_tiny_model()
    opt = build_optimizer(model, learning_rate=1e-3, weight_decay=0.05)
    decay_group, no_decay_group = opt.param_groups
    assert decay_group["weight_decay"] == 0.05
    assert no_decay_group["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay_group["params"])
    assert all(p.dim() < 2 for p in no_decay_group["params"])


def test_seed_everything_reproduces_torch_randomness():
    seed_everything(123)
    a = torch.randn(10)
    seed_everything(123)
    b = torch.randn(10)
    assert torch.equal(a, b)


def test_train_model_same_seed_same_loss_curve():
    device = torch.device("cpu")
    data = make_data()

    seed_everything(0)
    model_a = make_tiny_model(seed=42)
    history_a = train_model(
        model_a, data, data, device,
        block_size=16, batch_size=4, learning_rate=1e-3, max_steps=20,
        eval_interval=5, eval_batches=3, seed=7,
    )

    seed_everything(0)
    model_b = make_tiny_model(seed=42)
    history_b = train_model(
        model_b, data, data, device,
        block_size=16, batch_size=4, learning_rate=1e-3, max_steps=20,
        eval_interval=5, eval_batches=3, seed=7,
    )

    for a, b in zip(history_a, history_b):
        assert math.isclose(a["train_loss"], b["train_loss"], rel_tol=1e-4)
        assert math.isclose(a["val_loss"], b["val_loss"], rel_tol=1e-4)


def test_train_model_different_seed_different_loss_curve():
    device = torch.device("cpu")
    data = make_data()

    model_a = make_tiny_model(seed=1)
    history_a = train_model(
        model_a, data, data, device,
        block_size=16, batch_size=4, learning_rate=1e-3, max_steps=10,
        eval_interval=5, eval_batches=3, seed=1,
    )
    model_b = make_tiny_model(seed=2)
    history_b = train_model(
        model_b, data, data, device,
        block_size=16, batch_size=4, learning_rate=1e-3, max_steps=10,
        eval_interval=5, eval_batches=3, seed=2,
    )
    assert history_a[-1]["train_loss"] != history_b[-1]["train_loss"]


def test_train_model_reduces_loss_on_a_repetitive_stream():
    """Sanity check: given a token stream trivial enough to learn (heavy
    repetition), training loss should meaningfully drop, not stay flat or
    diverge -- catches wiring bugs (wrong target shift, broken optimizer
    param groups, schedule collapsing lr to ~0) that a shape-only test
    wouldn't."""
    device = torch.device("cpu")
    rng = np.random.default_rng(0)
    pattern = [1, 2, 3, 4, 5, 6, 7, 8]
    data = (pattern * 2000)

    model = make_tiny_model(seed=0)
    history = train_model(
        model, data, data, device,
        block_size=16, batch_size=8, learning_rate=3e-3, warmup_steps=5,
        max_steps=150, eval_interval=25, eval_batches=5, seed=0,
    )
    assert history[-1]["train_loss"] < history[0]["train_loss"] * 0.5


def test_train_model_elapsed_excludes_eval():
    """Regression test for the bug that confounded the paper's original
    LoRA-vs-full-FT timing: elapsed should reflect only the timed
    forward+backward+step region, not the eval_batches evaluation passes
    (which run at every eval_interval and would otherwise inflate elapsed
    roughly in proportion to how many eval checkpoints a run happens to
    hit)."""
    device = torch.device("cpu")
    data = make_data()

    model_frequent_eval = make_tiny_model(seed=0)
    history_frequent = train_model(
        model_frequent_eval, data, data, device,
        block_size=16, batch_size=4, learning_rate=1e-3, max_steps=20,
        eval_interval=1, eval_batches=20, seed=0,  # eval every step, expensive if it leaked into elapsed
    )

    model_rare_eval = make_tiny_model(seed=0)
    history_rare = train_model(
        model_rare_eval, data, data, device,
        block_size=16, batch_size=4, learning_rate=1e-3, max_steps=20,
        eval_interval=20, eval_batches=20, seed=0,  # eval only once, at the end
    )

    # Both ran the same 20 training steps on the same data/model/seed, so
    # their final elapsed (training-only time) should be close regardless
    # of how many (expensive) eval passes were interleaved.
    final_frequent = history_frequent[-1]["elapsed"]
    final_rare = history_rare[-1]["elapsed"]
    assert final_frequent < final_rare * 3  # generous bound; would be blown by eval leaking in
