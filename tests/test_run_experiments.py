"""
Tests for run_experiments.py's pure helper functions -- the actual
multi-hour experiment runs are exercised via --quick smoke tests, not unit
tests, but the small deterministic pieces (token-budget math, the E2 model
family's shape formula) are cheap to lock in directly.
"""

from run_experiments import e2_model_config, steps_for_token_budget


def test_steps_for_token_budget_basic():
    assert steps_for_token_budget(100_000, batch_size=32, block_size=512) == 100_000 // (32 * 512)


def test_steps_for_token_budget_at_least_one():
    assert steps_for_token_budget(1, batch_size=32, block_size=512) == 1


def test_e2_model_config_head_dim_fixed_at_16():
    for num_layers in [2, 4, 6, 8, 10, 12, 14]:
        cfg = e2_model_config(num_layers)
        assert cfg["embed_dim"] == 16 * num_layers
        assert cfg["num_heads"] == num_layers
        assert cfg["embed_dim"] % cfg["num_heads"] == 0
        assert cfg["embed_dim"] // cfg["num_heads"] == 16


def test_e2_model_config_param_count_grows_with_layers():
    from model import GPT

    prev_params = 0
    for num_layers in [2, 4, 6, 8]:
        cfg = e2_model_config(num_layers)
        model = GPT(vocab_size=1024, **cfg)
        params = sum(p.numel() for p in model.parameters())
        assert params > prev_params
        prev_params = params
