"""
Tests for analysis.py's statistics: the mean/CI helper against a
hand-computable case, aggregate's grouping, and fit_power_law against data
generated from a known power law (should recover the true parameters).
"""

import json
import math

import numpy as np

from analysis import aggregate, fit_power_law, load_runs, mean_ci95


def test_mean_ci95_single_value_has_zero_width():
    mean, ci95 = mean_ci95([5.0])
    assert mean == 5.0
    assert ci95 == 0.0


def test_mean_ci95_matches_hand_computed_t_interval():
    # Known case: 5 values, hand-computable via scipy's own t table.
    values = [10.0, 12.0, 11.0, 13.0, 9.0]
    mean, ci95 = mean_ci95(values)
    assert math.isclose(mean, 11.0, rel_tol=1e-9)

    from scipy import stats
    sem = np.std(values, ddof=1) / math.sqrt(5)
    expected_half_width = sem * stats.t.ppf(0.975, df=4)
    assert math.isclose(ci95, expected_half_width, rel_tol=1e-9)


def test_mean_ci95_wider_with_more_variance():
    tight, ci_tight = mean_ci95([10.0, 10.1, 9.9])
    wide, ci_wide = mean_ci95([5.0, 15.0, 10.0])
    assert ci_wide > ci_tight


def test_aggregate_groups_and_sorts_by_key():
    runs = [
        {"final_metrics": {"x": 1}, "config": {"size": 10}},
        {"final_metrics": {"x": 3}, "config": {"size": 10}},
        {"final_metrics": {"x": 2}, "config": {"size": 20}},
    ]
    rows = aggregate(runs, group_key=lambda r: r["config"]["size"], value_key=lambda r: r["final_metrics"]["x"])
    assert [r["group"] for r in rows] == [10, 20]
    assert rows[0]["n"] == 2
    assert math.isclose(rows[0]["mean"], 2.0)
    assert rows[1]["n"] == 1
    assert rows[1]["values"] == [2]


def test_load_runs_reads_all_json_files(tmp_path):
    for i in range(3):
        run = {"config": {"seed": i}, "final_metrics": {"val_loss": 1.0 + i}}
        (tmp_path / f"run{i}.json").write_text(json.dumps(run))
    runs = load_runs(str(tmp_path))
    assert len(runs) == 3
    assert all("_path" in r for r in runs)


def test_fit_power_law_recovers_known_parameters():
    """Generate exact (no-noise) samples from L(N) = E + A * N^-alpha and
    check the fit recovers those parameters closely."""
    true_e, true_a, true_alpha = 2.0, 50.0, 0.5
    sizes = [1e5, 3e5, 1e6, 3e6, 1e7]
    losses = [true_e + true_a * (n ** -true_alpha) for n in sizes]

    fit = fit_power_law(sizes, losses, num_bootstrap=20)
    assert fit is not None
    assert math.isclose(fit["E"], true_e, rel_tol=0.05)
    assert math.isclose(fit["A"], true_a, rel_tol=0.1)
    assert math.isclose(fit["alpha"], true_alpha, rel_tol=0.1)

    # fit_fn should reproduce the fitted curve at the input points
    predicted = fit["fit_fn"](sizes)
    assert np.allclose(predicted, losses, rtol=1e-2)


def test_fit_power_law_returns_none_with_too_few_points():
    assert fit_power_law([1, 2], [1.0, 0.9]) is None


def test_fit_power_law_ci_bounds_contain_point_estimate():
    true_e, true_a, true_alpha = 1.5, 30.0, 0.4
    sizes = [1e5, 3e5, 1e6, 3e6, 1e7, 3e7]
    rng = np.random.default_rng(0)
    losses = [true_e + true_a * (n ** -true_alpha) + rng.normal(0, 0.01) for n in sizes]

    fit = fit_power_law(sizes, losses, num_bootstrap=50)
    assert fit is not None
    for name in ["E", "A", "alpha"]:
        lo, hi = fit[f"{name}_ci95"]
        assert lo <= fit[name] <= hi
