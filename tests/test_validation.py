"""Tests for the validation spine.

The bar is correctness on synthetic null distributions: random returns
should produce ~50% PBO, an iid Sharpe SE should match the closed-form,
purged-CV should not leak, and so on.
"""

from __future__ import annotations

import numpy as np
import pytest

from stat_arb.validation import (
    PurgedKFold,
    adf_test,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    kpss_test,
    probability_of_backtest_overfitting,
    sharpe_ci_lo,
    sharpe_ratio,
    sharpe_se_lo,
    walk_forward_splits,
)


# -------------------- stationarity --------------------
def test_adf_rejects_random_walk_only_rarely():
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.standard_normal(500))
    res = adf_test(rw)
    assert not res.stationary_at_5pct  # null = unit root holds


def test_adf_rejects_for_strong_mean_reversion():
    """A strongly reverting AR(1) (phi=0.5) is unambiguously stationary."""
    rng = np.random.default_rng(0)
    n, phi = 1000, 0.5
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    assert adf_test(x).stationary_at_5pct       # ADF rejects unit root
    assert kpss_test(x).stationary_at_5pct      # KPSS fails to reject stationarity


# -------------------- Sharpe inference --------------------
def test_sharpe_se_iid_matches_closed_form():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0008, 0.01, size=2000)
    sr = sharpe_ratio(r)
    se = sharpe_se_lo(r, q=1)  # tiny q so the autocorr factor is ≈ 1
    # Closed-form iid SE: sqrt((1 + 0.5 SR_year^2 / ppy) / T) * sqrt(ppy)
    ppy = 252
    sr_period = sr / np.sqrt(ppy)
    expected_se = np.sqrt((1 + 0.5 * sr_period ** 2) / r.size) * np.sqrt(ppy)
    # Lo's autocorr correction is ≥ 1; for q=1 iid noise it'll be close to 1
    # but not exactly. Allow 30% slack.
    assert abs(se - expected_se) / expected_se < 0.30


def test_sharpe_ci_contains_true_value_under_iid():
    """Coverage check: 95% CI on iid normal returns covers true SR ~95% of the time."""
    rng = np.random.default_rng(1)
    true_sr_annual = 1.0
    mu_period = true_sr_annual / np.sqrt(252) * 0.01
    hits = 0
    n_trials = 80
    for _ in range(n_trials):
        r = rng.normal(mu_period, 0.01, size=750)
        _, lo, hi = sharpe_ci_lo(r)
        if lo <= true_sr_annual <= hi:
            hits += 1
    assert hits / n_trials >= 0.80


# -------------------- Deflated Sharpe --------------------
def test_expected_max_sharpe_increases_with_trials():
    a = expected_max_sharpe(10, 1.0)
    b = expected_max_sharpe(1000, 1.0)
    assert a < b


def test_dsr_punishes_winning_in_a_large_search():
    """Same returns + N=1 vs N=10000: the deflated SR collapses for large N."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=750)
    small = deflated_sharpe_ratio(
        r, n_trials=1, sr_variance_across_trials=0.0
    )
    large = deflated_sharpe_ratio(
        r, n_trials=10_000, sr_variance_across_trials=0.1
    )
    assert large["sr0"] > small["sr0"]
    assert large["dsr"] < small["dsr"]


# -------------------- Purged CV --------------------
def test_purged_kfold_train_test_disjoint():
    cv = PurgedKFold(n_splits=5, embargo_frac=0.02, label_horizon=3)
    X = np.arange(1000)
    for train, test in cv.split(X):
        assert set(train).isdisjoint(set(test))
        # embargo / purge means no train index in [test_start - h, test_end + h + emb]


def test_purged_kfold_yields_n_splits():
    cv = PurgedKFold(n_splits=4, embargo_frac=0.01)
    X = np.arange(500)
    folds = list(cv.split(X))
    assert len(folds) == 4


# -------------------- Walk-forward --------------------
def test_walk_forward_anchored_train_grows():
    splits = list(walk_forward_splits(n=300, train_size=100, test_size=50, mode="anchored"))
    sizes = [len(tr) for tr, _ in splits]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_walk_forward_rolling_train_constant():
    splits = list(walk_forward_splits(n=300, train_size=100, test_size=50, mode="rolling"))
    sizes = [len(tr) for tr, _ in splits]
    assert all(s == 100 for s in sizes)


# -------------------- PBO --------------------
def test_pbo_random_strategies_around_one_half(random_returns_matrix):
    """For independent zero-mean noise, in-sample 'winners' have no edge OOS."""
    out = probability_of_backtest_overfitting(random_returns_matrix, n_partitions=10)
    # Generous tolerance: with N=200 noise strategies, PBO should be > 0.4
    assert 0.40 <= out["pbo"] <= 0.60, f"PBO={out['pbo']:.2f}"
    assert out["n_combinations"] == 252  # C(10, 5)
