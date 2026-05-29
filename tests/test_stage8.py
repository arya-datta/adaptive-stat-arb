"""Stage 8 tests: the Neural SDE learns a mean-reverting drift, and the gate is
honest -- it does NOT beat the linear OU on genuine OU data, but DOES beat it
when the true drift is nonlinear (double-well)."""

from __future__ import annotations

import numpy as np
import pytest

from stat_arb.data import SyntheticOU
from stat_arb.stage8 import NeuralSDE, neural_vs_ou_gate


def _double_well(n=6000, theta=20.0, sigma=1.2, dt=1 / 252, seed=0):
    """Simulate dx = theta*(x - x^3) dt + sigma dW (stable wells at +/-1).

    Strong drift so the cubic nonlinearity dominates the noise within the
    explored well -- a regime where the linear OU is genuinely misspecified.
    """
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 0.5
    for t in range(1, n):
        drift = theta * (x[t - 1] - x[t - 1] ** 3)
        x[t] = x[t - 1] + drift * dt + sigma * np.sqrt(dt) * rng.standard_normal()
    return x


def test_neural_sde_learns_mean_reverting_drift():
    spread = SyntheticOU(kappa=3.0, mu=0.0, sigma=0.2).simulate(n=2500, seed=0)["spread"]
    nsde = NeuralSDE(hidden=16, n_epochs=400).fit(spread.to_numpy(), dt=1 / 252)
    # Drift should pull back toward the mean: positive below, negative above.
    assert nsde.drift(np.array([0.15]))[0] < nsde.drift(np.array([-0.15]))[0]
    assert nsde.sigma == pytest.approx(0.2, rel=0.4)


def test_gate_neural_does_not_beat_ou_on_ou_data():
    """Maturity check: on genuine OU data the linear drift is correct, so the
    neural SDE should not materially beat it out-of-sample."""
    full = SyntheticOU(kappa=3.0, mu=0.0, sigma=0.2).simulate(n=4000, seed=1)["spread"].to_numpy()
    train, test = full[:2000], full[2000:]
    res = neural_vs_ou_gate(train, test, dt=1 / 252, n_epochs=400)
    # Neural may tie but must not win by a meaningful margin (no free lunch on OU).
    assert res["delta"] < 0.02 * abs(res["ll_ou"])


def test_gate_neural_beats_ou_on_nonlinear_drift():
    """When the true drift is nonlinear (double-well), the neural SDE should
    beat the linear OU out-of-sample -- it earns its place."""
    x = _double_well(n=6000, seed=0)
    train, test = x[:3000], x[3000:]
    res = neural_vs_ou_gate(train, test, dt=1 / 252, hidden=24, n_epochs=800)
    assert res["neural_wins"]
    assert res["delta"] > 0
