r"""Covariance estimation and risk allocation for the portfolio engine.

* **Ledoit-Wolf (2004)** linear shrinkage toward a scaled-identity target.
  The sample covariance is badly conditioned when the number of assets
  approaches the number of observations; shrinkage trades a little bias for a
  large variance reduction, with an *analytically optimal* intensity (no
  cross-validation).
* **Hierarchical Risk Parity (López de Prado, 2016)** — allocate risk down a
  correlation dendrogram instead of inverting an ill-conditioned matrix; more
  robust out-of-sample than mean-variance weights.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def ledoit_wolf(returns: np.ndarray) -> dict:
    r"""Ledoit-Wolf shrinkage of the sample covariance toward ``mu * I``.

    Parameters
    ----------
    returns:
        ``(T, N)`` array of returns (rows = time).

    Returns
    -------
    dict with ``cov`` (the shrunk covariance), ``shrinkage`` (the intensity
    :math:`\delta \in [0, 1]`), and ``mu`` (the average-variance target scale).
    """
    X = np.asarray(returns, float)
    T, N = X.shape
    X = X - X.mean(axis=0, keepdims=True)
    S = (X.T @ X) / T

    mu = np.trace(S) / N
    target = mu * np.eye(N)

    d2 = np.sum((S - target) ** 2) / N          # ||S - mu I||_F^2 / N

    # b2: variance of the sample covariance entries (Frobenius), capped by d2.
    b2_sum = 0.0
    for t in range(T):
        xt = X[t][:, None]
        diff = xt @ xt.T - S
        b2_sum += np.sum(diff ** 2)
    b2 = min(b2_sum / (T ** 2) / N, d2)

    shrinkage = 0.0 if d2 == 0 else b2 / d2
    cov = shrinkage * target + (1.0 - shrinkage) * S
    return {"cov": cov, "shrinkage": float(shrinkage), "mu": float(mu)}


def hierarchical_risk_parity(cov: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity weights from a covariance matrix.

    Steps: correlation -> distance -> single-linkage tree -> quasi-diagonal
    ordering (seriation) -> recursive bisection inverse-variance allocation.
    Returns a length-``N`` weight vector summing to 1 (long-only risk budget).
    """
    cov = np.asarray(cov, float)
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)

    link = linkage(squareform(dist, checks=False), method="single")
    order = _quasi_diagonal(link, n)
    return _recursive_bisection(cov, order)


def _quasi_diagonal(link: np.ndarray, n: int) -> list[int]:
    """Return leaf order that places similar assets adjacent (seriation)."""
    link = link.astype(int)
    items = [2 * n - 2]  # root cluster id
    while True:
        expanded = []
        changed = False
        for it in items:
            if it < n:
                expanded.append(it)
            else:
                left, right = link[it - n, 0], link[it - n, 1]
                expanded.extend([left, right])
                changed = True
        items = expanded
        if not changed:
            break
    return items


def _recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Allocate risk by recursive inverse-variance bisection along ``order``."""
    n = cov.shape[0]
    w = np.ones(n)
    clusters = [order]
    while clusters:
        new_clusters = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= (1.0 - alpha)
            new_clusters.extend([left, right])
        clusters = new_clusters
    return w / w.sum()


def _cluster_variance(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)
