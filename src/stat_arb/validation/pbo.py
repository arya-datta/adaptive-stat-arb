r"""Probability of Backtest Overfitting (Bailey, Borwein, López de Prado, Zhu 2017).

Combinatorially-symmetric cross-validation. Algorithm:

1. Partition the T returns into ``S`` equal non-overlapping submatrices.
2. For every combination of ``S/2`` of those submatrices, define
   :math:`J` as the in-sample (IS) block and :math:`J^c` as the OOS block.
3. Pick the strategy with the best Sharpe on ``J``.
4. Record its OOS Sharpe rank within ``J^c``'s strategy distribution as a
   percentile :math:`\omega \in (0, 1)`, and compute :math:`\lambda =
   \log(\omega / (1 - \omega))`.
5. PBO is the fraction of combinations where :math:`\lambda \leq 0`,
   i.e. the best IS strategy underperformed median OOS.

A high PBO (≥ 0.5) means picking the best in-sample strategy gives you
worse-than-median performance OOS — the classic overfitting signature.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_partitions: int = 16,
) -> dict:
    """CSCV PBO over a strategy-returns matrix.

    Parameters
    ----------
    returns_matrix:
        Shape ``(T, N)`` — ``T`` time observations of ``N`` candidate
        strategies' per-period returns.
    n_partitions:
        Must be even (so we can split into halves). ``16`` is the
        recommendation in the paper for daily data with several years.

    Returns
    -------
    dict with keys ``pbo``, ``n_combinations``, ``logits`` (the per-combo
    :math:`\\lambda` values).
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T, N).")
    T, N = R.shape
    S = n_partitions
    if S % 2 != 0:
        raise ValueError("n_partitions must be even.")
    if T < S * 2 or N < 2:
        raise ValueError(
            f"Need T ≥ 2*S ({2*S}) and N ≥ 2; got T={T}, N={N}."
        )

    # Equal-sized contiguous partitions (trim the tail if T isn't divisible).
    fold_size = T // S
    R = R[: fold_size * S]
    folds = R.reshape(S, fold_size, N)

    logits: list[float] = []
    for is_idx in combinations(range(S), S // 2):
        oos_idx = tuple(i for i in range(S) if i not in is_idx)
        in_sample = np.concatenate([folds[i] for i in is_idx], axis=0)
        out_sample = np.concatenate([folds[i] for i in oos_idx], axis=0)

        is_sr = _sharpe_columnwise(in_sample)
        oos_sr = _sharpe_columnwise(out_sample)

        best_is = int(np.argmax(is_sr))
        # Rank of the chosen strategy within OOS distribution.
        # ``omega`` is its empirical CDF position.
        omega = float((oos_sr <= oos_sr[best_is]).sum()) / N
        omega = min(max(omega, 1.0 / (N + 1)), 1.0 - 1.0 / (N + 1))  # avoid 0/∞
        logits.append(float(np.log(omega / (1.0 - omega))))

    logits_arr = np.asarray(logits)
    return {
        "pbo": float((logits_arr <= 0).mean()),
        "n_combinations": int(logits_arr.size),
        "logits": logits_arr,
    }


def _sharpe_columnwise(R: np.ndarray) -> np.ndarray:
    means = R.mean(axis=0)
    stds = R.std(axis=0, ddof=1)
    out = np.where(stds > 0, means / stds, 0.0)
    return out
