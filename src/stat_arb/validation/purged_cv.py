r"""Purged k-fold cross-validation with embargo (López de Prado).

Naive k-fold leaks information when observations are correlated in time
(rolling windows, overlapping labels). Two corrections, both due to
López de Prado, *Advances in Financial Machine Learning* (2018):

* **Purging.** Remove training observations whose feature/label window
  overlaps the test set.
* **Embargo.** Drop an additional band of training observations
  immediately *after* the test set, since their features may have been
  computed using data that touches the test fold.

The classes mimic the scikit-learn splitter API
(``split(X, y=None) → Iterator[(train, test)]``) for drop-in use.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class PurgedKFold:
    """K-fold splitter with contiguous test folds, purging, and embargo.

    Parameters
    ----------
    n_splits:
        Number of folds.
    embargo_frac:
        Fraction of the total sample to embargo *after* each test fold.
        López de Prado recommends 0.01-0.02 of T for daily data.
    label_horizon:
        Maximum number of observations a label may span (purge radius).
        For point-in-time spread features this is 0; for rolling-window
        signals set it to the window size.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_frac: float = 0.01,
        label_horizon: int = 0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be ≥ 2.")
        if not 0 <= embargo_frac < 0.5:
            raise ValueError("embargo_frac must be in [0, 0.5).")
        if label_horizon < 0:
            raise ValueError("label_horizon must be ≥ 0.")
        self.n_splits = n_splits
        self.embargo_frac = embargo_frac
        self.label_horizon = label_horizon

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ARG002
        return self.n_splits

    def split(
        self,
        X,
        y=None,
        groups=None,  # noqa: ARG002
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        if n < self.n_splits * 2:
            raise ValueError(
                f"n={n} too small for n_splits={self.n_splits}."
            )

        embargo = int(np.ceil(self.embargo_frac * n))
        fold_edges = np.linspace(0, n, self.n_splits + 1, dtype=int)

        for k in range(self.n_splits):
            test_start, test_end = fold_edges[k], fold_edges[k + 1]
            test_idx = np.arange(test_start, test_end)

            # Purge: remove training samples within ``label_horizon`` of the test fold.
            purge_lo = max(0, test_start - self.label_horizon)
            purge_hi = min(n, test_end + self.label_horizon)
            # Embargo: extend the upper purge band.
            embargo_hi = min(n, purge_hi + embargo)

            train_mask = np.ones(n, dtype=bool)
            train_mask[purge_lo:embargo_hi] = False
            train_idx = np.where(train_mask)[0]

            if train_idx.size == 0:
                continue
            yield train_idx, test_idx
