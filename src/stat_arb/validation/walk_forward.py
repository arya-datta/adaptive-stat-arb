"""Walk-forward analysis splits.

Two flavours, both common in stat-arb research:

* **Anchored.** Training window grows; test window slides forward.
* **Rolling.** Both windows slide with a fixed training size.

Anchored is more honest when more recent data is more representative
(structural drift); rolling is more honest when distant history is
clearly stale.
"""

from __future__ import annotations

from collections.abc import Iterator


def walk_forward_splits(
    n: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
    mode: str = "anchored",
) -> Iterator[tuple[range, range]]:
    """Yield ``(train_idx, test_idx)`` as ``range`` objects.

    Parameters
    ----------
    n:
        Total sample length.
    train_size:
        For ``mode='anchored'``, the *initial* training length; for
        ``mode='rolling'``, the *constant* training length.
    test_size:
        Test window length.
    step:
        How far to advance the test window each iteration. Defaults to
        ``test_size`` (i.e. non-overlapping test folds).
    mode:
        ``"anchored"`` or ``"rolling"``.
    """
    if mode not in ("anchored", "rolling"):
        raise ValueError("mode must be 'anchored' or 'rolling'.")
    if train_size < 1 or test_size < 1 or n < train_size + test_size:
        raise ValueError("Bad sizes: need n ≥ train_size + test_size, both ≥ 1.")
    if step is None:
        step = test_size

    start = train_size
    while start + test_size <= n:
        if mode == "anchored":
            train = range(0, start)
        else:
            train = range(start - train_size, start)
        test = range(start, start + test_size)
        yield train, test
        start += step
