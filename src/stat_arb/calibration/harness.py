"""Parallel parameter-sweep over a callable.

The harness is intentionally tiny — heavy lifting (cross-validation,
walk-forward) belongs in :mod:`stat_arb.validation`. What this module
gives you is the *thousands of trials* fan-out: a thin
``ProcessPoolExecutor`` wrapper that collects ``TrialResult`` records and
guarantees deterministic ordering of outputs.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class TrialResult:
    """One ``(params, metrics)`` row from a sweep."""

    params: Mapping[str, Any]
    metrics: Mapping[str, float]
    trial_id: int = 0
    error: str | None = None


class ParameterGrid:
    """Cartesian product over a ``dict[name → iterable]``.

    Mirrors scikit-learn's ``ParameterGrid`` so callers can swap in their
    own search machinery without changing the harness.
    """

    def __init__(self, grid: Mapping[str, Iterable[Any]]) -> None:
        self.grid = {k: list(v) for k, v in grid.items()}

    def __iter__(self):
        keys = list(self.grid.keys())
        for combo in product(*(self.grid[k] for k in keys)):
            yield dict(zip(keys, combo))

    def __len__(self) -> int:
        n = 1
        for v in self.grid.values():
            n *= len(v)
        return n


def grid_search(
    objective: Callable[..., Mapping[str, float]],
    grid: ParameterGrid | Mapping[str, Iterable[Any]],
    *,
    max_workers: int | None = None,
    raise_on_error: bool = False,
) -> list[TrialResult]:
    """Evaluate ``objective`` at each combination of ``grid``.

    Parameters
    ----------
    objective:
        ``objective(**params) → dict[str, float]``. Must be picklable
        (top-level function) for ``ProcessPoolExecutor`` to spawn it.
    grid:
        :class:`ParameterGrid` or raw dict.
    max_workers:
        Defaults to ``os.cpu_count()``. Pass ``1`` to force serial
        execution — useful for debugging.
    raise_on_error:
        If False, individual failures become ``TrialResult`` rows with
        ``error`` set (so partial sweeps still return data).

    Returns
    -------
    list[TrialResult] ordered by ``trial_id``.
    """
    grid = ParameterGrid(grid) if not isinstance(grid, ParameterGrid) else grid
    combos = list(enumerate(grid))

    if max_workers == 1:
        results = []
        for trial_id, params in combos:
            try:
                metrics = objective(**params)
                results.append(TrialResult(params=params, metrics=metrics, trial_id=trial_id))
            except Exception as exc:  # noqa: BLE001
                if raise_on_error:
                    raise
                results.append(TrialResult(
                    params=params, metrics={}, trial_id=trial_id, error=repr(exc),
                ))
        return results

    results: list[TrialResult] = [None] * len(combos)  # type: ignore[list-item]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(objective, **params): (tid, params)
                   for tid, params in combos}
        for fut in as_completed(futures):
            tid, params = futures[fut]
            try:
                metrics = fut.result()
                results[tid] = TrialResult(params=params, metrics=metrics, trial_id=tid)
            except Exception as exc:  # noqa: BLE001
                if raise_on_error:
                    raise
                results[tid] = TrialResult(
                    params=params, metrics={}, trial_id=tid, error=repr(exc),
                )
    return results
