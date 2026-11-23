"""Blend OOF predictions of several runs; weights are chosen on OOF, never on the public LB.

Rank averaging is the safe default for correlation metrics (Spearman/IC) and AUC; plain
averaging for probabilities (accuracy, logloss); hill climbing (Caruana et al. 2004) when one
model clearly dominates. Always compare the blend's OOF score with the best single run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

Scorer = Callable[[np.ndarray], float]


def _common_mask(preds: Sequence[np.ndarray]) -> np.ndarray:
    masks = [~np.isnan(p if p.ndim == 1 else p[:, 0]) for p in map(np.asarray, preds)]
    return np.logical_and.reduce(masks)


def rank_normalise(pred: np.ndarray) -> np.ndarray:
    """Map 1-D predictions to (0, 1] ranks, preserving NaN."""
    arr = np.asarray(pred, dtype=float)
    out = np.full_like(arr, np.nan)
    ok = ~np.isnan(arr)
    out[ok] = stats.rankdata(arr[ok]) / ok.sum()
    return out


def weighted_blend(preds: Sequence[np.ndarray], weights: Sequence[float],
                   rank: bool = False) -> np.ndarray:
    """Weighted mean of predictions (optionally of their ranks). Weights are normalised."""
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        raise ValueError("weights must sum to a positive number")
    arrays = [rank_normalise(p) if rank else np.asarray(p, float) for p in preds]
    return sum(wi * a for wi, a in zip(w / w.sum(), arrays, strict=True))


def hill_climb(preds: Sequence[np.ndarray], scorer: Scorer, greater_is_better: bool = True,
               n_iter: int = 50, rank: bool = False) -> np.ndarray:
    """Greedy forward selection with replacement; returns integer counts as weights."""
    counts = np.zeros(len(preds))
    best_score = -np.inf if greater_is_better else np.inf

    def better(a: float, b: float) -> bool:
        return a > b if greater_is_better else a < b

    for _ in range(n_iter):
        trial = []
        for i in range(len(preds)):
            cand = counts.copy()
            cand[i] += 1
            trial.append(scorer(weighted_blend(preds, cand, rank)))
        pick = int(np.argmax(trial) if greater_is_better else np.argmin(trial))
        if not better(trial[pick], best_score):
            break
        counts[pick] += 1
        best_score = trial[pick]
    return counts


def correlation_table(preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Spearman correlation between runs' OOF predictions on rows all of them cover."""
    mask = _common_mask(list(preds.values()))
    frame = pd.DataFrame({k: np.asarray(v)[mask] if np.asarray(v).ndim == 1
                          else np.asarray(v)[mask].argmax(1) for k, v in preds.items()})
    return frame.corr(method="spearman")


def common_rows(preds: Sequence[np.ndarray]) -> np.ndarray:
    """Indices of rows where every prediction is defined (time-split OOFs have gaps)."""
    return np.flatnonzero(_common_mask(preds))
