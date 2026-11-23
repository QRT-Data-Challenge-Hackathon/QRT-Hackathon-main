"""Turn raw model outputs into what the metric rewards.

  * accuracy on a balanced-ish binary target -> pick the threshold on OOF (or predict the
    per-date median split: half up, half down, when the target is "beats the cross-section")
  * per-date correlation metrics -> only the within-date order matters; rank per date
  * multiclass accuracy -> argmax, optionally with class-prior adjustment
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qrtc.metrics import Metric


def best_threshold(y: np.ndarray, proba: np.ndarray, metric: Metric,
                   weight: np.ndarray | None = None,
                   grid: np.ndarray | None = None) -> tuple[float, float]:
    """Threshold maximising ``metric`` on OOF predictions; returns (threshold, score)."""
    p = np.asarray(proba, float)
    ok = ~np.isnan(p)
    candidates = grid if grid is not None else np.quantile(p[ok], np.linspace(0.05, 0.95, 91))
    w = None if weight is None else np.asarray(weight)[ok]
    scores = [metric(np.asarray(y)[ok], p[ok], weight=w, threshold=t) for t in candidates]
    best = int(np.argmax(scores) if metric.greater_is_better else np.argmin(scores))
    return float(candidates[best]), float(scores[best])


def rank_within(pred: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Percentile rank of predictions inside each group (e.g. date)."""
    frame = pd.DataFrame({"p": np.asarray(pred, float), "g": np.asarray(groups)})
    return frame.groupby("g")["p"].rank(pct=True).to_numpy()


def median_split_within(pred: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """1 for predictions above their group's median, else 0 (balanced labels per date)."""
    return (rank_within(pred, groups) > 0.5).astype(int)


def prior_adjust(proba: np.ndarray, train_prior: np.ndarray,
                 target_prior: np.ndarray) -> np.ndarray:
    """Re-weight multiclass probabilities to a different class prior, then renormalise."""
    adj = np.asarray(proba) * (np.asarray(target_prior) / np.asarray(train_prior))
    return adj / adj.sum(axis=1, keepdims=True)
