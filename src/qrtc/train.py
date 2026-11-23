"""The CV loop: out-of-fold predictions, fold-averaged test predictions, per-fold scores.

Every experiment should go through :func:`run_cv` so that scores are comparable across agents
(same splits, same metric) and so that OOF predictions can be blended later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from qrtc.config import Spec
from qrtc.cv import Split
from qrtc.metrics import get_metric
from qrtc.models import make_model


@dataclass(frozen=True)
class CVResult:
    """Immutable outcome of one CV experiment."""

    model: str
    params: dict[str, Any]
    features: tuple[str, ...]
    metric: str
    score: float
    score_std: float
    fold_scores: tuple[float, ...]
    oof: np.ndarray            # NaN rows were never in a validation fold (time splits)
    test_pred: np.ndarray
    elapsed_s: float
    importance: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def summary(self) -> str:
        folds = ", ".join(f"{s:.5f}" for s in self.fold_scores)
        return (f"{self.model}: {self.metric} = {self.score:.5f} +- {self.score_std:.5f} "
                f"[{folds}] ({len(self.features)} feats, {self.elapsed_s:.0f}s)")


def _empty_like(y: np.ndarray, task: str, n_classes: int, n: int) -> np.ndarray:
    if task == "multiclass":
        return np.full((n, n_classes), np.nan)
    return np.full(n, np.nan)


def _slice(arr: np.ndarray | None, idx: np.ndarray) -> np.ndarray | None:
    return None if arr is None else np.asarray(arr)[idx]


def score_predictions(spec: Spec, y: np.ndarray, pred: np.ndarray, idx: np.ndarray | None = None,
                      weight: np.ndarray | None = None, groups: np.ndarray | None = None,
                      y_train: np.ndarray | None = None) -> float:
    """Score ``pred`` against ``y`` on rows ``idx`` (default: rows with a prediction)."""
    pred = np.asarray(pred)
    if idx is None:
        mask = ~np.isnan(pred if pred.ndim == 1 else pred[:, 0])
        idx = np.flatnonzero(mask)
    params = dict(spec.metric_params)
    if spec.metric == "ipcw_cindex":
        params["y_train"] = y if y_train is None else y_train
    return get_metric(spec.metric)(np.asarray(y)[idx], pred[idx], weight=_slice(weight, idx),
                                   groups=_slice(groups, idx), **params)


def run_cv(spec: Spec, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame,
           splits: list[Split], model: str, params: dict[str, Any] | None = None,
           weight: np.ndarray | None = None, groups: np.ndarray | None = None,
           fit_weight: np.ndarray | None = None, verbose: bool = True) -> CVResult:
    """Train ``model`` on each fold; ``groups`` feeds per-date metrics, ``weight`` the metric,
    ``fit_weight`` the model's sample_weight."""
    start, y = time.time(), np.asarray(y)
    n_classes = int(y.max()) + 1 if spec.task == "multiclass" else 0
    oof = _empty_like(y, spec.task, n_classes, len(X))
    test_pred = _empty_like(y, spec.task, n_classes, len(X_test))
    test_pred[...] = 0.0
    scores, importances = [], []
    for fold, (tr, va) in enumerate(splits):
        mdl = make_model(model, spec.task, params, seed=spec.seed + fold)
        kwargs = {} if fit_weight is None else {"sample_weight": np.asarray(fit_weight)[tr]}
        mdl.fit(X.iloc[tr], y[tr], X.iloc[va], y[va], **kwargs)
        oof[va] = mdl.predict(X.iloc[va])
        test_pred += mdl.predict(X_test) / len(splits)
        scores.append(score_predictions(spec, y, oof, va, weight, groups, y_train=y[tr]))
        importances.append(mdl.feature_importance())
        if verbose:
            print(f"  fold {fold}: {spec.metric} = {scores[-1]:.5f}")
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    result = CVResult(model=model, params=dict(params or {}), features=tuple(X.columns),
                      metric=spec.metric, score=float(np.mean(scores)),
                      score_std=float(np.std(scores)), fold_scores=tuple(scores), oof=oof,
                      test_pred=test_pred, elapsed_s=time.time() - start, importance=imp)
    if verbose:
        print(result.summary())
    return result


def fit_full(spec: Spec, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, model: str,
             params: dict[str, Any] | None = None, n_seeds: int = 3) -> np.ndarray:
    """Refit on all training rows with ``n_seeds`` seeds and average test predictions.

    Use for the final submission once a configuration is chosen by CV (often slightly better
    than the fold average, because every row is used)."""
    preds = [make_model(model, spec.task, params, seed=spec.seed + 100 + s)
             .fit(X, np.asarray(y)).predict(X_test) for s in range(n_seeds)]
    return np.mean(preds, axis=0)
