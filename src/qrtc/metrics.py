"""Metric registry: every metric QRT-style challenges have used, behind one signature.

Prediction conventions (shared with ``qrtc.models``):
  * binary      -> 1-D probability of the positive class
  * multiclass  -> 2-D (n, k) class probabilities, classes encoded 0..k-1
  * regression  -> 1-D values
  * survival    -> 1-D risk score (higher = earlier event); y_true is (n, 2) [time, event]

Signature: ``fn(y_true, y_pred, weight=None, groups=None, **params) -> float``.
``groups`` (e.g. the date column) is used by per-date metrics such as ``mean_spearman``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics as skm

MetricFn = Callable[..., float]


@dataclass(frozen=True)
class Metric:
    name: str
    fn: MetricFn
    greater_is_better: bool
    description: str

    def __call__(self, y_true, y_pred, weight=None, groups=None, **params) -> float:
        return float(self.fn(y_true, y_pred, weight=weight, groups=groups, **params))

    def better(self, a: float, b: float) -> bool:
        """True when score ``a`` beats score ``b``."""
        return a > b if self.greater_is_better else a < b


def _labels(y_pred, threshold: float = 0.5) -> np.ndarray:
    arr = np.asarray(y_pred)
    if arr.ndim == 2:
        return arr.argmax(axis=1)
    if arr.dtype == bool:
        return arr.astype(int)
    return (arr > threshold).astype(int)


def _accuracy(y_true, y_pred, weight=None, groups=None, threshold=0.5):
    return skm.accuracy_score(np.asarray(y_true).astype(int), _labels(y_pred, threshold),
                              sample_weight=weight)


def _weighted_accuracy(y_true, y_pred, weight=None, groups=None, threshold=0.5):
    """Accuracy weighted by |weight| (QRT stock challenge: weight = |realised return|)."""
    if weight is None:
        raise ValueError("weighted_accuracy needs weight (set competition.weight_col)")
    w = np.abs(np.asarray(weight, dtype=float))
    hit = np.asarray(y_true).astype(int) == _labels(y_pred, threshold)
    return float((w * hit).sum() / w.sum())


def _balanced_accuracy(y_true, y_pred, weight=None, groups=None, threshold=0.5):
    return skm.balanced_accuracy_score(np.asarray(y_true).astype(int), _labels(y_pred, threshold),
                                       sample_weight=weight)


def _f1(y_true, y_pred, weight=None, groups=None, threshold=0.5, average="binary"):
    avg = "macro" if np.asarray(y_pred).ndim == 2 else average
    return skm.f1_score(np.asarray(y_true).astype(int), _labels(y_pred, threshold),
                        average=avg, sample_weight=weight)


def _auc(y_true, y_pred, weight=None, groups=None):
    arr = np.asarray(y_pred)
    multi = "ovr" if arr.ndim == 2 else "raise"
    return skm.roc_auc_score(np.asarray(y_true), arr, sample_weight=weight, multi_class=multi)


def _logloss(y_true, y_pred, weight=None, groups=None):
    arr = np.clip(np.asarray(y_pred, dtype=float), 1e-15, 1 - 1e-15)
    labels = np.arange(arr.shape[1]) if arr.ndim == 2 else None
    return skm.log_loss(np.asarray(y_true), arr, sample_weight=weight, labels=labels)


def _rmse(y_true, y_pred, weight=None, groups=None):
    return np.sqrt(skm.mean_squared_error(y_true, y_pred, sample_weight=weight))


def _mae(y_true, y_pred, weight=None, groups=None):
    return skm.mean_absolute_error(y_true, y_pred, sample_weight=weight)


def _r2(y_true, y_pred, weight=None, groups=None):
    return skm.r2_score(y_true, y_pred, sample_weight=weight)


def _zero_mean_r2(y_true, y_pred, weight=None, groups=None):
    """1 - sum w (y - p)^2 / sum w y^2 (Jane Street style, no mean subtraction)."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    w = np.ones_like(y) if weight is None else np.asarray(weight, float)
    return 1.0 - (w * (y - p) ** 2).sum() / (w * y**2).sum()


def _spearman(y_true, y_pred, weight=None, groups=None):
    return stats.spearmanr(y_true, y_pred).statistic


def _pearson(y_true, y_pred, weight=None, groups=None):
    return stats.pearsonr(np.asarray(y_true, float), np.asarray(y_pred, float)).statistic


def _per_group(y_true, y_pred, groups, corr: str) -> float:
    if groups is None:
        raise ValueError("per-group metric needs groups (set competition.date_col)")
    frame = pd.DataFrame({"y": np.asarray(y_true, float), "p": np.asarray(y_pred, float),
                          "g": np.asarray(groups)})
    per = frame.groupby("g").apply(lambda d: d["y"].corr(d["p"], method=corr) if len(d) > 2
                                   else np.nan, include_groups=False)
    return float(per.mean())


def _mean_spearman(y_true, y_pred, weight=None, groups=None):
    """Mean over groups (dates) of the cross-sectional Spearman correlation (Numerai/QRT IC)."""
    return _per_group(y_true, y_pred, groups, "spearman")


def _mean_pearson(y_true, y_pred, weight=None, groups=None):
    return _per_group(y_true, y_pred, groups, "pearson")


def _surv_arrays(y_true) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(y_true)
    return arr[:, 0].astype(float), arr[:, 1].astype(bool)


def _cindex(y_true, y_pred, weight=None, groups=None):
    """Harrell's C-index; y_pred is a risk score (higher = earlier event)."""
    from lifelines.utils import concordance_index

    time, event = _surv_arrays(y_true)
    return concordance_index(time, -np.asarray(y_pred, float), event)


def _ipcw_cindex(y_true, y_pred, weight=None, groups=None, tau=None, y_train=None):
    """Uno's IPCW C-index truncated at ``tau`` (QRT x Gustave Roussy AML challenge: tau=7)."""
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv

    time, event = _surv_arrays(y_true)
    t_tr, e_tr = _surv_arrays(y_train) if y_train is not None else (time, event)
    s_test = Surv.from_arrays(event=event, time=time)
    s_train = Surv.from_arrays(event=e_tr, time=t_tr)
    return concordance_index_ipcw(s_train, s_test, np.asarray(y_pred, float), tau=tau)[0]


METRICS: dict[str, Metric] = {
    m.name: m
    for m in [
        Metric("accuracy", _accuracy, True, "share of correct labels (proba > 0.5 / argmax)"),
        Metric("weighted_accuracy", _weighted_accuracy, True, "accuracy weighted by |weight|"),
        Metric("balanced_accuracy", _balanced_accuracy, True, "mean per-class recall"),
        Metric("f1", _f1, True, "F1 (macro for multiclass)"),
        Metric("auc", _auc, True, "ROC AUC (ovr for multiclass)"),
        Metric("logloss", _logloss, False, "cross-entropy"),
        Metric("rmse", _rmse, False, "root mean squared error"),
        Metric("mae", _mae, False, "mean absolute error"),
        Metric("r2", _r2, True, "coefficient of determination"),
        Metric("zero_mean_r2", _zero_mean_r2, True, "weighted R2 without mean subtraction"),
        Metric("spearman", _spearman, True, "global Spearman rank correlation"),
        Metric("pearson", _pearson, True, "global Pearson correlation"),
        Metric("mean_spearman", _mean_spearman, True, "mean per-date Spearman (IC)"),
        Metric("mean_pearson", _mean_pearson, True, "mean per-date Pearson"),
        Metric("cindex", _cindex, True, "Harrell concordance index"),
        Metric("ipcw_cindex", _ipcw_cindex, True, "Uno IPCW concordance, truncated at tau"),
    ]
}


def get_metric(name: str) -> Metric:
    try:
        return METRICS[name]
    except KeyError as exc:
        raise KeyError(f"unknown metric {name!r}; known: {sorted(METRICS)}") from exc
