"""Cross-validation splits that mirror how the organisers split train and test.

The single most important modelling decision in a QRT-style challenge is a CV that tracks the
private leaderboard. Rules of thumb:
  * test dates disjoint from train dates, interleaved  -> ``date_blocks`` (contiguous date
    blocks, optional embargo of ``embargo`` dates on each side of the validation block)
  * test dates strictly after train dates             -> ``time`` (expanding window)
  * test entities disjoint (new patients / stocks)     -> ``group`` on ``group_col``
  * i.i.d. rows                                        -> ``stratified`` / ``kfold``
Never use plain KFold when a date column exists: rows of the same date share information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from qrtc.config import Spec

Split = tuple[np.ndarray, np.ndarray]


def detect_strategy(train: pd.DataFrame, test: pd.DataFrame, spec: Spec) -> str:
    """Pick a CV strategy from how train and test relate (used when spec.cv == 'auto')."""
    if spec.date_col and spec.date_col in train and spec.date_col in test:
        tr, te = train[spec.date_col], test[spec.date_col]
        ordered = pd.api.types.is_numeric_dtype(tr) or pd.api.types.is_datetime64_any_dtype(tr)
        if ordered and te.min() > tr.max():
            return "time"
        return "date_blocks"
    if spec.group_col and spec.group_col in train and spec.group_col in test:
        overlap = np.isin(test[spec.group_col].unique(), train[spec.group_col].unique()).mean()
        if overlap < 0.5:
            return "group"
    return "stratified" if spec.task in ("binary", "multiclass") else "kfold"


def date_block_splits(dates: np.ndarray, n_splits: int, embargo: int = 0) -> list[Split]:
    """K contiguous blocks of unique dates; drop ``embargo`` dates either side from training."""
    uniq = np.sort(pd.unique(dates))
    if len(uniq) < n_splits:
        raise ValueError(f"only {len(uniq)} unique dates for {n_splits} splits")
    pos = np.searchsorted(uniq, dates)
    splits: list[Split] = []
    for block in np.array_split(np.arange(len(uniq)), n_splits):
        lo, hi = block[0], block[-1]
        val = (pos >= lo) & (pos <= hi)
        train = (pos < lo - embargo) | (pos > hi + embargo)
        splits.append((np.flatnonzero(train), np.flatnonzero(val)))
    return splits


def time_splits(dates: np.ndarray, n_splits: int, embargo: int = 0) -> list[Split]:
    """Expanding window: train on dates before each validation block (minus embargo)."""
    uniq = np.sort(pd.unique(dates))
    if len(uniq) < n_splits + 1:
        raise ValueError(f"only {len(uniq)} unique dates for {n_splits} time splits")
    pos = np.searchsorted(uniq, dates)
    blocks = np.array_split(np.arange(len(uniq)), n_splits + 1)[1:]
    splits: list[Split] = []
    for block in blocks:
        lo, hi = block[0], block[-1]
        train = pos < lo - embargo
        val = (pos >= lo) & (pos <= hi)
        splits.append((np.flatnonzero(train), np.flatnonzero(val)))
    return splits


def make_splits(train: pd.DataFrame, y: np.ndarray, spec: Spec, strategy: str | None = None,
                test: pd.DataFrame | None = None) -> list[Split]:
    """Build CV splits; ``strategy`` overrides spec.cv; 'auto' needs ``test`` to decide."""
    strategy = strategy or spec.cv
    if strategy == "auto":
        strategy = detect_strategy(train, test if test is not None else train.iloc[:0], spec)
    n = len(train)
    if strategy == "date_blocks":
        return date_block_splits(train[spec.date_col].to_numpy(), spec.n_splits, spec.embargo)
    if strategy == "time":
        return time_splits(train[spec.date_col].to_numpy(), spec.n_splits, spec.embargo)
    if strategy == "group":
        groups = train[spec.group_col].to_numpy()
        return list(GroupKFold(n_splits=spec.n_splits).split(np.zeros(n), groups=groups))
    if strategy == "stratified":
        labels = np.asarray(y)
        labels = labels.argmax(axis=1) if labels.ndim == 2 else labels
        cv = StratifiedKFold(n_splits=spec.n_splits, shuffle=True, random_state=spec.seed)
        return list(cv.split(np.zeros(n), labels))
    if strategy == "kfold":
        cv = KFold(n_splits=spec.n_splits, shuffle=True, random_state=spec.seed)
        return list(cv.split(np.zeros(n)))
    raise ValueError(f"unknown CV strategy {strategy!r}")


def describe_splits(splits: list[Split]) -> pd.DataFrame:
    """Fold sizes, for a quick sanity check in notebooks."""
    return pd.DataFrame(
        [{"fold": i, "n_train": len(tr), "n_val": len(va)} for i, (tr, va) in enumerate(splits)]
    )
