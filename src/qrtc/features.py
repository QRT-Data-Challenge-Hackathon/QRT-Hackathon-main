"""Feature engineering helpers for anonymised financial panels. All functions are pure:
they return a new DataFrame and never modify their input.

Typical QRT panel: one row per (date, asset) with blocks like RET_1..RET_20, VOLUME_1..VOLUME_20
and categorical SECTOR / INDUSTRY. The strongest generic transforms are cross-sectional
(per-date) ranks / z-scores, group de-meaning (date x sector) and row statistics over a block.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

from qrtc.cv import Split


def col_block(df: pd.DataFrame, prefix: str) -> list[str]:
    """Columns named ``<prefix>_<int>`` sorted by the integer (RET_1, RET_2, ..., RET_20)."""
    pattern = re.compile(rf"^{re.escape(prefix)}_?(\d+)$")
    found = [(int(m.group(1)), c) for c in df.columns if (m := pattern.match(c))]
    return [c for _, c in sorted(found)]


def _with(df: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, new.set_index(df.index)], axis=1)


def cs_rank(df: pd.DataFrame, cols: Sequence[str], by: str | list[str]) -> pd.DataFrame:
    """Percentile rank (0..1) of each column within each ``by`` group (usually the date)."""
    ranks = df.groupby(by)[list(cols)].rank(pct=True)
    return _with(df, ranks.add_suffix("_csrank"))


def cs_zscore(df: pd.DataFrame, cols: Sequence[str], by: str | list[str],
              clip: float = 5.0) -> pd.DataFrame:
    """Per-group z-score, clipped to +-``clip``."""
    grouped = df.groupby(by)[list(cols)]
    z = (df[list(cols)] - grouped.transform("mean")) / grouped.transform("std").replace(0, np.nan)
    return _with(df, z.clip(-clip, clip).add_suffix("_csz"))


def group_demean(df: pd.DataFrame, cols: Sequence[str], by: list[str],
                 suffix: str | None = None) -> pd.DataFrame:
    """Column minus its mean within ``by`` (e.g. [DATE, SECTOR] removes the sector move)."""
    tag = suffix or "_dm_" + "_".join(by)
    means = df.groupby(by)[list(cols)].transform("mean")
    return _with(df, (df[list(cols)] - means).add_suffix(tag))


def group_agg(df: pd.DataFrame, cols: Sequence[str], by: list[str],
              aggs: Sequence[str] = ("mean", "std")) -> pd.DataFrame:
    """Broadcast group statistics back to rows (e.g. sector mean return on that date)."""
    parts = []
    for agg in aggs:
        stat = df.groupby(by)[list(cols)].transform(agg)
        parts.append(stat.add_suffix(f"_{'_'.join(by)}_{agg}"))
    return _with(df, pd.concat(parts, axis=1))


def row_stats(df: pd.DataFrame, cols: Sequence[str], prefix: str) -> pd.DataFrame:
    """Row-wise mean/std/min/max/sum/n_missing over a block of columns."""
    block = df[list(cols)]
    stats = pd.DataFrame({
        f"{prefix}_mean": block.mean(axis=1),
        f"{prefix}_std": block.std(axis=1),
        f"{prefix}_min": block.min(axis=1),
        f"{prefix}_max": block.max(axis=1),
        f"{prefix}_sum": block.sum(axis=1),
        f"{prefix}_nnan": block.isna().sum(axis=1),
    })
    return _with(df, stats)


def window_sums(df: pd.DataFrame, cols: Sequence[str], windows: Sequence[int],
                prefix: str) -> pd.DataFrame:
    """Cumulative sums over the first k columns of an ordered block (RET_1..RET_k = k-day ret)."""
    out = {f"{prefix}_sum{k}": df[list(cols)[:k]].sum(axis=1, min_count=1) for k in windows}
    return _with(df, pd.DataFrame(out))


def entity_lags(df: pd.DataFrame, cols: Sequence[str], entity: str, order: str,
                lags: Sequence[int] = (1,)) -> pd.DataFrame:
    """Lagged values per entity in ``order`` (only valid when rows form a time series)."""
    ordered = df.sort_values([entity, order])
    parts = [ordered.groupby(entity)[list(cols)].shift(k).add_suffix(f"_lag{k}") for k in lags]
    return pd.concat([df, pd.concat(parts, axis=1).reindex(df.index)], axis=1)


def missing_flags(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Indicator columns for missingness (missingness itself is often informative)."""
    return _with(df, df[list(cols)].isna().astype("int8").add_suffix("_isna"))


def oof_target_encode(train: pd.DataFrame, test: pd.DataFrame, col: str, y: np.ndarray,
                      splits: list[Split], smoothing: float = 20.0
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold mean-target encoding of ``col`` (no target leakage into train rows)."""
    target = np.asarray(y, dtype=float)
    prior = float(np.nanmean(target))
    enc_name = f"{col}_te"

    def _encode(keys: pd.Series, fit_keys: pd.Series, fit_y: np.ndarray) -> np.ndarray:
        stats = pd.DataFrame({"k": fit_keys.to_numpy(), "y": fit_y}).groupby("k")["y"].agg(
            ["sum", "count"])
        smooth = (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)
        return keys.map(smooth).fillna(prior).to_numpy(dtype=float)

    oof = np.full(len(train), prior)
    for tr_idx, va_idx in splits:
        oof[va_idx] = _encode(train[col].iloc[va_idx], train[col].iloc[tr_idx], target[tr_idx])
    test_enc = _encode(test[col], train[col], target)
    return (train.assign(**{enc_name: oof}), test.assign(**{enc_name: test_enc}))


def to_category(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cast columns to a shared pandas Categorical so train/test codes agree."""
    new_train, new_test = train.copy(), test.copy()
    for col in cols:
        cats = pd.Index(pd.concat([train[col], test[col]]).dropna().unique())
        new_train[col] = pd.Categorical(train[col], categories=cats)
        new_test[col] = pd.Categorical(test[col], categories=cats)
    return new_train, new_test


def auto_categoricals(df: pd.DataFrame, max_unique: int = 50) -> list[str]:
    """Object/category/bool columns plus low-cardinality integer columns named like ids/codes."""
    cats = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "category"]
    for col in df.columns:
        if col in cats or not pd.api.types.is_integer_dtype(df[col]):
            continue
        if df[col].nunique() <= max_unique and re.search(r"SECTOR|INDUSTRY|COUNTRY|TYPE|CODE",
                                                         col, re.IGNORECASE):
            cats.append(col)
    return cats


def rank_target(y: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Gaussianised rank of the target (per group if given). Training on ranks helps rank
    metrics (Spearman/IC) and tames fat tails; score against the ORIGINAL target."""
    from scipy import stats

    frame = pd.DataFrame({"y": np.asarray(y, float),
                          "g": 0 if groups is None else np.asarray(groups)})
    pct = frame.groupby("g")["y"].rank(pct=True).to_numpy()
    counts = frame.groupby("g")["y"].transform("count").to_numpy()
    return stats.norm.ppf((pct * counts - 0.5) / counts)
