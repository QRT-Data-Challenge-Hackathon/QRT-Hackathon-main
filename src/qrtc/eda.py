"""Fast, opinionated EDA for the first 15 minutes: what is the data, how were train/test split,
which features carry signal, and is there leakage or drift?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from qrtc.config import Spec
from qrtc.cv import detect_strategy


def overview(df: pd.DataFrame, max_cols: int = 200) -> pd.DataFrame:
    """One row per column: dtype, missing share, cardinality, basic stats, example value."""
    rows = []
    for col in df.columns[:max_cols]:
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        rows.append({
            "column": col, "dtype": str(s.dtype), "missing": round(s.isna().mean(), 4),
            "n_unique": s.nunique(dropna=True),
            "mean": s.mean() if numeric else np.nan, "std": s.std() if numeric else np.nan,
            "min": s.min() if numeric else np.nan, "max": s.max() if numeric else np.nan,
            "example": s.dropna().iloc[0] if s.notna().any() else None,
        })
    return pd.DataFrame(rows).set_index("column")


def split_structure(train: pd.DataFrame, test: pd.DataFrame, spec: Spec) -> dict:
    """How do train and test relate? Drives the choice of CV (see qrtc.cv)."""
    info: dict = {"n_train": len(train), "n_test": len(test),
                  "recommended_cv": detect_strategy(train, test, spec)}
    for key in filter(None, [spec.date_col, spec.group_col]):
        if key in train and key in test:
            tr, te = set(train[key].dropna()), set(test[key].dropna())
            info[f"{key}_n_train"], info[f"{key}_n_test"] = len(tr), len(te)
            info[f"{key}_test_share_seen_in_train"] = round(len(te & tr) / max(len(te), 1), 4)
            if pd.api.types.is_numeric_dtype(train[key]):
                info[f"{key}_train_range"] = (train[key].min(), train[key].max())
                info[f"{key}_test_range"] = (test[key].min(), test[key].max())
    if spec.date_col and spec.date_col in train:
        per_date = train.groupby(spec.date_col).size()
        info["rows_per_date"] = per_date.describe()[["min", "50%", "max"]].to_dict()
    return info


def target_summary(y: pd.DataFrame, task: str) -> pd.DataFrame:
    """Class balance for classification, distribution for regression / survival."""
    if task in ("binary", "multiclass"):
        if y.shape[1] > 1:
            return y.sum().to_frame("count").assign(share=lambda d: d["count"] / len(y))
        counts = y.iloc[:, 0].value_counts()
        return counts.to_frame("count").assign(share=lambda d: d["count"] / len(y))
    return y.describe().T


def signal_scan(X: pd.DataFrame, y: np.ndarray, dates: np.ndarray | None = None,
                top: int = 30) -> pd.DataFrame:
    """Per-feature Spearman correlation with the target.

    With ``dates``: mean per-date IC, its t-stat (mean/std*sqrt(n_dates)) and the share of dates
    with the same sign — far more reliable than a pooled correlation on panel data."""
    num = X.select_dtypes(include=np.number)
    target = np.asarray(y, dtype=float)
    if dates is not None and pd.Series(dates).value_counts().median() < 10:
        dates = None  # too few rows per date for a cross-sectional IC: use pooled Spearman
    if dates is None:
        corr = num.corrwith(pd.Series(target, index=num.index), method="spearman")
        return corr.to_frame("spearman").assign(abs=lambda d: d["spearman"].abs()).sort_values(
            "abs", ascending=False).head(top)
    d = pd.Series(np.asarray(dates), index=num.index)
    xr = num.groupby(d).rank(pct=True)
    yr = pd.Series(target, index=num.index).groupby(d).rank(pct=True)
    xc = xr - xr.groupby(d).transform("mean")
    yc = yr - yr.groupby(d).transform("mean")
    cov = xc.mul(yc, axis=0).groupby(d).sum()
    den = np.sqrt(xc.pow(2).groupby(d).sum().mul(yc.pow(2).groupby(d).sum(), axis=0))
    ic = (cov / den.replace(0, np.nan))  # per-date Spearman of every feature with the target
    out = pd.DataFrame({"ic_mean": ic.mean(), "ic_std": ic.std(),
                        "hit": (np.sign(ic) == np.sign(ic.mean())).mean()})
    out["t_stat"] = out["ic_mean"] / out["ic_std"] * np.sqrt(ic.notna().sum())
    return out.reindex(out["t_stat"].abs().sort_values(ascending=False).index).head(top)


def adversarial_validation(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                           seed: int = 0, n_splits: int = 3) -> tuple[float, pd.Series]:
    """AUC of a classifier separating train from test, and the features it uses most.

    AUC ~ 0.5: same distribution. AUC >> 0.5: drift (or a time/id feature) -> drop or
    normalise the top features, and prefer a CV that mimics the shift."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    data = pd.concat([train[features], test[features]], ignore_index=True)
    data = data.apply(lambda s: s.astype("category") if s.dtype == object else s)
    label = np.r_[np.zeros(len(train)), np.ones(len(test))]
    oof, imp = np.zeros(len(data)), pd.Series(0.0, index=features)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in cv.split(data, label):
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31,
                                 verbose=-1, random_state=seed)
        clf.fit(data.iloc[tr], label[tr])
        oof[va] = clf.predict_proba(data.iloc[va])[:, 1]
        imp += pd.Series(clf.feature_importances_, index=features)
    return float(roc_auc_score(label, oof)), imp.sort_values(ascending=False)


def leakage_checks(train: pd.DataFrame, y: np.ndarray, spec: Spec) -> list[str]:
    """Cheap red flags: id/row order predicting the target, constant or duplicate columns."""
    flags = []
    target = pd.Series(np.asarray(y, dtype=float))
    for name, values in (("row order", pd.Series(np.arange(len(train)))),
                         (f"id {spec.id_col}", pd.to_numeric(train[spec.id_col],
                                                             errors="coerce"))):
        corr = values.corr(target, method="spearman")
        if pd.notna(corr) and abs(corr) > 0.05:
            flags.append(f"{name} correlates with target (spearman {corr:.3f})")
    constant = [c for c in train.columns if train[c].nunique(dropna=False) <= 1]
    if constant:
        flags.append(f"constant columns: {constant}")
    hashes = {c: int(pd.util.hash_pandas_object(train[c], index=False).sum()) for c in train}
    seen: dict[int, str] = {}
    dupes = [f"{c}=={seen.setdefault(h, c)}" for c, h in hashes.items() if seen.get(h, c) != c]
    if dupes:
        flags.append(f"duplicate columns: {dupes}")
    if spec.date_col and spec.date_col in train and spec.id_col in train:
        ids = pd.to_numeric(train[spec.id_col], errors="coerce")
        corr = ids.corr(pd.to_numeric(train[spec.date_col], errors="coerce"), method="spearman")
        if pd.notna(corr) and abs(corr) > 0.3:
            flags.append(f"id order tracks {spec.date_col} (spearman {corr:.2f}): row/id order "
                         "may reveal time order -> check whether shuffled dates are recoverable")
    return flags


def lag_chain_links(df: pd.DataFrame, entity: str, cols: list[str], decimals: int = 8
                    ) -> pd.DataFrame:
    """Find (row, next_row) pairs of the same entity where next_row's lags are this row's
    lags shifted by one (RET_2..RET_k of next == RET_1..RET_{k-1} of this).

    Non-empty output means the shuffled date order can be rebuilt (and RET_1 of the next
    date is this row's target). ASK THE ORGANISERS before exploiting it."""
    vals = df[cols].round(decimals)
    head = vals[cols[:-1]].astype(str).agg("|".join, axis=1)
    tail = vals[cols[1:]].astype(str).agg("|".join, axis=1)
    left = pd.DataFrame({"entity": df[entity].to_numpy(), "key": head.to_numpy(),
                         "row": df.index.to_numpy()})
    right = pd.DataFrame({"entity": df[entity].to_numpy(), "key": tail.to_numpy(),
                          "next_row": df.index.to_numpy()})
    links = left.merge(right, on=["entity", "key"])
    return links[links["row"] != links["next_row"]][["entity", "row", "next_row"]]
