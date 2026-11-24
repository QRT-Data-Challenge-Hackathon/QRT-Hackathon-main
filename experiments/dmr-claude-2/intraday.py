"""Raw 5-minute prices -> one model row per (day, stock), for the "beat your country over the
last 30 minutes" task.

Features use only the bars a forecaster has at prediction time: the day up to 30 minutes
before the close. Train and test features are built the same way, from the concatenated
days, so per-stock history carries over the train/test boundary. All functions are pure.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from qrtc import features as F

HIDDEN_BARS = 6  # last 30 minutes of 5-minute bars
LAGS = (1, 3, 6, 12, 24)
ROLL_DAYS = (5, 20)


def to_wide(long: pd.DataFrame, day: str, stock: str, time: str, price: str) -> pd.DataFrame:
    """Long (day, stock, time, price) -> rows (day, stock), one column per bar in time order."""
    wide = long.pivot_table(index=[day, stock], columns=time, values=price, aggfunc="last")
    return wide.sort_index(axis=1)


def log_prices(wide: pd.DataFrame) -> pd.DataFrame:
    """Log prices with gaps forward-filled along the day (a missing bar = no trade)."""
    return np.log(wide.astype(float).where(wide > 0)).ffill(axis=1)


def tail_return(wide: pd.DataFrame, hidden: int = HIDDEN_BARS) -> pd.Series:
    """Log return over the last ``hidden`` bars (needs full-day prices, i.e. train only)."""
    lp = log_prices(wide)
    return (lp.iloc[:, -1] - lp.iloc[:, -hidden - 1]).rename("tail_ret")


def visible_bars(wide: pd.DataFrame, hidden: int = HIDDEN_BARS) -> pd.DataFrame:
    """Drop the hidden tail from full-day prices so train looks like test."""
    return wide.iloc[:, :-hidden]


def bar_features(visible: pd.DataFrame) -> pd.DataFrame:
    """Per-row intraday features from the visible bars (index kept)."""
    lp = log_prices(visible)
    first, last = lp.bfill(axis=1).iloc[:, 0], lp.iloc[:, -1]
    n = lp.shape[1]
    out = {"r_day": last - first, "r_first6": lp.iloc[:, min(6, n - 1)] - first}
    for k in LAGS:
        if k < n:
            out[f"r_last{k}"] = last - lp.iloc[:, -1 - k]
    rets = lp.diff(axis=1)
    out["vol"] = rets.std(axis=1)
    out["vol_last12"] = rets.iloc[:, -12:].std(axis=1)
    hi, lo = lp.max(axis=1), lp.min(axis=1)
    out["range"] = hi - lo
    out["pos_in_range"] = (last - lo) / (hi - lo).replace(0, np.nan)
    out["n_bars"] = visible.notna().sum(axis=1)
    out["first_logp"], out["last_logp"] = first, last
    return pd.DataFrame(out, index=visible.index)


def return_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("r_")]


def add_history(df: pd.DataFrame, day: str, stock: str) -> pd.DataFrame:
    """Overnight gap and multi-day rolling features per stock (days must be in time order)."""
    out = df.sort_values([stock, day]).copy()
    g = out.groupby(stock)
    out["r_gap"] = out["first_logp"] - g["last_logp"].shift(1)
    for w in ROLL_DAYS:
        out[f"hist_rday_mean{w}"] = g["r_day"].transform(lambda s, w=w: s.rolling(w, 1).mean())
        out[f"hist_vol_mean{w}"] = g["vol"].transform(lambda s, w=w: s.rolling(w, 1).mean())
    return out.sort_index()


def add_relative(df: pd.DataFrame, day: str, country: str) -> pd.DataFrame:
    """Each return relative to its country that day, the country's own move, and ranks."""
    rets = return_cols(df)
    out = F.group_demean(df, rets, [day, country], suffix="_rel")
    out = F.group_agg(out, rets, [day, country], aggs=("mean",))
    rel = [f"{c}_rel" for c in rets]
    out = F.cs_rank(out, rel, [day, country])
    out = F.cs_zscore(out, ["vol", "vol_last12", "range"], day)
    return out


def build_panel(visible: pd.DataFrame, country_of: pd.Series, day: str, stock: str,
                country: str) -> pd.DataFrame:
    """Full feature table for all (day, stock) rows; ``country_of`` maps stock -> country."""
    df = bar_features(visible).reset_index()
    df[country] = df[stock].map(country_of)
    if df[country].isna().any():
        raise ValueError(f"{df[country].isna().sum()} rows have no country")
    df = add_history(df, day, stock)
    df = add_relative(df, day, country)
    return df.drop(columns=["first_logp", "last_logp"])


def detect_bar_cols(df: pd.DataFrame, exclude: Sequence[str]) -> list[str]:
    """Numeric price columns in file order (everything numeric that is not a key/target)."""
    skip = set(exclude)
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def features_from_rows(train: pd.DataFrame, test: pd.DataFrame, bar_cols: Sequence[str],
                       id_col: str, day: str, stock: str, country: str
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Organiser layout: one row per (stock, day), prices every 5 min up to the cutoff.

    Train and test are stacked so per-stock history (gap, rolling means) runs across the
    boundary; no target is used. Returns (train_features, test_features) with ``id_col``.
    """
    keys = [id_col, day, stock, country]
    both = pd.concat([train[keys + list(bar_cols)], test[keys + list(bar_cols)]],
                     ignore_index=True)
    if both.duplicated([day, stock]).any():
        raise ValueError("duplicate (day, stock) rows")
    visible = both.set_index([day, stock])[list(bar_cols)]
    country_of = both.drop_duplicates(stock).set_index(stock)[country]
    panel = build_panel(visible, country_of, day, stock, country)
    panel = panel.merge(both[[id_col, day, stock]], on=[day, stock], validate="one_to_one")
    is_train = panel[id_col].isin(set(train[id_col]))
    return panel[is_train].reset_index(drop=True), panel[~is_train].reset_index(drop=True)


def relative_target(tail: pd.Series, keys: pd.DataFrame, day: str, country: str,
                    weights: Sequence[float] | None = None) -> pd.Series:
    """Stock tail return minus its country's (equal- or ``weights``-weighted) tail return."""
    frame = keys[[day, country]].assign(r=tail.to_numpy())
    if weights is None:
        bench = frame.groupby([day, country])["r"].transform("mean")
    else:
        frame = frame.assign(w=np.asarray(weights, float), rw=lambda f: f.r * f.w)
        grp = frame.groupby([day, country])
        bench = grp["rw"].transform("sum") / grp["w"].transform("sum")
    return (frame["r"] - bench).rename("target")
