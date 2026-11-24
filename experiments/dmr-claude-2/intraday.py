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
    start = lp.notna().to_numpy().argmax(axis=1)  # rows are right-aligned (short days)
    at6 = lp.to_numpy()[np.arange(len(lp)), np.minimum(start + 6, n - 1)]
    out = {"r_day": last - first, "r_first6": pd.Series(at6, index=lp.index) - first}
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
    """Multi-day rolling features per stock (only valid if ``day`` is in time order).

    No overnight gap: tonight's prices are rescaled per row, so levels do not link days."""
    out = df.sort_values([stock, day]).copy()
    g = out.groupby(stock)
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


def country_beta(visible: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Intraday beta / correlation of each stock's 5-min returns to its country's mean return.

    The target is stock minus country, so a high-beta stock "beats" its country whenever the
    country rallies into the close: beta x country momentum is a natural interaction."""
    rets = log_prices(visible).diff(axis=1).iloc[:, 1:]
    rets = rets.set_axis(range(rets.shape[1]), axis=1).reset_index(drop=True)
    mkt = rets.groupby([groups.iloc[:, 0].to_numpy(), groups.iloc[:, 1].to_numpy()])
    mkt = mkt.transform("mean")
    r, m = rets.to_numpy(), mkt.to_numpy()
    ok = ~(np.isnan(r) | np.isnan(m))
    r, m = np.where(ok, r, 0.0), np.where(ok, m, 0.0)
    cnt = ok.sum(axis=1).clip(min=2)
    rc = r - (r.sum(axis=1) / cnt)[:, None] * ok
    mc = m - (m.sum(axis=1) / cnt)[:, None] * ok
    cov, var_m, var_r = (rc * mc).sum(1), (mc * mc).sum(1), (rc * rc).sum(1)
    beta = np.where(var_m > 0, cov / np.where(var_m > 0, var_m, 1), np.nan)
    corr = np.where((var_m > 0) & (var_r > 0), cov / np.sqrt(var_m * var_r + 1e-300), np.nan)
    resid = rc - beta[:, None] * mc
    return pd.DataFrame({"beta_day": beta, "corr_day": corr,
                         "idio_vol": np.sqrt((resid ** 2).sum(1) / cnt),
                         "idio_last12": resid[:, -12:].sum(1)}, index=visible.index)


def build_panel(visible: pd.DataFrame, country_of: pd.Series, day: str, stock: str,
                country: str, history: bool = False) -> pd.DataFrame:
    """Full feature table for all (day, stock) rows; ``country_of`` maps stock -> country."""
    df = bar_features(visible).reset_index()
    df[country] = df[stock].map(country_of)
    if df[country].isna().any():
        raise ValueError(f"{df[country].isna().sum()} rows have no country")
    beta = country_beta(visible, df[[day, country]])
    df = pd.concat([df, beta.reset_index(drop=True)], axis=1)
    if history:
        df = add_history(df, day, stock)
    df = add_relative(df, day, country)
    df["beta_x_country_rlast12"] = df["beta_day"] * df[f"r_last12_{day}_{country}_mean"]
    df["beta_x_country_rday"] = df["beta_day"] * df[f"r_day_{day}_{country}_mean"]
    return df.drop(columns=["first_logp", "last_logp"])


def stack_prices(series: pd.Series, width: int) -> pd.DataFrame:
    """Column of price lists -> matrix right-aligned at the cutoff (short days padded left)."""
    out = np.full((len(series), width), np.nan)
    for i, arr in enumerate(series.to_numpy()):
        a = np.asarray(arr, dtype=float)[-width:]
        out[i, width - len(a):] = a
    return pd.DataFrame(out, index=series.index, columns=[f"P{j:03d}" for j in range(width)])


def detect_bar_cols(df: pd.DataFrame, exclude: Sequence[str]) -> list[str]:
    """Numeric price columns in file order (everything numeric that is not a key/target)."""
    skip = set(exclude)
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def features_from_rows(train: pd.DataFrame, test: pd.DataFrame, bar_cols: Sequence[str],
                       id_col: str, day: str, stock: str, country: str, history: bool = False
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
    panel = build_panel(visible, country_of, day, stock, country, history=history)
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
