"""Lane A (dmr-claude-3): cross-sectional features for the intraday 5-minute price panel.

Data (QRT x ETH, 24 Nov 2026): one row = one stock on one day; ``Price TS`` holds that day's
5-minute prices from the open up to the cutoff (close - 30 min), 96 bars on a full day,
48 on a half day. Target ``Close return`` = the stock's cutoff-to-close return minus the
equal-weight mean of its country that day (it sums to 0 per DATE_ID x COUNTRY); the metric
is accuracy on its sign, weighted by its size.

Every feature uses only the row's own day: its own price path and same-day cross-sections
(day, or day x country). Train and test days are disjoint, so nothing crosses the split.
All functions are pure: they return new frames and never modify their inputs.

Feature families (each returns only the new columns, indexed like the input):
  ret    own path: window log returns ending at the cutoff, day return, vol, range
  rel    ret minus the day x country mean (the target's own reference) and the day mean
  rank   percentile rank / clipped z-score of ret within day x country
  beta   intraday beta to the equal-weight country path, residual (idio) window returns
  ctx    country context: country mean / dispersion of window returns, country vol
  bars   last 6 bars relative to the country path, vol-scaled relative window moves
  shape  trend slope / R^2, drawdown, run-up, distance to the day's high/low, autocorrelation
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from qrtc.features import cs_rank, cs_zscore

PRICE_COL = "Price TS"
WINDOWS = (1, 2, 3, 6, 12, 24, 48)  # bars of 5 min ending at the cutoff (5 min .. 4 h)
VOL_TAIL = 12                       # bars for the "recent" volatility (last hour)
EPS = 1e-12


def price_matrix(series: pd.Series, n_bars: int | None = None) -> np.ndarray:
    """Stack per-row price arrays right-aligned at the cutoff (shorter days left-padded NaN)."""
    arrays = [np.asarray(a, dtype=float) for a in series]
    width = n_bars or max(len(a) for a in arrays)
    out = np.full((len(arrays), width), np.nan)
    for i, arr in enumerate(arrays):
        tail = arr[-width:]
        out[i, width - len(tail):] = tail
    return out


def log_prices(prices: np.ndarray) -> np.ndarray:
    """log(price) with non-positive prices as missing, gaps filled forward then backward."""
    clean = np.where(prices > 0, prices, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = np.log(clean)
    return pd.DataFrame(logged).ffill(axis=1).bfill(axis=1).to_numpy()


def _windows(n_bars: int) -> list[int]:
    return [k for k in WINDOWS if k < n_bars]


def own_path(lp: np.ndarray, n_valid: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Family ``ret``: window returns ending at the cutoff, day return, vol, range, position."""
    rets = np.diff(lp, axis=1)
    last = lp[:, -1]
    hi, lo = np.nanmax(lp, axis=1), np.nanmin(lp, axis=1)
    data = {f"ret_w{k}": last - lp[:, -1 - k] for k in _windows(lp.shape[1])}
    data["ret_day"] = last - lp[:, 0]
    data["vol_day"] = np.nanstd(rets, axis=1)
    data["vol_tail"] = np.nanstd(rets[:, -VOL_TAIL:], axis=1)
    data["range_day"] = hi - lo
    data["pos_in_range"] = (last - lo) / (hi - lo + EPS)
    data["absret_last"] = np.abs(rets[:, -1])
    data["n_bars"] = n_valid.astype(float)
    return pd.DataFrame(data, index=index)


def ret_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c.startswith("ret_")]


def relative(keys: pd.DataFrame, ret: pd.DataFrame, date: str, country: str) -> pd.DataFrame:
    """Family ``rel``: returns minus the day x country mean and minus the day mean."""
    rcols, vcols = ret_columns(ret), ["vol_day", "vol_tail"]
    frame = pd.concat([keys[[date, country]], ret[rcols + vcols]], axis=1)
    by_dc = frame.groupby([date, country])[rcols + vcols].transform("mean")
    by_d = frame.groupby(date)[rcols].transform("mean")
    rel_c = (frame[rcols] - by_dc[rcols]).add_suffix("_xc")
    rel_d = (frame[rcols] - by_d).add_suffix("_xd")
    vol_ratio = (frame[vcols] / (by_dc[vcols] + EPS)).add_suffix("_ratio_c")
    return pd.concat([rel_c, rel_d, vol_ratio], axis=1)


def ranks(keys: pd.DataFrame, ret: pd.DataFrame, date: str, country: str) -> pd.DataFrame:
    """Family ``rank``: percentile rank and clipped z-score within day x country."""
    rcols = ret_columns(ret)
    cols = rcols + ["vol_day", "vol_tail"]
    frame = pd.concat([keys[[date, country]], ret[cols]], axis=1)
    ranked = cs_rank(frame, cols, [date, country])
    zs = cs_zscore(frame, rcols, [date, country])
    return pd.concat([ranked[[f"{c}_csrank" for c in cols]], zs[[f"{c}_csz" for c in rcols]]],
                     axis=1)


def _group_mean_path(rets: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Equal-weight mean return path of each row's group, broadcast back to the rows."""
    means = pd.DataFrame(rets).groupby(codes).mean()
    return means.loc[codes].to_numpy()


def beta_residual(keys: pd.DataFrame, lp: np.ndarray, date: str, country: str) -> pd.DataFrame:
    """Family ``beta``: intraday beta to the country path and residual (idio) window returns."""
    rets = np.nan_to_num(np.diff(lp, axis=1))
    codes = keys.groupby([date, country], sort=False).ngroup().to_numpy()
    cpath = _group_mean_path(rets, codes)
    rc, cc = rets - rets.mean(axis=1, keepdims=True), cpath - cpath.mean(axis=1, keepdims=True)
    var_c = (cc ** 2).sum(axis=1)
    beta = (rc * cc).sum(axis=1) / (var_c + EPS)
    resid = rets - beta[:, None] * cpath
    corr = (rc * cc).sum(axis=1) / (np.sqrt((rc ** 2).sum(axis=1) * var_c) + EPS)
    data = {f"idio_w{k}": resid[:, -k:].sum(axis=1) for k in _windows(lp.shape[1])}
    data.update(beta=beta, corr_country=corr, idio_vol=resid.std(axis=1),
                idio_vol_tail=resid[:, -VOL_TAIL:].std(axis=1), idio_day=resid.sum(axis=1))
    return pd.DataFrame(data, index=keys.index)


def bar_profile(keys: pd.DataFrame, lp: np.ndarray, date: str, country: str,
                n_last: int = 6) -> pd.DataFrame:
    """Family ``bars``: last bars relative to the country, and vol-scaled relative moves."""
    rets = np.nan_to_num(np.diff(lp, axis=1))
    codes = keys.groupby([date, country], sort=False).ngroup().to_numpy()
    rel = rets - _group_mean_path(rets, codes)
    vol = rets.std(axis=1) + EPS
    data = {f"bar{j}_xc": rel[:, -j] for j in range(1, n_last + 1)}
    data.update({f"bar{j}_xc_z": rel[:, -j] / vol for j in (1, 2, 3)})
    for k in (1, 3, 6, 12, 48):
        data[f"relsum_w{k}_z"] = rel[:, -k:].sum(axis=1) / (vol * np.sqrt(k))
    data["relsum_day_z"] = rel.sum(axis=1) / (vol * np.sqrt(rel.shape[1]))
    return pd.DataFrame(data, index=keys.index)


def path_shape(keys: pd.DataFrame, lp: np.ndarray, date: str, country: str) -> pd.DataFrame:
    """Family ``shape``: trend slope / R^2, drawdown, run-up, distance to high/low, autocorr."""
    rets = np.nan_to_num(np.diff(lp, axis=1))
    tc = np.arange(lp.shape[1]) - (lp.shape[1] - 1) / 2
    lpc = lp - lp.mean(axis=1, keepdims=True)
    slope = (lpc * tc).sum(axis=1) / (tc ** 2).sum()
    resid = lpc - slope[:, None] * tc
    data = {
        "trend_slope": slope,
        "trend_r2": 1 - (resid ** 2).sum(axis=1) / ((lpc ** 2).sum(axis=1) + EPS),
        "trend_resid_last": resid[:, -1],
        "drawdown": (lp - np.maximum.accumulate(lp, axis=1)).min(axis=1),
        "runup": (lp - np.minimum.accumulate(lp, axis=1)).max(axis=1),
        "from_high": lp[:, -1] - lp.max(axis=1),
        "from_low": lp[:, -1] - lp.min(axis=1),
        "autocorr1": (rets[:, 1:] * rets[:, :-1]).sum(axis=1) / ((rets ** 2).sum(axis=1) + EPS),
        "zero_bar_frac": (rets == 0).mean(axis=1),
    }
    return pd.DataFrame(data, index=keys.index)


def context(keys: pd.DataFrame, ret: pd.DataFrame, date: str, country: str) -> pd.DataFrame:
    """Family ``ctx``: the country's mean / dispersion of window returns and its vol that day."""
    rcols = ret_columns(ret)
    frame = pd.concat([keys[[date, country]], ret[rcols + ["vol_day"]]], axis=1)
    grouped = frame.groupby([date, country])
    mean = grouped[rcols + ["vol_day"]].transform("mean").add_suffix("_cmean")
    disp = grouped[rcols].transform("std").add_suffix("_cstd")
    count = grouped[rcols[0]].transform("size").rename("n_country").to_frame()
    return pd.concat([mean, disp, count], axis=1)


FAMILIES: dict[str, Callable[..., pd.DataFrame]] = {     # (keys, ret, date, country)
    "rel": relative, "rank": ranks, "ctx": context,
}
PATH_FAMILIES: dict[str, Callable[..., pd.DataFrame]] = {  # (keys, log prices, date, country)
    "beta": beta_residual, "bars": bar_profile, "shape": path_shape,
}
ALL_FAMILIES = ("ret", *FAMILIES, *PATH_FAMILIES)


def build_features(df: pd.DataFrame, date: str, country: str, families: Sequence[str],
                   price_col: str = PRICE_COL, n_bars: int = 96) -> pd.DataFrame:
    """All requested families for one frame (train or test), indexed like ``df``."""
    unknown = set(families) - set(ALL_FAMILIES)
    if unknown:
        raise ValueError(f"unknown families {sorted(unknown)}; known {ALL_FAMILIES}")
    prices = price_matrix(df[price_col], n_bars)
    n_valid = (prices > 0).sum(axis=1)
    lp = log_prices(prices)
    keys = df[[date, country]]
    ret = own_path(lp, n_valid, df.index)
    parts = [ret] if "ret" in families else []
    parts += [FAMILIES[f](keys, ret, date, country) for f in families if f in FAMILIES]
    parts += [PATH_FAMILIES[f](keys, lp, date, country) for f in families if f in PATH_FAMILIES]
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
