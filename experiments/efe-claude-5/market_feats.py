"""Issue #27: same-day market (DATE_ID x COUNTRY) features and each stock's intraday beta.

Uses only bars before the cutoff, from all 219,157 raw rows (train and test dates are disjoint,
so every date x country group is complete inside one split). Returns a frame indexed by ID.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qrtc import config

RAW = config.SHARED_ROOT / "data" / "raw" / "qrt-x-eth-hackathon-24-11-2026"
NB = 95  # 96 prices -> 95 five-minute returns, right-aligned (last = just before cutoff)
MKT_COLS = ["mkt_ret_day", "mkt_ret_12", "mkt_ret_3", "mkt_vol", "beta", "corr",
            "resid_vol", "beta_x_mkt12", "beta_x_mkt3"]


def _returns_matrix(series: pd.Series) -> np.ndarray:
    out = np.full((len(series), NB), np.nan)
    for i, p in enumerate(series):
        r = np.diff(np.log(np.asarray(p, dtype=float)))[-NB:]
        out[i, NB - len(r):] = r
    return out


def _nan_sum_last(m: np.ndarray, k: int) -> np.ndarray:
    return np.nansum(m[:, -k:], axis=1)


def market_features() -> pd.DataFrame:
    raw = pd.concat([pd.read_parquet(RAW / f, columns=["ID", "DATE_ID", "COUNTRY", "Price TS"])
                     for f in ("x_train.parquet", "x_test.parquet")], ignore_index=True)
    R = _returns_matrix(raw["Price TS"])
    key = raw["DATE_ID"].astype(str) + "_" + raw["COUNTRY"].astype(str)
    codes, uniq = pd.factorize(key)
    M = np.full_like(R, np.nan)
    for g in range(len(uniq)):  # equal-weight country index per bar
        idx = codes == g
        M[idx] = np.nanmean(R[idx], axis=0)
    ok = np.isfinite(R) & np.isfinite(M)
    r0, m0 = np.where(ok, R, 0.0), np.where(ok, M, 0.0)
    n = ok.sum(axis=1).clip(min=2)
    rm, mm = r0.sum(1) / n, m0.sum(1) / n
    cov = ((r0 - rm[:, None]) * (m0 - mm[:, None]) * ok).sum(1) / n
    var_m = (((m0 - mm[:, None]) ** 2) * ok).sum(1) / n
    var_r = (((r0 - rm[:, None]) ** 2) * ok).sum(1) / n
    beta = cov / (var_m + 1e-12)
    corr = cov / np.sqrt(var_m * var_r + 1e-18)
    resid = np.where(ok, R - beta[:, None] * M, np.nan)
    feats = pd.DataFrame({
        "mkt_ret_day": np.nansum(M, axis=1),
        "mkt_ret_12": _nan_sum_last(M, 12),
        "mkt_ret_3": _nan_sum_last(M, 3),
        "mkt_vol": np.nanstd(M, axis=1),
        "beta": beta,
        "corr": corr,
        "resid_vol": np.nanstd(resid, axis=1),
    }, index=raw["ID"].to_numpy())
    feats["beta_x_mkt12"] = feats["beta"] * feats["mkt_ret_12"]
    feats["beta_x_mkt3"] = feats["beta"] * feats["mkt_ret_3"]
    return feats[MKT_COLS]
