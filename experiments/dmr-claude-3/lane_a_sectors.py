"""Lane A (dmr-claude-3): sectors recovered from co-movement.

The data has no sector column. Stocks of one sector move together beyond their country, so
cluster the stocks by the correlation of their country-residual 5-minute return paths over
every day (train + test features, never the target), then add residual features vs the
day x sector path. Pure functions; inputs are never modified.
"""

from __future__ import annotations

import lane_a_features as laf
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def residual_paths(df: pd.DataFrame, date: str, country: str, n_bars: int = 96) -> np.ndarray:
    """Per-row residual 5-minute returns vs the day x country path (n_rows x n_bars-1)."""
    lp = laf.log_prices(laf.price_matrix(df[laf.PRICE_COL], n_bars))
    return laf._country_fit(df[[date, country]], lp, date, country)[3]


def stock_clusters(keys: pd.DataFrame, resid: np.ndarray, stock: str, date: str, k: int,
                   seed: int) -> pd.Series:
    """KMeans on the stocks' standardised residual paths (stock x (day, bar)) -> cluster id."""
    stocks, s_idx = np.unique(keys[stock].to_numpy(), return_inverse=True)
    days, d_idx = np.unique(keys[date].to_numpy(), return_inverse=True)
    paths = np.zeros((len(stocks), len(days), resid.shape[1]), dtype=np.float32)
    paths[s_idx, d_idx] = resid
    flat = paths.reshape(len(stocks), -1)
    flat = (flat - flat.mean(axis=1, keepdims=True)) / (flat.std(axis=1, keepdims=True) + 1e-12)
    labels = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(flat)
    return pd.Series(labels, index=stocks, name="sector")


def sector_features(keys: pd.DataFrame, resid: np.ndarray, sector: pd.Series, stock: str,
                    date: str) -> pd.DataFrame:
    """Sector id, the sector's residual path (its move vs the country) and the stock vs it."""
    sec = sector.reindex(keys[stock].to_numpy()).to_numpy()
    codes = pd.DataFrame({"d": keys[date].to_numpy(), "s": sec}).groupby(
        ["d", "s"], sort=False).ngroup().to_numpy()
    spath = laf._group_mean_path(resid, codes)
    rel = resid - spath
    vol = rel.std(axis=1) + laf.EPS
    data = {"sector": sec}
    for k in (6, 12, 48):
        data[f"sector_w{k}"] = spath[:, -k:].sum(axis=1)
        data[f"idio_sec_w{k}"] = rel[:, -k:].sum(axis=1)
    data.update(sector_day=spath.sum(axis=1), idio_sec_day=rel.sum(axis=1),
                idio_sec_day_t=rel.sum(axis=1) / (vol * np.sqrt(rel.shape[1])),
                n_sector=pd.Series(codes).map(pd.Series(codes).value_counts()).to_numpy())
    return pd.DataFrame(data, index=keys.index)


def sector_ranks(keys: pd.DataFrame, feats: pd.DataFrame, date: str, country: str
                 ) -> pd.DataFrame:
    """Ranks: the stock vs its sector that day, and the sector's move vs the country's others."""
    frame = pd.concat([keys[[date, country]], feats], axis=1)
    in_sec = frame.groupby([date, "sector"])[["idio_sec_day_t", "idio_sec_w48"]].rank(pct=True)
    sec_in_c = frame.groupby([date, country])[["sector_day", "sector_w48", "sector_w6"]].rank(
        pct=True)
    return pd.concat([in_sec.add_suffix("_secrank"), sec_in_c.add_suffix("_crank")], axis=1)
