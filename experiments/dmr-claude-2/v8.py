"""v1 tables -> v8: sector-relative moves from correlation clusters (one change vs v1).

There is no sector column, so stocks are clustered (KMeans, K sectors) on the correlation of
their country-relative 5-minute returns over ALL days (prices only, never the target). Then
each signal is split into stock-vs-sector and sector-vs-country parts.

  uv run python experiments/dmr-claude-2/v8.py data/derived data/derived_v8 [K]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
import build as B  # noqa: E402
import intraday as I  # noqa: E402

from qrtc.config import SHARED_ROOT  # noqa: E402

SIGNALS = ["r_last1_rel", "r_last3_rel", "r_last12_rel", "r_day_rel", "idio_last12"]


def relative_paths(raw: Path) -> pd.DataFrame:
    """Rows (DATE_ID, STOCK_ID, COUNTRY) x 95 bars of stock-minus-country log returns."""
    train, test, _ = B.load(raw)
    flat = pd.concat([B.flatten(train), B.flatten(test)], ignore_index=True)
    bars = [c for c in flat.columns if c.startswith("P")]
    rets = I.log_prices(flat[bars]).diff(axis=1).iloc[:, 1:]
    mkt = rets.groupby([flat["DATE_ID"], flat["COUNTRY"]]).transform("mean")
    rel = (rets - mkt).fillna(0.0)
    return pd.concat([flat[["DATE_ID", "STOCK_ID"]], rel], axis=1)


def sectors(rel: pd.DataFrame, k: int) -> pd.Series:
    days = np.sort(rel["DATE_ID"].unique())
    stocks = np.sort(rel["STOCK_ID"].unique())
    nb = rel.shape[1] - 2
    cube = np.zeros((len(stocks), len(days), nb), dtype=np.float32)
    si, di = np.searchsorted(stocks, rel["STOCK_ID"]), np.searchsorted(days, rel["DATE_ID"])
    cube[si, di] = rel.iloc[:, 2:].to_numpy(np.float32)
    corr = np.nan_to_num(np.corrcoef(cube.reshape(len(stocks), -1)))
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(corr)
    return pd.Series(labels, index=stocks, name="sector")


def add_v8(df: pd.DataFrame, sector_of: pd.Series) -> pd.DataFrame:
    out = df.assign(sector=df["STOCK_ID"].map(sector_of).astype(int))
    means = out.groupby(["DATE_ID", "sector"])[SIGNALS].transform("mean")
    for s in SIGNALS:
        out[f"{s}_vs_sector"] = out[s] - means[s]
        out[f"sector_{s}"] = means[s]
    return out.drop(columns="sector")


def main(src: str, dst: str, k: str = "25") -> int:
    src_dir, dst_dir = SHARED_ROOT / src, SHARED_ROOT / dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    sector_of = sectors(relative_paths(SHARED_ROOT / "data" / "raw"), int(k))
    print("sector sizes:", sector_of.value_counts().sort_values().tolist())
    for name in ("X_train", "X_test", "X_submit"):
        add_v8(pd.read_parquet(src_dir / f"{name}.parquet"), sector_of).to_parquet(
            dst_dir / f"{name}.parquet", index=False)
    shutil.copy(src_dir / "y_train.parquet", dst_dir / "y_train.parquet")
    print(f"wrote {dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
