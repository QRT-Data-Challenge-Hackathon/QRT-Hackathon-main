"""Derived table -> per-day rank-gauss copy (robust inputs for linear models).

Each numeric feature becomes the normal score of its percentile rank within its DATE_ID.
Only same-day rows are used, so nothing leaks across days or folds; outliers are bounded.

  uv run python experiments/dmr-claude-2/rankgauss.py data/derived data/derived_rg
"""

from __future__ import annotations

import shutil
import sys

import pandas as pd
from scipy.stats import norm

from qrtc.config import SHARED_ROOT

KEYS = {"ID", "DATE_ID", "STOCK_ID", "COUNTRY", "open_hour"}


def rank_gauss(df: pd.DataFrame, day: str = "DATE_ID") -> pd.DataFrame:
    cols = [c for c in df.columns if c not in KEYS and pd.api.types.is_numeric_dtype(df[c])]
    grouped = df.groupby(day)[cols]
    pct = grouped.rank(pct=True)
    n = grouped.transform("count")
    scores = norm.ppf(((pct * n) - 0.5) / n.where(n > 0))  # ranks mapped to (0, 1) exclusive
    return pd.concat([df.drop(columns=cols), pd.DataFrame(scores, columns=cols,
                                                          index=df.index)], axis=1)


def main(src: str, dst: str) -> int:
    src_dir, dst_dir = SHARED_ROOT / src, SHARED_ROOT / dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("X_train", "X_test", "X_submit"):
        df = pd.read_parquet(src_dir / f"{name}.parquet")
        rank_gauss(df).to_parquet(dst_dir / f"{name}.parquet", index=False)
    shutil.copy(src_dir / "y_train.parquet", dst_dir / "y_train.parquet")
    print(f"wrote {dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
