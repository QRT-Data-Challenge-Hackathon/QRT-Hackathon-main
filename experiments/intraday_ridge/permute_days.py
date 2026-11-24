"""Copy a derived table with DATE_ID relabelled by a random bijection -> new date_blocks folds.

DATE_ID only defines groups (it is shuffled and dropped as a feature), so a bijection keeps
every feature identical but cuts the days into different CV folds: a fresh partition to test
whether a candidate's CV edge survives other folds (selection-overfitting check).

  uv run python experiments/intraday_ridge/permute_days.py data/derived data/derived_p1 1
"""

from __future__ import annotations

import shutil
import sys

import numpy as np
import pandas as pd

from qrtc.config import SHARED_ROOT


def main(src: str, dst: str, seed: str) -> int:
    src_dir, dst_dir = SHARED_ROOT / src, SHARED_ROOT / dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    days = np.sort(pd.read_parquet(src_dir / "X_submit.parquet", columns=["DATE_ID"])
                   ["DATE_ID"].unique())
    mapping = dict(zip(days, np.random.default_rng(int(seed)).permutation(days), strict=True))
    for name in ("X_train", "X_test", "X_submit"):
        df = pd.read_parquet(src_dir / f"{name}.parquet")
        df.assign(DATE_ID=df["DATE_ID"].map(mapping)).to_parquet(dst_dir / f"{name}.parquet",
                                                                 index=False)
    shutil.copy(src_dir / "y_train.parquet", dst_dir / "y_train.parquet")
    print(f"wrote {dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
