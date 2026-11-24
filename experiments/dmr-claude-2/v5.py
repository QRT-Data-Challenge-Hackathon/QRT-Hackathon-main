"""v1 tables -> v5: reversal x co-movement interactions (one change vs v1).

Short-term reversal of the stock's own move is the strongest signal; its strength may depend
on how idiosyncratic the stock is today (corr / beta to the country) and on its volatility.

  uv run python experiments/dmr-claude-2/v5.py data/derived data/derived_v5
"""

from __future__ import annotations

import shutil
import sys

import pandas as pd

from qrtc.config import SHARED_ROOT

SIGNALS = ["r_last1_rel", "r_last3_rel", "r_last12_rel", "idio_last12"]
MODIFIERS = ["corr_day", "beta_day", "vol_csz"]


def add_v5(df: pd.DataFrame) -> pd.DataFrame:
    new = {f"{s}_x_{m}": df[s] * df[m] for s in SIGNALS for m in MODIFIERS}
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def main(src: str, dst: str) -> int:
    src_dir, dst_dir = SHARED_ROOT / src, SHARED_ROOT / dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("X_train", "X_test", "X_submit"):
        add_v5(pd.read_parquet(src_dir / f"{name}.parquet")).to_parquet(
            dst_dir / f"{name}.parquet", index=False)
    shutil.copy(src_dir / "y_train.parquet", dst_dir / "y_train.parquet")
    print(f"wrote {dst_dir} (+{len(SIGNALS) * len(MODIFIERS)} interaction cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
