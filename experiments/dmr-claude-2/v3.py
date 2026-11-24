"""v1 tables -> v3: stock profile (average intraday beta / corr / vol over ALL days, prices
only, no target) and beta x country-move interactions.

The target is stock minus country over the last 30 minutes, which overlap US trading. When
the country moves into the close, high-beta stocks beat it, so beta x (predictable part of
the country's move) matters; a stock's average beta is far less noisy than one day's.

  uv run python experiments/dmr-claude-2/v3.py data/derived data/derived_v3
"""

from __future__ import annotations

import shutil
import sys

import pandas as pd

from qrtc.config import SHARED_ROOT

PROFILE = ["beta_day", "corr_day", "vol", "idio_vol"]
COUNTRY_MOVES = ["r_first6", "r_last1", "r_last3", "r_last6", "r_last12", "r_day"]


def add_v3(df: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(profile, on="STOCK_ID", how="left", validate="many_to_one")
    for c in COUNTRY_MOVES:
        move = out[f"{c}_DATE_ID_COUNTRY_mean"]
        out[f"sbeta_x_country_{c}"] = out["stock_beta_day"] * move
        if f"beta_x_country_{c}" not in out:
            out[f"beta_x_country_{c}"] = out["beta_day"] * move
    out["beta_dev"] = out["beta_day"] - out["stock_beta_day"]
    return out


def main(src: str, dst: str) -> int:
    src_dir, dst_dir = SHARED_ROOT / src, SHARED_ROOT / dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    every = pd.read_parquet(src_dir / "X_submit.parquet")  # all ids, features only
    profile = every.groupby("STOCK_ID")[PROFILE].median().add_prefix("stock_").reset_index()
    for name in ("X_train", "X_test", "X_submit"):
        df = pd.read_parquet(src_dir / f"{name}.parquet")
        add_v3(df, profile).to_parquet(dst_dir / f"{name}.parquet", index=False)
    shutil.copy(src_dir / "y_train.parquet", dst_dir / "y_train.parquet")
    print(f"wrote {dst_dir} (+{profile.shape[1] - 1} profile cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
