"""v1 tables -> v4: per-country slopes for the key signals (feature x 1[COUNTRY == c]).

Closing auctions differ by exchange, so reversal / beta effects may have country-specific
strength. A linear model on COUNTRY one-hots only learns intercepts (useless: the target is
zero-mean per country-day); these interactions let it learn per-country slopes.

  uv run python experiments/intraday_ridge/country_slopes.py data/derived data/derived_v4
"""

from __future__ import annotations

import shutil
import sys

import pandas as pd

from qrtc.config import SHARED_ROOT

KEY_FEATURES = ["r_last1_rel", "r_last3_rel", "r_last12_rel", "r_day_rel", "idio_last12",
                "beta_x_country_rday", "beta_x_country_rlast12", "r_first6_rel"]


def add_country_slopes(df: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    new = {f"{f}__{c}": df[f].where(df["COUNTRY"] == c, 0.0)
           for f in KEY_FEATURES for c in countries}
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def main(src: str, dst: str) -> int:
    src_dir, dst_dir = SHARED_ROOT / src, SHARED_ROOT / dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    countries = sorted(pd.read_parquet(src_dir / "X_submit.parquet", columns=["COUNTRY"])
                       ["COUNTRY"].unique())
    for name in ("X_train", "X_test", "X_submit"):
        df = pd.read_parquet(src_dir / f"{name}.parquet")
        add_country_slopes(df, countries).to_parquet(dst_dir / f"{name}.parquet", index=False)
    shutil.copy(src_dir / "y_train.parquet", dst_dir / "y_train.parquet")
    print(f"wrote {dst_dir} (+{len(KEY_FEATURES) * len(countries)} interaction cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
