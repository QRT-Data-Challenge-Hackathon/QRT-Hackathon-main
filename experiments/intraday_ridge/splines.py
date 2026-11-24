"""v1 tables -> v6: piecewise-linear spline basis for the strongest signals (one change vs v1).

Ridge is linear; trees overfit. Degree-1 splines with quantile knots let ridge bend the
reversal curve (e.g. saturation for big moves) while staying linear in its parameters.
The transformer sees features only (all ids), never the target.

  uv run python experiments/intraday_ridge/splines.py data/derived data/derived_v6
"""

from __future__ import annotations

import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer

from qrtc.config import SHARED_ROOT

SIGNALS = ["r_last1_rel", "r_last3_rel", "r_last12_rel", "r_day_rel", "idio_last12",
           "beta_x_country_rday", "beta_x_country_rlast12"]


def fit_splines(every: pd.DataFrame) -> dict[str, SplineTransformer]:
    fitted = {}
    for col in SIGNALS:
        x = every[[col]].to_numpy(float)
        lo, hi = np.nanquantile(x, [0.005, 0.995])
        spline = SplineTransformer(n_knots=6, degree=1, knots="quantile", extrapolation="linear")
        fitted[col] = (spline.fit(np.clip(x[~np.isnan(x[:, 0])], lo, hi)), lo, hi)
    return fitted


def add_v6(df: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    parts = []
    for col, (spline, lo, hi) in fitted.items():
        x = np.clip(df[[col]].to_numpy(float), lo, hi)
        basis = spline.transform(np.nan_to_num(x, nan=float(np.nanmedian(x))))
        parts.append(pd.DataFrame(basis, index=df.index,
                                  columns=[f"{col}_sp{i}" for i in range(basis.shape[1])]))
    return pd.concat([df, *parts], axis=1)


def main(src: str, dst: str) -> int:
    src_dir, dst_dir = SHARED_ROOT / src, SHARED_ROOT / dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    fitted = fit_splines(pd.read_parquet(src_dir / "X_submit.parquet"))
    for name in ("X_train", "X_test", "X_submit"):
        add_v6(pd.read_parquet(src_dir / f"{name}.parquet"), fitted).to_parquet(
            dst_dir / f"{name}.parquet", index=False)
    shutil.copy(src_dir / "y_train.parquet", dst_dir / "y_train.parquet")
    print(f"wrote {dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
