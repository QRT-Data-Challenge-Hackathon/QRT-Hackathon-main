"""Issue #34: US-open and US-macro-release reaction features.

The 96-bar paths show volatility spikes in the return into bar 79 (US cash open, 1.21x the
median bar) and bar 67 (08:30 ET US macro releases, 1.05x vs ~0.87 around it). Windows below
use 0-based price indices (index i = bar i+1); everything ends at the cutoff (index 95).
Only full 96-bar days get values (48-bar half days close before the US open): NaN otherwise.
Base = dmr-claude-3 beta-idio + lgbm_shallow + |y| weights (repro 0.54335).

Usage: uv run python experiments/efe-claude-5/us_open.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dmr-claude-3"))
import run_lane_a as lane  # noqa: E402

from qrtc import config, io, tracking  # noqa: E402
from qrtc.cv import make_splits  # noqa: E402
from qrtc.train import run_cv  # noqa: E402

RAW = config.SHARED_ROOT / "data" / "raw" / "qrt-x-eth-hackathon-24-11-2026"
WINDOWS = {"macro": (65, 67), "pre_us": (66, 77), "us_open": (77, 81), "since_us": (77, 95),
           "us_late": (81, 95)}


def log_price_matrix(series: pd.Series) -> np.ndarray:
    lp = np.full((len(series), 96), np.nan)
    for i, p in enumerate(series):
        if len(p) == 96:
            lp[i] = np.log(np.asarray(p, dtype=float))
    return pd.DataFrame(lp).ffill(axis=1).to_numpy()  # carry last price over missing bars


def us_features() -> pd.DataFrame:
    raw = pd.concat([pd.read_parquet(RAW / f, columns=["ID", "DATE_ID", "COUNTRY", "Price TS"])
                     for f in ("x_train.parquet", "x_test.parquet")], ignore_index=True)
    lp = log_price_matrix(raw["Price TS"])
    r = np.diff(lp, axis=1)                                   # 95 returns, r[:, j] into bar j+2
    out = pd.DataFrame({f"us_{k}": lp[:, b] - lp[:, a] for k, (a, b) in WINDOWS.items()})
    pre, post = r[:, 40:76], r[:, 76:95]
    out["us_vol_ratio"] = np.nanstd(post, axis=1) / (np.nanstd(pre, axis=1) + 1e-9)
    grp = raw["DATE_ID"].astype(str) + "_" + raw["COUNTRY"].astype(str)
    mkt = pd.DataFrame(r[:, 76:95]).groupby(grp.to_numpy()).transform("mean").to_numpy()
    rc, mc = r[:, 76:95] - np.nanmean(r[:, 76:95], 1, keepdims=True), mkt - np.nanmean(mkt, 1, keepdims=True)
    out["us_beta_post"] = np.nansum(rc * mc, 1) / (np.nansum(mc * mc, 1) + 1e-12)
    keys = pd.DataFrame({"g": grp})
    for k in WINDOWS:
        col = out[f"us_{k}"]
        out[f"us_{k}_idio"] = col - col.groupby(keys["g"]).transform("mean")
        out[f"us_{k}_rank"] = out[f"us_{k}_idio"].groupby(keys["g"]).rank(pct=True)
    out["us_beta_x_open"] = out["us_beta_post"] * (out["us_us_open"] - out["us_us_open_idio"])
    return out.set_index(raw["ID"].to_numpy())


def main() -> None:
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    largs = lane.parse_args(["--families", "beta,idio", "--base", "spec"])
    X, X_test = lane.features_for(ds, spec, largs, ["beta", "idio"])
    us = us_features()
    X = pd.concat([X.reset_index(drop=True),
                   us.loc[ds.train[spec.id_col]].reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True),
                        us.loc[ds.test[spec.id_col]].reset_index(drop=True)], axis=1)
    splits = make_splits(ds.train, ds.y_vector, spec, test=ds.test)
    weight = io.train_weight(ds, spec)
    print(f"feats={X.shape[1]} us_cols={us.shape[1]}")
    res = run_cv(spec, X, ds.y_vector, X_test, splits, "lgbm_shallow", None, weight,
                 ds.train[spec.date_col].to_numpy(), fit_weight=weight)
    run_id = tracking.save_run(res, spec, "usopen-beta-idio-lgs-fw",
                               ds.train[spec.id_col].to_numpy(),
                               ds.test[spec.id_col].to_numpy(), notes="issue #34")
    print("saved", run_id)
    imp = res.importance
    print(imp[[c for c in imp.index if str(c).startswith("us_")]].head(12).to_string())
    print("top:", ", ".join(imp.head(10).index.astype(str)))


if __name__ == "__main__":
    main()
