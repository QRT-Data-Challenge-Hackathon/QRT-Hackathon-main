"""Issue #36: the Captain's interaction list on top of beta-idio lgs-fw (repro 0.54335).

- beta_dev = beta - DATE_ID x COUNTRY mean beta; x country move (day, last 6 bars)
  (country move = ret_k - ret_k_rel, the group mean the spec already subtracts)
- extreme flags: bottom/top 10% of ret_day_rank and idio_day_t_csrank within (day, country)
- illiquidity (dmr-claude-3 tick profile: flat_frac, roll_spread) x extremes and x idio_day_t
Raw tick columns are dropped (they did not help alone, 0.5398): only the interactions are added.

Usage: uv run python experiments/efe-claude-5/interactions.py
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


def interactions(X: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    g = (keys["DATE_ID"].astype(str) + "_" + keys["COUNTRY"].astype(str)).to_numpy()
    beta_dev = X["beta"] - X["beta"].groupby(g).transform("mean")
    cm_day = X["ret_day"] - X["ret_day_rel"]
    cm_6 = X["ret_6"] - X["ret_6_rel"]
    lo = (X["ret_day_rank"] <= 0.1).astype(float) - (X["ret_day_rank"] >= 0.9).astype(float)
    lo_t = ((X["idio_day_t_csrank"] <= 0.1).astype(float)
            - (X["idio_day_t_csrank"] >= 0.9).astype(float))
    flat_r = X["flat_frac"].groupby(g).rank(pct=True)
    roll_r = X["roll_spread"].groupby(g).rank(pct=True)
    return pd.DataFrame({
        "ix_beta_dev": beta_dev,
        "ix_betadev_x_cday": beta_dev * cm_day,
        "ix_betadev_x_c6": beta_dev * cm_6,
        "ix_extreme_day": lo,
        "ix_extreme_idio": lo_t,
        "ix_flat_x_extreme": flat_r * lo,
        "ix_roll_x_extreme": roll_r * lo,
        "ix_flat_x_idio": flat_r * X["idio_day_t"],
        "ix_roll_x_idio": roll_r * X["idio_day_t"],
    }, index=X.index)


def build(X: pd.DataFrame, keys: pd.DataFrame, tick_cols: list[str]) -> pd.DataFrame:
    X = X.reset_index(drop=True)
    ix = interactions(X, keys.reset_index(drop=True))
    return pd.concat([X.drop(columns=tick_cols), ix], axis=1)


def main() -> None:
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    base_args = lane.parse_args(["--families", "beta,idio", "--base", "spec"])
    Xb, _ = lane.features_for(ds, spec, base_args, ["beta", "idio"])
    largs = lane.parse_args(["--families", "beta,idio,tick", "--base", "spec"])
    X, X_test = lane.features_for(ds, spec, largs, ["beta", "idio", "tick"])
    tick_cols = [c for c in X.columns if c not in Xb.columns]
    X = build(X, ds.train[["DATE_ID", "COUNTRY"]], tick_cols)
    X_test = build(X_test, ds.test[["DATE_ID", "COUNTRY"]], tick_cols)
    assert np.isfinite(X.select_dtypes("number").to_numpy(dtype=float)).any()
    splits = make_splits(ds.train, ds.y_vector, spec, test=ds.test)
    weight = io.train_weight(ds, spec)
    print(f"feats={X.shape[1]} (base {Xb.shape[1]}, dropped tick {len(tick_cols)})")
    res = run_cv(spec, X, ds.y_vector, X_test, splits, "lgbm_shallow", None, weight,
                 ds.train[spec.date_col].to_numpy(), fit_weight=weight)
    run_id = tracking.save_run(res, spec, "ix-pack-beta-idio-lgs-fw",
                               ds.train[spec.id_col].to_numpy(),
                               ds.test[spec.id_col].to_numpy(), notes="issue #36")
    print("saved", run_id)
    imp = res.importance
    print(imp[[c for c in imp.index if str(c).startswith("ix_")]].to_string())


if __name__ == "__main__":
    main()
