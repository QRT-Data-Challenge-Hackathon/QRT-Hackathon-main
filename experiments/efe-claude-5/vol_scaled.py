"""Issue #33: vol-scaled features for the calmer test period.

Scale-dependent features (returns, vols, idio paths) are divided by the same-day
DATE_ID x COUNTRY median of the 5-minute vol, so calm and stormy days share one scale.
Base = dmr-claude-3 beta-idio features + lgbm_shallow + |y| weights (repro 0.54335).

Usage: uv run python experiments/efe-claude-5/vol_scaled.py {add|replace}
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dmr-claude-3"))
import run_lane_a as lane  # noqa: E402

from qrtc import config, io, tracking  # noqa: E402
from qrtc.cv import make_splits  # noqa: E402
from qrtc.train import run_cv  # noqa: E402

SCALE_RE = re.compile(r"^(ret_\w+|bar1|vol|vol_last12|idio_w\d+|idio_vol\w*|idio_day|idio_seg\d"
                      r"|idio_drawdown|idio_runup|idio_from_high|idio_from_low)(_rel)?$")


def scale_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if SCALE_RE.match(c) and not c.endswith("_rank")]


def day_vol(keys: pd.DataFrame, vol: pd.Series) -> pd.Series:
    """Median 5-min vol of the stock's DATE_ID x COUNTRY group (all rows of that day)."""
    frame = keys.assign(v=vol.to_numpy())
    return frame.groupby(["DATE_ID", "COUNTRY"])["v"].transform("median").clip(lower=1e-6)


def scaled(X: pd.DataFrame, keys: pd.DataFrame, mode: str) -> pd.DataFrame:
    cols = scale_cols(X)
    dv = day_vol(keys, X["vol"]).to_numpy()
    new = X[cols].div(dv, axis=0)
    if mode == "replace":
        return X.assign(**{c: new[c].to_numpy() for c in cols})
    return pd.concat([X, new.add_suffix("_vs")], axis=1)


def main(mode: str) -> None:
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    largs = lane.parse_args(["--families", "beta,idio", "--base", "spec"])
    X, X_test = lane.features_for(ds, spec, largs, ["beta", "idio"])
    keys_tr = ds.train[["DATE_ID", "COUNTRY"]].reset_index(drop=True)
    keys_te = ds.test[["DATE_ID", "COUNTRY"]].reset_index(drop=True)
    X = scaled(X.reset_index(drop=True), keys_tr, mode)
    X_test = scaled(X_test.reset_index(drop=True), keys_te, mode)
    splits = make_splits(ds.train, ds.y_vector, spec, test=ds.test)
    weight = io.train_weight(ds, spec)
    print(f"mode={mode} feats={X.shape[1]} scaled={len(scale_cols(X_test))}")
    res = run_cv(spec, X, ds.y_vector, X_test, splits, "lgbm_shallow", None, weight,
                 ds.train[spec.date_col].to_numpy(), fit_weight=weight)
    run_id = tracking.save_run(res, spec, f"volscaled-{mode}-beta-idio-lgs-fw",
                               ds.train[spec.id_col].to_numpy(),
                               ds.test[spec.id_col].to_numpy(), notes="issue #33")
    print("saved", run_id)
    print(res.importance.head(12).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
