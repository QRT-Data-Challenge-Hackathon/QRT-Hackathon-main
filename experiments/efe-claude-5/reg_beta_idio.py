"""Scoreboard #21 open idea 1: LightGBM *regression* on the real Close return with
dmr-claude-3's beta/idio features (+ |y| weights), scored by its sign.

Features: experiments/dmr-claude-3/run_lane_a.py (imported, base=spec, families beta,idio),
the same set as 1124-185723_dmr-claude-3_a-s-beta-idio-lgs-fw (0.5434).
Regression wrapper: same mapping as dmr-claude-2's regress_cv (sigmoid(pred / std)) so the
OOF blends with classifiers and scores with the contract metric.

Usage: uv run python experiments/efe-claude-5/reg_beta_idio.py [--fw] [--clip Q] [--model M]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dmr-claude-3"))
import run_lane_a as lane  # noqa: E402

from qrtc import config, io, tracking  # noqa: E402
from qrtc.cv import make_splits  # noqa: E402
from qrtc.train import CVResult, run_cv, score_predictions  # noqa: E402


def regress_cv(spec, ds, X, X_test, splits, model, weight, fit_weight, clip_q) -> CVResult:
    raw = ds.y_raw.iloc[:, 0].to_numpy(dtype=float)
    if clip_q:
        lo, hi = np.quantile(raw, [clip_q, 1 - clip_q])
        raw = np.clip(raw, lo, hi)
    rspec = replace(spec, task="regression", metric="pearson")
    res = run_cv(rspec, X, raw, X_test, splits, model, None, None, None,
                 fit_weight=fit_weight, verbose=False)
    scale = float(np.nanstd(res.oof)) or 1.0
    oof, test = (1.0 / (1.0 + np.exp(-a / scale)) for a in (res.oof, res.test_pred))
    groups = ds.train[spec.date_col].to_numpy()
    folds = [score_predictions(spec, ds.y_vector, oof, va, weight, groups) for _, va in splits]
    for i, s in enumerate(folds):
        print(f"  fold {i}: {spec.metric} = {s:.5f}")
    return replace(res, metric=spec.metric, score=float(np.mean(folds)),
                   score_std=float(np.std(folds)), fold_scores=tuple(folds), oof=oof,
                   test_pred=test)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fw", action="store_true", help="|y| sample weights")
    p.add_argument("--clip", type=float, default=0.0, help="winsorise fit target quantile")
    p.add_argument("--model", default="lgbm_shallow")
    a = p.parse_args()
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    largs = lane.parse_args(["--families", "beta,idio", "--base", "spec"])
    X, X_test = lane.features_for(ds, spec, largs, ["beta", "idio"])
    splits = make_splits(ds.train, ds.y_vector, spec, test=ds.test)
    weight = io.train_weight(ds, spec)
    fit_w = weight / weight.mean() if a.fw else None
    print(f"reg model={a.model} feats={X.shape[1]} fw={a.fw} clip={a.clip}")
    res = regress_cv(spec, ds, X, X_test, splits, a.model, weight, fit_w, a.clip)
    print(f"{spec.metric} = {res.score:.5f} ± {res.score_std:.5f}")
    slug = f"reg-beta-idio-{a.model}" + ("-fw" if a.fw else "") + (f"-c{a.clip}" if a.clip else "")
    run_id = tracking.save_run(res, spec, slug.replace(".", ""), ds.train[spec.id_col].to_numpy(),
                               ds.test[spec.id_col].to_numpy(), notes="scoreboard #21 idea 1")
    print("saved", run_id)


if __name__ == "__main__":
    main()
