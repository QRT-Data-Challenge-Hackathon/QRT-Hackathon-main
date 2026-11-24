"""Issue #37: "the board votes": bag the beta-idio lgs-fw model with row + feature subsampling.

N members, each lgbm_shallow + |y| weights with subsample=0.8, colsample_bytree=0.7 and its own
model seed; the contract splits are built once with the spec seed (identical folds to everyone).
OOF / test = mean of the members' probabilities. Base: repro 0.54335.

Usage: uv run python experiments/efe-claude-5/committee.py [N]
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dmr-claude-3"))
import run_lane_a as lane  # noqa: E402

from qrtc import config, io, tracking  # noqa: E402
from qrtc.cv import make_splits  # noqa: E402
from qrtc.train import run_cv, score_predictions  # noqa: E402

BAG = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.7)


def main(n: int) -> None:
    t0 = time.time()
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    largs = lane.parse_args(["--families", "beta,idio", "--base", "spec"])
    X, X_test = lane.features_for(ds, spec, largs, ["beta", "idio"])
    splits = make_splits(ds.train, ds.y_vector, spec, test=ds.test)
    weight = io.train_weight(ds, spec)
    groups = ds.train[spec.date_col].to_numpy()
    members = []
    for m in range(n):
        mspec = replace(spec, seed=spec.seed + 1000 * (m + 1))
        r = run_cv(mspec, X, ds.y_vector, X_test, splits, "lgbm_shallow", BAG, weight, groups,
                   fit_weight=weight, verbose=False)
        print(f"member {m}: {r.score:.5f} ± {r.score_std:.5f}")
        members.append(r)
    oof = np.mean([r.oof for r in members], axis=0)
    test = np.mean([r.test_pred for r in members], axis=0)
    folds = [score_predictions(spec, ds.y_vector, oof, va, weight, groups) for _, va in splits]
    for i, s in enumerate(folds):
        print(f"  fold {i}: {spec.metric} = {s:.5f}")
    res = replace(members[0], model=f"lgbm_shallow_bag{n}", params={**BAG, "members": n},
                  score=float(np.mean(folds)), score_std=float(np.std(folds)),
                  fold_scores=tuple(folds), oof=oof, test_pred=test, elapsed_s=time.time() - t0)
    print(f"committee x{n}: {res.score:.5f} ± {res.score_std:.5f} ({res.elapsed_s:.0f}s)")
    run_id = tracking.save_run(res, spec, f"committee{n}-beta-idio-lgs-fw",
                               ds.train[spec.id_col].to_numpy(),
                               ds.test[spec.id_col].to_numpy(), notes="issue #37")
    print("saved", run_id)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
