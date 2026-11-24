"""Issue #25: score blends of saved runs per fold on the contract splits (mean ± std).

Usage: uv run python experiments/efe-claude-5/blend_eval.py RUN_ID [RUN_ID ...]
Prints mean and rank-mean blends; never writes a submission.
"""

from __future__ import annotations

import sys

import numpy as np

from qrtc import config, ensemble, io, tracking
from qrtc.cv import make_splits
from qrtc.train import score_predictions


def fold_scores(spec, ds, pred, splits, weight, groups) -> list[float]:
    return [score_predictions(spec, ds.y_vector, pred, va, weight, groups) for _, va in splits]


def main(run_ids: list[str]) -> None:
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    splits = make_splits(ds.train, ds.y_vector, spec, spec.cv)
    weight = io.train_weight(ds, spec)
    groups = ds.train[spec.date_col].to_numpy()
    ids = ds.train[spec.id_col].to_numpy()
    oofs = {r: np.asarray(tracking.load_preds(r, ids)["oof"], dtype=float) for r in run_ids}
    print(ensemble.correlation_table(oofs).round(3).to_string())
    blends = {
        "mean": np.mean(list(oofs.values()), axis=0),
        "rank": np.mean([ensemble.rank_normalise(p) for p in oofs.values()], axis=0),
    }
    for name, pred in {**oofs, **blends}.items():
        s = fold_scores(spec, ds, pred, splits, weight, groups)
        print(f"{name[:45]:45s} {np.mean(s):.5f} ± {np.std(s):.5f}  {np.round(s, 5).tolist()}")


if __name__ == "__main__":
    main(sys.argv[1:])
