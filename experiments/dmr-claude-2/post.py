"""Post-process a saved run within DATE_ID x COUNTRY groups, re-score on the same folds, save.

The target sums to zero inside every DATE_ID x COUNTRY group, so a prediction component
shared by a whole group carries no information:
  demean  logit(p) minus its group mean   (keeps the model's spread)
  median  above the group median -> +1    (exactly half up, half down per group)

  uv run python experiments/dmr-claude-2/post.py RUN_ID --mode demean
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from qrtc import config, io, submit, tracking
from qrtc.cv import make_splits
from qrtc.postprocess import rank_within
from qrtc.train import CVResult, score_predictions


def group_keys(df: pd.DataFrame) -> np.ndarray:
    return (df["DATE_ID"].astype(str) + "_" + df["COUNTRY"].astype(str)).to_numpy()


def transform(p: np.ndarray, groups: np.ndarray, mode: str) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    if mode == "median":
        return np.where(rank_within(p, groups) > 0.5, 0.75, 0.25)
    z = np.log(p / (1 - p))
    z = z - pd.Series(z).groupby(groups).transform("mean").to_numpy()
    return 1.0 / (1.0 + np.exp(-z))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--mode", choices=["demean", "median"], default="demean")
    a = ap.parse_args()
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    ids, test_ids = ds.train[spec.id_col].to_numpy(), ds.test[spec.id_col].to_numpy()
    preds = tracking.load_preds(a.run_id, ids, test_ids)
    oof = transform(preds["oof"], group_keys(ds.train), a.mode)
    test = transform(preds["test"], group_keys(ds.test), a.mode)
    splits = make_splits(ds.train, ds.y_vector, spec)
    weight = io.train_weight(ds, spec)
    folds = [score_predictions(spec, ds.y_vector, oof, va, weight) for _, va in splits]
    result = CVResult(model=f"post-{a.mode}", params={"run": a.run_id}, features=(),
                      metric=spec.metric, score=float(np.mean(folds)),
                      score_std=float(np.std(folds)), fold_scores=tuple(folds), oof=oof,
                      test_pred=test, elapsed_s=0.0)
    print(result.summary())
    run_id = tracking.save_run(result, spec, f"{a.mode}", ids, test_ids,
                               notes=f"{a.mode} within DATE_ID x COUNTRY of {a.run_id}")
    path = submit.write_submission(submit.build_submission(spec, test_ids, test, ds.classes),
                                   spec, test_ids, run_id)
    print(f"run {run_id}\nsubmission {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
