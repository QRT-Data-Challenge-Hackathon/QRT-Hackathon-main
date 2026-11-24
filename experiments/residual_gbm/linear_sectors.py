"""Ridge (and ExtraTrees) regressed on the real Close return, on the beta + idio + sectors
features of the residual-path LightGBM.

Features come from `run.py ... --export data/derived/residual_export` (the same builder,
unchanged). Same recipe as the intraday ridge (ridge-v1-reg): fit the real return (scaling
inside the sklearn pipeline, i.e. inside the folds), map pred to sigmoid(pred / std), score
the sign with the spec metric on the official date_blocks folds.

Usage: uv run python experiments/residual_gbm/linear_sectors.py TAG [--alpha 10] [--fit-weight]
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from qrtc import config, cv, io, submit, tracking
from qrtc.train import run_cv, score_predictions

DERIVED = config.SHARED_ROOT / "data" / "derived" / "residual_export"


def load(tag: str, ds: io.Dataset, spec: config.Spec) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = []
    for name, ids in (("train", ds.train[spec.id_col]), ("test", ds.test[spec.id_col])):
        f = pd.read_parquet(DERIVED / f"residual_{tag}_{name}.parquet").set_index(spec.id_col)
        out.append(f.loc[ids.to_numpy()].reset_index(drop=True))
    return out[0], out[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tag")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--fit-weight", action="store_true")
    a = p.parse_args()
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    X, X_test = load(a.tag, ds, spec)
    splits = cv.make_splits(ds.train, ds.y_vector, spec)
    weight = io.train_weight(ds, spec)
    raw = ds.y_raw.iloc[:, 0].to_numpy(dtype=float)
    rspec = replace(spec, task="regression", metric="pearson")
    res = run_cv(rspec, X, raw, X_test, splits, "ridge", {"alpha": a.alpha}, None, None,
                 fit_weight=weight if a.fit_weight else None, verbose=False)
    scale = float(np.nanstd(res.oof)) or 1.0
    oof, test = (1.0 / (1.0 + np.exp(-z / scale)) for z in (res.oof, res.test_pred))
    folds = [score_predictions(spec, ds.y_vector, oof, va, weight) for _, va in splits]
    res = replace(res, metric=spec.metric, score=float(np.mean(folds)),
                  score_std=float(np.std(folds)), fold_scores=tuple(folds), oof=oof,
                  test_pred=test)
    slug = f"ridge-sectors-a{a.alpha:g}{'-fw' if a.fit_weight else ''}"
    tr_ids, te_ids = ds.train[spec.id_col].to_numpy(), ds.test[spec.id_col].to_numpy()
    run_id = tracking.save_run(res, spec, slug, tr_ids, te_ids, notes=f"residual features tag={a.tag}")
    path = submit.write_submission(submit.build_submission(spec, te_ids, test), spec, te_ids,
                                   run_id)
    print(f"ridge: {spec.metric} = {res.score:.5f} +- {res.score_std:.5f} {np.round(folds, 5)}")
    print(f"run {run_id}\nsubmission {path}")


if __name__ == "__main__":
    main()
