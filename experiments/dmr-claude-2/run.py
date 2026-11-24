"""CV one model on the spec's tables, training with |target| as sample weight.

The metric is accuracy weighted by |move|, so a classifier should also be trained with that
weight (``qrtc baseline`` passes the weight to the metric only). Same splits, metric, run
registry and submission writer as ``qrtc baseline``.

  uv run python experiments/dmr-claude-2/run.py --model lgbm_shallow --params '{"n_jobs": 4}'
  uv run python experiments/dmr-claude-2/run.py --model logreg --drop-prefix hist_ --slug no-hist
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np

from qrtc import config, io, submit, tracking
from qrtc.cv import detect_strategy, make_splits
from qrtc.features import auto_categoricals, to_category
from qrtc.train import CVResult, run_cv, score_predictions


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lgbm_shallow")
    p.add_argument("--params", help="JSON dict of model params")
    p.add_argument("--slug", default="")
    p.add_argument("--cv", help="override the spec's CV strategy (default: spec / auto)")
    p.add_argument("--no-fit-weight", action="store_true", help="train unweighted")
    p.add_argument("--drop-prefix", nargs="*", default=[], help="drop features with prefix")
    p.add_argument("--only-prefix", nargs="*", default=[], help="keep only these prefixes")
    p.add_argument("--cat", nargs="*", default=[], help="extra columns to treat as categorical")
    p.add_argument("--regress", action="store_true", help="fit the real target, use its sign")
    return p.parse_args()


def regress_cv(spec, ds, X, X_test, splits, model, params, weight, fit_weight) -> CVResult:
    """Fit ``model`` on the real target, score its sign with the spec's metric.

    Predictions are mapped to pseudo-probabilities sigmoid(pred / std) so that the 0.5
    threshold of the label metrics equals pred = 0 and runs blend with classifiers."""
    raw = ds.y_raw.iloc[:, 0].to_numpy(dtype=float)
    rspec = replace(spec, task="regression", metric="pearson")
    res = run_cv(rspec, X, raw, X_test, splits, model, params, None, None,
                 fit_weight=fit_weight, verbose=False)
    scale = float(np.nanstd(res.oof)) or 1.0
    oof, test = (1.0 / (1.0 + np.exp(-a / scale)) for a in (res.oof, res.test_pred))
    folds = [score_predictions(spec, ds.y_vector, oof, va, weight) for _, va in splits]
    for i, s in enumerate(folds):
        print(f"  fold {i}: {spec.metric} = {s:.5f}")
    return replace(res, metric=spec.metric, score=float(np.mean(folds)),
                   score_std=float(np.std(folds)), fold_scores=tuple(folds), oof=oof,
                   test_pred=test)


def select(feats: list[str], drop: list[str], only: list[str]) -> list[str]:
    keep = [f for f in feats if not any(f.startswith(d) for d in drop)]
    return [f for f in keep if any(f.startswith(o) for o in only)] if only else keep


def main() -> int:
    args = parse()
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    feats = select(io.feature_columns(ds.train, spec), args.drop_prefix, args.only_prefix)
    cats = [c for c in auto_categoricals(ds.train[feats]) if c in feats]
    cats += [c for c in args.cat if c in feats and c not in cats]
    X, X_test = to_category(ds.train[feats], ds.test[feats], cats)
    strategy = args.cv or (detect_strategy(ds.train, ds.test, spec) if spec.cv == "auto"
                           else spec.cv)
    splits = make_splits(ds.train, ds.y_vector, spec, strategy)
    weight = io.train_weight(ds, spec)
    groups = ds.train[spec.date_col].to_numpy() if spec.date_col else None
    fit_weight = None if args.no_fit_weight else weight
    params = json.loads(args.params) if args.params else None
    print(f"cv={strategy} folds={len(splits)} model={args.model} features={X.shape[1]} "
          f"fit_weight={'|y|' if fit_weight is not None else 'none'} cats={cats}")
    if args.regress:
        result = regress_cv(spec, ds, X, X_test, splits, args.model, params, weight, fit_weight)
        print(result.summary())
    else:
        result = run_cv(spec, X, ds.y_vector, X_test, splits, args.model, params, weight,
                        groups, fit_weight=fit_weight)
    tags = ["reg" if args.regress else "", "w" if fit_weight is not None else ""]
    slug = "-".join(filter(None, [args.model, args.slug, *tags]))
    notes = f"cv={strategy} fit_weight={fit_weight is not None} drop={args.drop_prefix} " \
            f"only={args.only_prefix}"
    ids, test_ids = ds.train[spec.id_col].to_numpy(), ds.test[spec.id_col].to_numpy()
    run_id = tracking.save_run(result, spec, slug, ids, test_ids, notes=notes)
    sub = submit.build_submission(spec, test_ids, result.test_pred, ds.classes)
    path = submit.write_submission(sub, spec, test_ids, run_id)
    print(f"run {run_id}\nsubmission {path}")
    print(result.importance.head(15).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
