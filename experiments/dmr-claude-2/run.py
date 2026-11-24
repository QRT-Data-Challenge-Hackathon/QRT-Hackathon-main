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

from qrtc import config, io, submit, tracking
from qrtc.cv import detect_strategy, make_splits
from qrtc.features import auto_categoricals, to_category
from qrtc.train import run_cv


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lgbm_shallow")
    p.add_argument("--params", help="JSON dict of model params")
    p.add_argument("--slug", default="")
    p.add_argument("--cv", help="override the spec's CV strategy (default: spec / auto)")
    p.add_argument("--no-fit-weight", action="store_true", help="train unweighted")
    p.add_argument("--drop-prefix", nargs="*", default=[], help="drop features with prefix")
    p.add_argument("--only-prefix", nargs="*", default=[], help="keep only these prefixes")
    return p.parse_args()


def select(feats: list[str], drop: list[str], only: list[str]) -> list[str]:
    keep = [f for f in feats if not any(f.startswith(d) for d in drop)]
    return [f for f in keep if any(f.startswith(o) for o in only)] if only else keep


def main() -> int:
    args = parse()
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    feats = select(io.feature_columns(ds.train, spec), args.drop_prefix, args.only_prefix)
    cats = [c for c in auto_categoricals(ds.train[feats]) if c in feats]
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
    result = run_cv(spec, X, ds.y_vector, X_test, splits, args.model, params, weight, groups,
                    fit_weight=fit_weight)
    slug = "-".join(filter(None, [args.model, args.slug, "w" if fit_weight is not None else ""]))
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
