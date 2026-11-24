"""Lane A runner (dmr-claude-3): feature families -> standard CV -> saved run + submission.

  export AGENT_NAME=dmr-claude-3
  uv run python experiments/dmr-claude-3/run_lane_a.py --families ret --model lgbm_shallow
  uv run python experiments/dmr-claude-3/run_lane_a.py --families ret,rel --baseline <run_id>

  --no-save     CV only (no run, no submission): for quick checks before the spec is merged
  --fit-weight  train with |target| sample weights (aligns the loss with weighted accuracy)
  --export DIR  also write ID + features parquet (train/test) under the main checkout's DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lane_a_features as laf  # noqa: E402

from qrtc import config, io, submit, tracking  # noqa: E402
from qrtc.cv import make_splits  # noqa: E402
from qrtc.features import auto_categoricals, to_category  # noqa: E402
from qrtc.submit import SubmissionError  # noqa: E402
from qrtc.train import CVResult, run_cv  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lane A: cross-sectional features of the price path")
    p.add_argument("--families", default="ret", help=f"comma list of {laf.ALL_FAMILIES}")
    p.add_argument("--model", default="lgbm_shallow")
    p.add_argument("--params", default="", help="JSON model params")
    p.add_argument("--country", default="COUNTRY")
    p.add_argument("--base", choices=("country", "spec"), default="country",
                   help="country only, or every spec feature column (stack on prepped tables)")
    p.add_argument("--raw-dir", default="data/raw/qrt-x-eth-hackathon-24-11-2026",
                   help="raw Kaggle files (for Price TS) under the main checkout")
    p.add_argument("--baseline", default="", help="run_id to compare against (paired by fold)")
    p.add_argument("--fit-weight", action="store_true")
    p.add_argument("--stock-profile", action="store_true",
                   help="add per-stock feature means over all rows and today-vs-own-mean ratios")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--export", default="")
    p.add_argument("--slug", default="")
    return p.parse_args(argv)


def with_raw_columns(frame: pd.DataFrame, raw: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """``frame`` plus the raw columns it lacks (``Price TS``, keys), matched by id, order kept."""
    missing = [c for c in raw.columns if c not in frame.columns]
    if not missing:
        return frame
    merged = frame[[id_col]].merge(raw[[id_col, *missing]], on=id_col, how="left")
    if merged[missing[0]].isna().any():
        raise ValueError(f"{int(merged[missing[0]].isna().sum())} ids have no raw row")
    return pd.concat([frame, merged[missing].set_index(frame.index)], axis=1)


def load_raw(raw_dir: str, id_col: str) -> pd.DataFrame:
    """Raw train + test rows (one table) for re-attaching ``Price TS`` to flat spec tables."""
    base = config.SHARED_ROOT / raw_dir
    cols = [id_col, "DATE_ID", "COUNTRY", laf.PRICE_COL]
    return pd.concat([pd.read_parquet(base / n, columns=cols)
                      for n in ("x_train.parquet", "x_test.parquet")], ignore_index=True)


def base_columns(frame: pd.DataFrame, spec: config.Spec, base: str, country: str) -> list[str]:
    """``country``: only the country; ``spec``: every spec feature column (prepped features)."""
    if base == "country":
        return [country]
    cols = io.feature_columns(frame, spec)
    return [c for c in cols if c != laf.PRICE_COL and not isinstance(frame[c].iloc[0],
                                                                     (list, np.ndarray))]


def features_for(ds: io.Dataset, spec: config.Spec, args: argparse.Namespace,
                 families: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Base columns + the requested families, built for train and test separately."""
    train, test = ds.train, ds.test
    if laf.PRICE_COL not in train.columns or laf.PRICE_COL not in test.columns:
        raw = load_raw(args.raw_dir, spec.id_col)
        train, test = (with_raw_columns(f, raw, spec.id_col) for f in (train, test))
    base = base_columns(ds.train, spec, args.base, args.country)
    new_train = laf.build_features(train, spec.date_col, args.country, families)
    new_test = laf.build_features(test, spec.date_col, args.country, families)
    X = pd.concat([train[base], new_train.drop(columns=base, errors="ignore")], axis=1)
    X_test = pd.concat([test[base], new_test.drop(columns=base, errors="ignore")], axis=1)
    cats = [c for c in auto_categoricals(X) if c in X.columns] + [args.country]
    return to_category(X, X_test, sorted(set(cats) & set(X.columns)))


PROFILE_COLS = ("vol", "vol_last12", "idio_vol", "beta", "corr_country", "tick_rel",
                "flat_frac", "roll_spread", "med_move")


def stock_profile(X: pd.DataFrame, X_test: pd.DataFrame, ids: pd.Series, ids_test: pd.Series,
                  stock: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-stock means of non-target features over every row (train + test, no targets used),
    and today's value relative to the stock's own mean."""
    cols = [c for c in PROFILE_COLS if c in X.columns]
    if not cols:
        raise ValueError(f"--stock-profile needs one of {PROFILE_COLS} among the features")
    rows = pd.concat([X[[stock, *cols]].assign(_id=ids.to_numpy()),
                      X_test[[stock, *cols]].assign(_id=ids_test.to_numpy())])
    means = rows.drop_duplicates("_id").groupby(stock, observed=True)[cols].mean()

    def attach(frame: pd.DataFrame) -> pd.DataFrame:
        prof = means.reindex(frame[stock].to_numpy()).set_index(frame.index)
        ratio = frame[cols] / (prof[cols] + laf.EPS)
        return pd.concat([frame, prof.add_suffix("_stock"), ratio.add_suffix("_vs_stock")], axis=1)

    return attach(X), attach(X_test)


def make_slug(args: argparse.Namespace, tag: str) -> str:
    """Short slug (run ids keep 40 chars): a-<base>-<families>-<model>[-fw][-sp]."""
    model = {"lgbm_shallow": "lgs", "logreg": "lr", "catboost_shallow": "cbs"}.get(args.model,
                                                                                  args.model)
    return (f"a-{args.base[0]}-{tag}-{model}{'-fw' if args.fit_weight else ''}"
            f"{'-sp' if args.stock_profile else ''}")


def compare(baseline: str, result: CVResult) -> str:
    """Paired fold-by-fold delta vs a saved run (same splits => same folds)."""
    if not baseline:
        return "no baseline given"
    runs = tracking.all_runs()
    row = runs[runs["run_id"] == baseline] if not runs.empty else runs
    if row.empty:
        return f"baseline {baseline} not found"
    base_folds = np.asarray(row["fold_scores"].iloc[0], dtype=float)
    folds = np.asarray(result.fold_scores, dtype=float)
    if len(base_folds) != len(folds):
        return f"baseline {baseline} has {len(base_folds)} folds, this run {len(folds)}"
    delta = folds - base_folds
    return (f"vs {baseline}: {base_folds.mean():.5f} ± {base_folds.std():.5f} -> delta "
            f"{delta.mean():+.5f} ± {delta.std():.5f} (folds better: {(delta > 0).sum()}"
            f"/{len(delta)})")


def export(out_dir: str, tag: str, ds: io.Dataset, spec: config.Spec, X: pd.DataFrame,
           X_test: pd.DataFrame) -> None:
    """ID + features parquet for other agents (data/ is gitignored and shared)."""
    out = config.SHARED_ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, frame, ids in (("train", X, ds.train[spec.id_col]), ("test", X_test,
                                                                    ds.test[spec.id_col])):
        path = out / f"lane_a_{tag}_{name}.parquet"
        frame.assign(**{spec.id_col: ids.to_numpy()}).to_parquet(path)
        print(f"exported {path}")


def save(result: CVResult, spec: config.Spec, ds: io.Dataset, slug: str, notes: str) -> str:
    test_ids = ds.test[spec.id_col].to_numpy()
    run_id = tracking.save_run(result, spec, slug, ds.train[spec.id_col].to_numpy(), test_ids,
                               notes=notes)
    print(f"run {run_id}")
    try:
        sub = submit.build_submission(spec, test_ids, result.test_pred, ds.classes)
        print(f"submission {submit.write_submission(sub, spec, test_ids, run_id)}")
    except SubmissionError as exc:
        print(f"WARNING: run saved but submission not written: {exc}", file=sys.stderr)
    return run_id


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = config.load_spec()
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    ds = io.load_dataset(spec)
    X, X_test = features_for(ds, spec, args, families)
    if args.stock_profile:
        X, X_test = stock_profile(X, X_test, ds.train[spec.id_col], ds.test[spec.id_col],
                                  spec.group_col)
    y = ds.y_vector
    splits = make_splits(ds.train, y, spec, test=ds.test)
    weight = io.train_weight(ds, spec)
    params = json.loads(args.params) if args.params else None
    print(f"families={families} base={args.base} model={args.model} feats={X.shape[1]} "
          f"folds={len(splits)} fit_weight={args.fit_weight} spec={spec.name} cv={spec.cv}")
    result = run_cv(spec, X, y, X_test, splits, args.model, params, weight,
                    ds.train[spec.date_col].to_numpy(),
                    fit_weight=weight if args.fit_weight else None)
    print(compare(args.baseline, result))
    print("top features:", ", ".join(result.importance.head(10).index.astype(str)))
    tag = "-".join(families) or "none"
    if args.export:
        export(args.export, tag, ds, spec, X, X_test)
    if not args.no_save:
        slug = args.slug or make_slug(args, tag)
        notes = json.dumps({"families": families, "base": args.base, "model": args.model,
                            "fit_weight": args.fit_weight, "stock_profile": args.stock_profile,
                            "baseline": args.baseline})
        save(result, spec, ds, slug, notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
