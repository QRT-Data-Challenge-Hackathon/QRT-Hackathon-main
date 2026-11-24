"""Residual-path runner: feature families -> standard CV -> saved run + submission.

  uv run python experiments/residual_gbm/run.py --families ret --model lgbm_shallow
  uv run python experiments/residual_gbm/run.py --families ret,rel --baseline <run_id>

  --no-save     CV only (no run, no submission): for quick checks before the spec is merged
  --fit-weight  train with |target| sample weights (aligns the loss with weighted accuracy)
  --export DIR  also write ID + features parquet (train/test) under the main checkout's DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import path_features as pf  # noqa: E402
import sectors as sec  # noqa: E402

from qrtc import config, io, submit, tracking  # noqa: E402
from qrtc.cv import make_splits  # noqa: E402
from qrtc.features import auto_categoricals, to_category  # noqa: E402
from qrtc.submit import SubmissionError  # noqa: E402
from qrtc.train import CVResult, run_cv, score_predictions  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-sectional features of the price path")
    p.add_argument("--families", default="ret", help=f"comma list of {pf.ALL_FAMILIES}")
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
    p.add_argument("--sectors", type=int, default=0,
                   help="K co-movement sectors (KMeans on residual paths) + vs-sector features")
    p.add_argument("--sector-ranks", action="store_true",
                   help="with --sectors: ranks within day x sector and of sectors in a country")
    p.add_argument("--seeds", type=int, default=1, help=f"average over the first N of {SEEDS}")
    p.add_argument("--target", choices=TARGETS, default="class",
                   help="class: the 0/1 sign; reg / reg_clip: regress Close return (clip q99)")
    p.add_argument("--stock-cat", action="store_true", help="STOCK_ID as a categorical")
    p.add_argument("--drop", default="", help="comma list of feature columns to drop")
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
    cols = [id_col, "DATE_ID", "COUNTRY", pf.PRICE_COL]
    return pd.concat([pd.read_parquet(base / n, columns=cols)
                      for n in ("x_train.parquet", "x_test.parquet")], ignore_index=True)


def base_columns(frame: pd.DataFrame, spec: config.Spec, base: str, country: str) -> list[str]:
    """``country``: only the country; ``spec``: every spec feature column (prepped features)."""
    if base == "country":
        return [country]
    cols = io.feature_columns(frame, spec)
    return [c for c in cols if c != pf.PRICE_COL and not isinstance(frame[c].iloc[0],
                                                                     (list, np.ndarray))]


def frames_with_prices(ds: io.Dataset, spec: config.Spec, args: argparse.Namespace
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train / test frames with ``Price TS`` (re-attached from the raw files if needed)."""
    train, test = ds.train, ds.test
    if pf.PRICE_COL not in train.columns or pf.PRICE_COL not in test.columns:
        raw = load_raw(args.raw_dir, spec.id_col)
        train, test = (with_raw_columns(f, raw, spec.id_col) for f in (train, test))
    return train, test


def add_sectors(train: pd.DataFrame, test: pd.DataFrame, X: pd.DataFrame, X_test: pd.DataFrame,
                spec: config.Spec, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster stocks into ``args.sectors`` co-movement sectors (features of all rows, no
    target) and add residual-vs-sector features."""
    keys = [spec.id_col, spec.date_col, args.country, spec.group_col]
    r_train = sec.residual_paths(train, spec.date_col, args.country)
    r_test = sec.residual_paths(test, spec.date_col, args.country)
    rows = pd.concat([train[keys], test[keys]], ignore_index=True)
    first = ~rows[spec.id_col].duplicated().to_numpy()
    sector = sec.stock_clusters(rows[first], np.vstack([r_train, r_test])[first],
                                spec.group_col, spec.date_col, args.sectors, spec.seed)
    print(f"sectors: sizes {sorted(sector.value_counts().tolist(), reverse=True)}")
    new = [sec.sector_features(f[keys], r, sector, spec.group_col, spec.date_col)
           for f, r in ((train, r_train), (test, r_test))]
    if args.sector_ranks:
        new = [pd.concat([n, sec.sector_ranks(f[keys], n, spec.date_col, args.country)], axis=1)
               for f, n in ((train, new[0]), (test, new[1]))]
    X, X_test = (pd.concat([a, b], axis=1) for a, b in zip((X, X_test), new, strict=True))
    return to_category(X, X_test, ["sector"])


def features_for(ds: io.Dataset, spec: config.Spec, args: argparse.Namespace,
                 families: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Base columns + the requested families, built for train and test separately."""
    train, test = frames_with_prices(ds, spec, args)
    base = base_columns(ds.train, spec, args.base, args.country)
    new_train = pf.build_features(train, spec.date_col, args.country, families)
    new_test = pf.build_features(test, spec.date_col, args.country, families)
    X = pd.concat([train[base], new_train.drop(columns=base, errors="ignore")], axis=1)
    X_test = pd.concat([test[base], new_test.drop(columns=base, errors="ignore")], axis=1)
    cats = [c for c in auto_categoricals(X) if c in X.columns] + [args.country]
    X, X_test = to_category(X, X_test, sorted(set(cats) & set(X.columns)))
    if args.sectors:
        X, X_test = add_sectors(train, test, X, X_test, spec, args)
    return X, X_test


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
        ratio = frame[cols] / (prof[cols] + pf.EPS)
        return pd.concat([frame, prof.add_suffix("_stock"), ratio.add_suffix("_vs_stock")], axis=1)

    return attach(X), attach(X_test)


SEEDS = (0, 7, 13, 21, 42)


TARGETS = ("class", "reg", "reg_clip")  # classify the sign | regress the real return (clipped)


def fit_target(spec: config.Spec, ds: io.Dataset, kind: str) -> tuple[config.Spec, np.ndarray]:
    """Spec + target the models are fitted on: the 0/1 sign, or the real ``Close return``
    (regression; MSE already weights big moves like the metric does)."""
    if kind == "class":
        return spec, ds.y_vector
    raw = ds.y_raw.iloc[:, 0].to_numpy(dtype=float)
    if kind == "reg_clip":
        lim = float(np.quantile(np.abs(raw), 0.99))
        raw = np.clip(raw, -lim, lim)
    return replace(spec, task="regression", metric="rmse", binarize_target=False,
                   weight_col=None), raw


def _seed_params(args: argparse.Namespace, params: dict | None) -> list[dict | None]:
    """One param dict per bagging seed (LightGBM models only; others are deterministic)."""
    if args.seeds <= 1 or not args.model.startswith("lgbm"):
        return [params]
    return [{**(params or {}), "bagging_seed": s, "feature_fraction_seed": s}
            for s in SEEDS[:args.seeds]]


def _to_prob(pred: np.ndarray, scale: float) -> np.ndarray:
    """Regression output -> (0, 1), monotone, 0.5 at a zero return (so the sign is kept)."""
    return 1.0 / (1.0 + np.exp(-pred / scale))


def cv_target(spec: config.Spec, X: pd.DataFrame, X_test: pd.DataFrame, ds: io.Dataset,
              splits: list, args: argparse.Namespace, params: dict | None) -> CVResult:
    """Fit on ``args.target`` over the seeds; average OOF / test; score weighted accuracy
    per fold on the average (seed noise of one lgbm_shallow run is ~0.0015 here)."""
    spec_fit, y_fit = fit_target(spec, ds, args.target)
    weight, groups = io.train_weight(ds, spec), ds.train[spec.date_col].to_numpy()
    fit_w = weight if args.fit_weight else None
    metric_w, metric_g = (weight, groups) if spec_fit is spec else (None, None)
    results = []
    for p in _seed_params(args, params):
        results.append(run_cv(spec_fit, X, y_fit, X_test, splits, args.model, p, metric_w,
                              metric_g, fit_weight=fit_w, verbose=False))
        print(f"  fit: {results[-1].summary()}")
    oof = np.mean([r.oof for r in results], axis=0)
    test = np.mean([r.test_pred for r in results], axis=0)
    if spec_fit.task == "regression":
        scale = float(np.nanstd(oof)) or 1.0
        oof, test = _to_prob(oof, scale), _to_prob(test, scale)
    folds = tuple(score_predictions(spec, ds.y_vector, oof, va, weight, groups)
                  for _, va in splits)
    imp = pd.concat([r.importance for r in results], axis=1).mean(axis=1)
    return replace(results[0], metric=spec.metric, oof=oof, test_pred=test,
                   params={**(params or {}), "target": args.target, "n_fits": len(results)},
                   score=float(np.mean(folds)), score_std=float(np.std(folds)),
                   fold_scores=folds, importance=imp.sort_values(ascending=False),
                   elapsed_s=sum(r.elapsed_s for r in results))


def make_slug(args: argparse.Namespace, tag: str) -> str:
    """Short slug (run ids keep 40 chars): a-<base>-<families>-<model>[-fw][-sp]."""
    model = {"lgbm_shallow": "lgs", "logreg": "lr", "catboost_shallow": "cbs"}.get(args.model,
                                                                                  args.model)
    return (f"a-{args.base[0]}-{tag}-{model}{'-fw' if args.fit_weight else ''}"
            f"{'-sp' if args.stock_profile else ''}{f'-k{args.sectors}' if args.sectors else ''}"
            f"{'r' if args.sector_ranks else ''}"
            f"{f'-x{args.seeds}' if args.seeds > 1 else ''}"
            f"{'' if args.target == 'class' else '-' + args.target.replace('_', '')}"
            f"{'-sc' if args.stock_cat else ''}")


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
    """ID + features parquet for other experiments (data/ is gitignored)."""
    out = config.SHARED_ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, frame, ids in (("train", X, ds.train[spec.id_col]), ("test", X_test,
                                                                    ds.test[spec.id_col])):
        path = out / f"residual_{tag}_{name}.parquet"
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
    if args.stock_cat:
        X, X_test = to_category(X, X_test, [spec.group_col])
    if args.drop:
        cols = [c.strip() for c in args.drop.split(",") if c.strip()]
        X, X_test = X.drop(columns=cols), X_test.drop(columns=cols)
    splits = make_splits(ds.train, ds.y_vector, spec, test=ds.test)
    params = json.loads(args.params) if args.params else None
    print(f"families={families} base={args.base} model={args.model} feats={X.shape[1]} "
          f"folds={len(splits)} fit_weight={args.fit_weight} target={args.target} "
          f"spec={spec.name} cv={spec.cv}")
    result = cv_target(spec, X, X_test, ds, splits, args, params)
    print(result.summary())
    print(compare(args.baseline, result))
    print("top features:", ", ".join(result.importance.head(10).index.astype(str)))
    tag = "-".join(families) or "none"
    if args.export:
        export(args.export, tag, ds, spec, X, X_test)
    if not args.no_save:
        slug = args.slug or make_slug(args, tag)
        notes = json.dumps({"families": families, "base": args.base, "model": args.model,
                            "fit_weight": args.fit_weight, "stock_profile": args.stock_profile,
                            "sectors": args.sectors, "target": args.target,
                            "stock_cat": args.stock_cat, "seeds": args.seeds,
                            "baseline": args.baseline})
        save(result, spec, ds, slug, notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
