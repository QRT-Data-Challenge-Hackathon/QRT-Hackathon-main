"""Command line entry point: ``uv run qrtc <command>``.

  synth <kind>               write a rehearsal dataset (stock_sign, allocation, electricity, ...)
  eda                        overview, split structure, target, signal scan, leakage flags
  baseline [--model lgbm]    CV a model on raw features, save the run, write a submission
  lb                         team leaderboard (all runs: CV, public LB, agent, branch)
  blend RUN [RUN ...]        blend runs' OOF/test preds (mean | rank | hill), save + submit
  record-lb RUN SCORE        store the public leaderboard score of a submitted run
  check-sub PATH             validate a submission file against the spec
  score-hidden PATH          score a submission against y_test_hidden (rehearsal only)
  upload PATH [--final]      designated uploader only: validate, budget-check, log, upload
  upload-confirm ID          record a browser/kaggle upload's outcome (--score S | --failed R)
  uploads [--reveal]         ledger of uploads + remaining budget (--reveal: rehearsal private)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qrtc import config, eda, ensemble, io, platforms, submit, synthetic, tracking
from qrtc import upload as uploads
from qrtc.cv import detect_strategy, make_splits
from qrtc.features import auto_categoricals, to_category
from qrtc.metrics import get_metric
from qrtc.train import CVResult, run_cv, score_predictions


def _prepared(spec: config.Spec) -> tuple[io.Dataset, pd.DataFrame, pd.DataFrame]:
    ds = io.load_dataset(spec)
    feats = io.feature_columns(ds.train, spec)
    cats = [c for c in auto_categoricals(ds.train[feats]) if c in feats]
    X, X_test = to_category(ds.train[feats], ds.test[feats], cats)
    return ds, X, X_test


def cmd_synth(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else config.SHARED_ROOT / "data" / "synthetic" / args.kind
    synthetic.generate(args.kind, out, seed=args.seed)
    print(f"wrote {out}; use it with: QRTC_CONFIG=configs/rehearsal/{args.kind}.toml")
    return 0


def cmd_eda(args: argparse.Namespace) -> int:
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(eda.overview(ds.train).head(60).to_string())
    print("\nsplit structure:", json.dumps(eda.split_structure(ds.train, ds.test, spec),
                                           indent=1, default=str))
    print("\ntarget:\n", eda.target_summary(ds.y, spec.task).to_string())
    if spec.task != "survival":
        dates = ds.train[spec.date_col].to_numpy() if spec.date_col else None
        print("\nsignal scan:\n", eda.signal_scan(ds.train[io.feature_columns(ds.train, spec)],
                                                  ds.y_vector, dates).to_string())
    print("\nleakage flags:", eda.leakage_checks(ds.train, ds.y_vector if ds.y_vector.ndim == 1
                                                 else ds.y_vector[:, 0], spec) or "none")
    return 0


def _metric_inputs(ds: io.Dataset, spec: config.Spec) -> tuple:
    groups = ds.train[spec.date_col].to_numpy() if spec.date_col else None
    return io.train_weight(ds, spec), groups


def _save_and_submit(result: CVResult, spec: config.Spec, ds: io.Dataset, slug: str,
                     notes: str) -> str:
    run_id = tracking.save_run(result, spec, slug, ds.train[spec.id_col].to_numpy(),
                               ds.test[spec.id_col].to_numpy(), notes=notes)
    sub = submit.build_submission(spec, ds.test[spec.id_col].to_numpy(), result.test_pred,
                                  ds.classes)
    path = submit.write_submission(sub, spec, ds.test[spec.id_col].to_numpy(), run_id)
    print(f"run {run_id}\nsubmission {path}")
    return run_id


def cmd_baseline(args: argparse.Namespace) -> int:
    spec = config.load_spec()
    ds, X, X_test = _prepared(spec)
    strategy = args.cv or (detect_strategy(ds.train, ds.test, spec) if spec.cv == "auto"
                           else spec.cv)
    splits = make_splits(ds.train, ds.y_vector, spec, strategy)
    weight, groups = _metric_inputs(ds, spec)
    params = json.loads(args.params) if args.params else None
    print(f"cv={strategy} folds={len(splits)} model={args.model} features={X.shape[1]}")
    result = run_cv(spec, X, ds.y_vector, X_test, splits, args.model, params, weight, groups)
    _save_and_submit(result, spec, ds, f"baseline-{args.model}", f"cv={strategy}")
    return 0


def cmd_lb(args: argparse.Namespace) -> int:
    table = tracking.leaderboard(competition=None if args.all else config.load_spec().name)
    print(table.head(args.n).to_string() if not table.empty else "no runs yet")
    return 0


def _blend_oof(spec: config.Spec, preds: list[dict], ds: io.Dataset, method: str
               ) -> tuple[np.ndarray, np.ndarray, list[float]]:
    oofs, tests = [p["oof"] for p in preds], [p["test"] for p in preds]
    weight, groups = _metric_inputs(ds, spec)
    rows = ensemble.common_rows(oofs)
    y = ds.y_vector

    def scorer(pred: np.ndarray) -> float:
        return score_predictions(spec, y, pred, rows, weight, groups)

    rank = method == "rank" or (method == "hill" and spec.metric in ("spearman", "mean_spearman",
                                                                    "auc", "cindex"))
    w = (ensemble.hill_climb(oofs, scorer, get_metric(spec.metric).greater_is_better, rank=rank)
         if method == "hill" else np.ones(len(oofs)))
    blend_oof = ensemble.weighted_blend(oofs, w, rank)
    return blend_oof, ensemble.weighted_blend(tests, w, rank), list(map(float, w))


def cmd_blend(args: argparse.Namespace) -> int:
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    ids = ds.train[spec.id_col].to_numpy(), ds.test[spec.id_col].to_numpy()
    preds = [tracking.load_preds(r, *ids) for r in args.runs]
    blend_oof, blend_test, weights = _blend_oof(spec, preds, ds, args.method)
    weight, groups = _metric_inputs(ds, spec)
    score = score_predictions(spec, ds.y_vector, blend_oof, None, weight, groups)
    print(f"blend {args.method} weights={weights} OOF {spec.metric} = {score:.5f}")
    result = CVResult(model=f"blend-{args.method}", params={"runs": args.runs, "w": weights},
                      features=(), metric=spec.metric, score=score, score_std=0.0,
                      fold_scores=(score,), oof=blend_oof, test_pred=blend_test, elapsed_s=0.0)
    _save_and_submit(result, spec, ds, f"blend-{args.method}", ",".join(args.runs))
    return 0


def cmd_record_lb(args: argparse.Namespace) -> int:
    print(tracking.record_lb(args.run_id, args.score, args.note))
    return 0


def cmd_check_sub(args: argparse.Namespace) -> int:
    spec = config.load_spec()
    test_ids = io.read_features(spec, spec.x_test)[spec.id_col].to_numpy()
    sub = pd.read_csv(config.resolve_shared(args.path))
    problems = submit.validate_submission(sub, spec, test_ids, submit.load_sample(spec))
    print("OK" if not problems else "INVALID: " + "; ".join(problems))
    return 1 if problems else 0


def cmd_score_hidden(args: argparse.Namespace) -> int:
    spec = config.load_spec()
    hidden = spec.path(spec.hidden_y_test)
    if not hidden or not hidden.exists():
        print("no hidden_y_test configured (only rehearsal datasets have one)")
        return 1
    score = platforms.hidden_score(spec, pd.read_csv(config.resolve_shared(args.path)))
    print(f"hidden {spec.metric} = {score:.5f}")
    return 0


def _upload_line(rec: dict) -> str:
    public = rec.get("public")
    score = f"public={public:.5f}" if isinstance(public, float) else "public=?"
    final = " FINAL" if rec.get("final") else ""
    return (f"{rec['upload_id']} {rec['status']}{final} {score} backend={rec['backend']} "
            f"run={rec.get('run_id') or '-'} file={Path(rec['path']).name}")


def cmd_upload(args: argparse.Namespace) -> int:
    rec = uploads.upload_file(args.path, args.message, args.final, args.run, args.dry_run,
                              args.allow_duplicate)
    print(_upload_line(rec))
    if rec.get("detail"):
        print(rec["detail"])
    return 0


def cmd_upload_confirm(args: argparse.Namespace) -> int:
    print(_upload_line(uploads.confirm(args.upload_id, args.score, args.failed)))
    return 0


def cmd_uploads(args: argparse.Namespace) -> int:
    spec, cfg = config.load_spec(), platforms.load_upload_config()
    table = uploads.upload_table(spec.name, reveal=args.reveal)
    print(table.to_string() if not table.empty else "no uploads yet")
    state = uploads.budget(uploads.read_ledger(), cfg, spec.name)
    left = "unlimited" if state.left is None else state.left
    print(f"\nbackend={cfg.backend} uploaders={list(cfg.uploaders)} used={state.used} "
          f"(today {state.used_today}) left={left} kept_for_finals={cfg.reserve_final}")
    return 0


def _add(sub, name: str, fn, *arguments: tuple[tuple, dict]) -> None:
    parser = sub.add_parser(name)
    for args, kwargs in arguments:
        parser.add_argument(*args, **kwargs)
    parser.set_defaults(fn=fn)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qrtc", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add(sub, "synth", cmd_synth, (("kind",), {"choices": sorted(synthetic.GENERATORS)}),
         (("--out",), {}), (("--seed",), {"type": int, "default": 0}))
    _add(sub, "eda", cmd_eda)
    _add(sub, "baseline", cmd_baseline, (("--model",), {"default": "lgbm"}), (("--cv",), {}),
         (("--params",), {"help": "JSON dict of model params"}))
    _add(sub, "lb", cmd_lb, (("-n",), {"type": int, "default": 30}),
         (("--all",), {"action": "store_true", "help": "all competitions"}))
    _add(sub, "blend", cmd_blend, (("runs",), {"nargs": "+"}),
         (("--method",), {"choices": ["mean", "rank", "hill"], "default": "hill"}))
    _add(sub, "record-lb", cmd_record_lb, (("run_id",), {}), (("score",), {"type": float}),
         (("--note",), {"default": ""}))
    _add(sub, "check-sub", cmd_check_sub, (("path",), {}))
    _add(sub, "score-hidden", cmd_score_hidden, (("path",), {}))
    _add(sub, "upload", cmd_upload, (("path",), {}), (("-m", "--message"), {"default": ""}),
         (("--final",), {"action": "store_true", "help": "may use slots kept for finals"}),
         (("--run",), {"help": "run id (default: from the file name)"}),
         (("--dry-run",), {"action": "store_true", "help": "check everything, upload nothing"}),
         (("--allow-duplicate",), {"action": "store_true"}))
    _add(sub, "upload-confirm", cmd_upload_confirm, (("upload_id",), {}),
         (("--score",), {"type": float, "help": "public score shown by the platform"}),
         (("--failed",), {"metavar": "REASON", "help": "upload did not go through"}))
    _add(sub, "uploads", cmd_uploads,
         (("--reveal",), {"action": "store_true", "help": "show rehearsal private scores"}))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (config.SpecError, io.DataError, submit.SubmissionError, FileNotFoundError,
            ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
