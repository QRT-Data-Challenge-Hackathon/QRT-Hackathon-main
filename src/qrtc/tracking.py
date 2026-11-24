"""Run registry: every experiment's predictions and metadata, keyed by a unique ``run_id``.

``run_id`` = timestamp + slug, so parallel experiments never write the same file:
  * ``<shared>/runs/<run_id>/{preds.npz,meta.json}``  local, gitignored (OOF + test preds)
  * ``results/registry/<run_id>.json``   the same metadata, small enough to commit
  * ``results/lb/<run_id>.json`` + ``<shared>/runs/_lb/``  public-leaderboard score
``leaderboard()`` unions all of these into one table sorted by CV score.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qrtc import config
from qrtc.config import Spec
from qrtc.metrics import get_metric
from qrtc.train import CVResult


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5,
                             cwd=config.REPO_ROOT, check=False)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def make_run_id(slug: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "-", slug).strip("-").lower()[:40] or "run"
    return f"{datetime.now():%m%d-%H%M%S}_{clean}"


def save_run(result: CVResult, spec: Spec, slug: str, train_ids: np.ndarray,
             test_ids: np.ndarray, notes: str = "", extra: dict[str, Any] | None = None) -> str:
    """Persist predictions + metadata; returns the run_id."""
    run_id = make_run_id(slug)
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(run_dir / "preds.npz", oof=result.oof, test=result.test_pred,
                        train_ids=np.asarray(train_ids), test_ids=np.asarray(test_ids))
    meta = {
        "run_id": run_id, "slug": slug, "competition": spec.name,
        "metric": result.metric, "cv": result.score, "cv_std": result.score_std,
        "fold_scores": list(result.fold_scores), "model": result.model,
        "params": result.params, "n_features": len(result.features),
        "features": list(result.features)[:300], "elapsed_s": round(result.elapsed_s, 1),
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "created": datetime.now().isoformat(timespec="seconds"), "notes": notes,
        **(extra or {}),
    }
    text = json.dumps(meta, indent=2, default=str)
    (run_dir / "meta.json").write_text(text)
    config.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    (config.REGISTRY_DIR / f"{run_id}.json").write_text(text)
    return run_id


def load_preds(run_id: str, train_ids: np.ndarray | None = None,
               test_ids: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """OOF/test predictions and ids of a run (must exist locally under runs/).

    Pass the current ``train_ids``/``test_ids`` to fail loudly when the run was made on other
    data (another competition, regenerated data, a different row order)."""
    path = config.RUNS_DIR / run_id / "preds.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: predictions only exist on the machine "
                                "that produced the run; rerun it or copy runs/<run_id>/")
    with np.load(path, allow_pickle=True) as data:
        preds = {k: data[k] for k in data.files}
    for key, expected in (("train_ids", train_ids), ("test_ids", test_ids)):
        if expected is not None and not np.array_equal(
                preds[key].astype(str), np.asarray(expected).astype(str)):
            raise ValueError(f"run {run_id}: {key} do not match the current data "
                             "(different dataset or row order); do not blend it")
    return preds


def record_lb(run_id: str, public_score: float, note: str = "") -> Path:
    """Store the public leaderboard score of a submitted run."""
    payload = {"run_id": run_id, "public_lb": public_score, "note": note,
               "recorded": datetime.now().isoformat(timespec="seconds")}
    text = json.dumps(payload, indent=2)
    config.SHARED_LB_DIR.mkdir(parents=True, exist_ok=True)
    (config.SHARED_LB_DIR / f"{run_id}.json").write_text(text)
    config.LB_DIR.mkdir(parents=True, exist_ok=True)
    path = config.LB_DIR / f"{run_id}.json"
    path.write_text(text)
    return path


def _read_jsons(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        try:
            rows.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping unreadable {path}: {exc}")
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates("run_id", keep="last") if "run_id" in frame else frame


def all_runs() -> pd.DataFrame:
    """Metadata of every run known here: committed registry + local runs/."""
    paths = sorted(config.REGISTRY_DIR.glob("*.json")) + sorted(config.RUNS_DIR.glob("*/meta.json"))
    return _read_jsons(paths)


def leaderboard(metric: str | None = None, competition: str | None = None) -> pd.DataFrame:
    """All registered runs (optionally filtered), best CV first, with public LB if known."""
    runs = all_runs()
    if runs.empty:
        return runs
    if competition:
        runs = runs[runs["competition"] == competition]
    if metric:
        runs = runs[runs["metric"] == metric]
    lb = _read_jsons(sorted(config.LB_DIR.glob("*.json")) + sorted(config.SHARED_LB_DIR.glob("*.json")))
    if not lb.empty:
        runs = runs.merge(lb[["run_id", "public_lb"]], on="run_id", how="left")
    if runs.empty:
        return runs
    ascending = not get_metric(runs["metric"].iloc[0]).greater_is_better
    runs = runs.assign(local=[(config.RUNS_DIR / r / "preds.npz").exists() for r in runs.run_id])
    cols = [c for c in ["run_id", "cv", "cv_std", "public_lb", "model", "n_features", "local",
                        "notes"] if c in runs.columns]
    return runs.sort_values("cv", ascending=ascending)[cols].reset_index(drop=True)
