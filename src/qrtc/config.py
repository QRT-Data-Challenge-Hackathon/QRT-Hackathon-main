"""Competition spec: one TOML file (``competition.toml``) describes the event's problem.

Every script, notebook and agent reads the spec through :func:`load_spec` so that the problem
definition (files, id/target/date columns, task, metric, CV) lives in exactly one place.
Override the file with the ``QRTC_CONFIG`` environment variable (used for rehearsal configs).
"""

from __future__ import annotations

import os
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _shared_root() -> Path:
    """Main checkout shared by all git worktrees ($QRTC_SHARED_DIR overrides).

    Gitignored state (data/, runs/, submissions/) lives here so that every agent worktree on
    this machine reads the same data and can blend every other agent's predictions."""
    env = os.environ.get("QRTC_SHARED_DIR")
    if env:
        return Path(env).expanduser().resolve()
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--path-format=absolute",
             "--git-common-dir"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return REPO_ROOT
    common = Path(out.stdout.strip())
    return common.parent if out.returncode == 0 and common.name == ".git" else REPO_ROOT


SHARED_ROOT = _shared_root()
DEFAULT_CONFIG = REPO_ROOT / "competition.toml"
RUNS_DIR = SHARED_ROOT / "runs"                     # local, gitignored, shared by worktrees
SHARED_LB_DIR = RUNS_DIR / "_lb"                    # public LB scores seen on this machine
SUBMISSIONS_DIR = SHARED_ROOT / "submissions"       # local, gitignored
REGISTRY_DIR = REPO_ROOT / "results" / "registry"   # committed via PRs (cross-machine)
LB_DIR = REPO_ROOT / "results" / "lb"               # committed via PRs (cross-machine)

TASKS = frozenset({"binary", "multiclass", "regression", "survival"})
CV_STRATEGIES = frozenset({"auto", "kfold", "stratified", "group", "date_blocks", "time"})


class SpecError(ValueError):
    """Raised when competition.toml is missing fields or inconsistent."""


@dataclass(frozen=True)
class Spec:
    """Immutable description of the competition problem."""

    name: str
    task: str
    metric: str
    id_col: str
    target_cols: tuple[str, ...]
    data_dir: Path
    x_train: str | tuple[str, ...]
    x_test: str | tuple[str, ...]
    y_train: str | None = None
    sample_submission: str | None = None
    hidden_y_test: str | None = None
    date_col: str | None = None
    group_col: str | None = None
    weight_col: str | None = None          # "abs_target" = |raw target| (QRT 2021 metric)
    binarize_target: bool = False          # real-valued target scored on its sign (QRT 2026)
    x_prefixes: tuple[str, ...] = ()       # prefixes when x_train/x_test are lists of files
    submission_cols: tuple[str, ...] = ()
    drop_cols: tuple[str, ...] = ()
    cv: str = "auto"
    n_splits: int = 5
    embargo: int = 0
    seed: int = 42
    metric_params: dict = field(default_factory=dict)

    @property
    def target(self) -> str:
        """Single target column name (first entry for multi-column targets)."""
        return self.target_cols[0]

    @property
    def out_cols(self) -> tuple[str, ...]:
        """Columns the submission file must contain besides the id column."""
        return self.submission_cols or self.target_cols

    def path(self, name: str | None) -> Path | None:
        return None if not name else self.data_dir / name

    def paths(self, names: str | tuple[str, ...]) -> list[Path]:
        return [self.data_dir / n for n in ((names,) if isinstance(names, str) else names)]


def _opt(value: str | None) -> str | None:
    return value or None


def _files(value: str | list[str]) -> str | tuple[str, ...]:
    return value if isinstance(value, str) else tuple(value)


def _validate(spec: Spec) -> Spec:
    if spec.task not in TASKS:
        raise SpecError(f"task must be one of {sorted(TASKS)}, got {spec.task!r}")
    if spec.cv not in CV_STRATEGIES:
        raise SpecError(f"cv must be one of {sorted(CV_STRATEGIES)}, got {spec.cv!r}")
    if spec.task == "survival" and len(spec.target_cols) != 2:
        raise SpecError("survival needs target_cols = [time_col, event_col]")
    if spec.n_splits < 2:
        raise SpecError("n_splits must be >= 2")
    if spec.binarize_target and spec.task != "binary":
        raise SpecError("binarize_target requires task = 'binary'")
    if spec.x_prefixes and isinstance(spec.x_train, str):
        raise SpecError("x_prefixes needs x_train/x_test given as lists of files")
    from qrtc.metrics import METRICS  # local import: metrics imports nothing from config

    if spec.metric not in METRICS:
        raise SpecError(f"unknown metric {spec.metric!r}; known: {sorted(METRICS)}")
    return spec


def load_spec(path: str | Path | None = None) -> Spec:
    """Load and validate the competition spec (default: $QRTC_CONFIG or competition.toml)."""
    cfg_path = Path(path or os.environ.get("QRTC_CONFIG") or DEFAULT_CONFIG)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.exists():
        raise SpecError(f"config not found: {cfg_path}")
    raw = tomllib.loads(cfg_path.read_text())
    comp, files = raw.get("competition", {}), raw.get("files", {})
    missing = [k for k in ("task", "metric", "id_col", "target_cols") if k not in comp]
    missing += [f"files.{k}" for k in ("dir", "x_train", "x_test") if k not in files]
    if missing:
        raise SpecError(f"{cfg_path.name} is missing: {missing}")
    data_dir = Path(files["dir"])
    spec = Spec(
        name=comp.get("name", cfg_path.stem),
        task=comp["task"],
        metric=comp["metric"],
        id_col=comp["id_col"],
        target_cols=tuple(comp["target_cols"]),
        data_dir=data_dir if data_dir.is_absolute() else SHARED_ROOT / data_dir,
        x_train=_files(files["x_train"]),
        x_test=_files(files["x_test"]),
        y_train=_opt(files.get("y_train")),
        sample_submission=_opt(files.get("sample_submission")),
        hidden_y_test=_opt(files.get("hidden_y_test")),
        date_col=_opt(comp.get("date_col")),
        group_col=_opt(comp.get("group_col")),
        weight_col=_opt(comp.get("weight_col")),
        binarize_target=bool(comp.get("binarize_target", False)),
        x_prefixes=tuple(files.get("x_prefixes", ())),
        submission_cols=tuple(comp.get("submission_cols", ())),
        drop_cols=tuple(comp.get("drop_cols", ())),
        cv=comp.get("cv", "auto"),
        n_splits=int(comp.get("n_splits", 5)),
        embargo=int(comp.get("embargo", 0)),
        seed=int(comp.get("seed", 42)),
        metric_params=dict(raw.get("metric_params", {})),
    )
    return _validate(spec)
