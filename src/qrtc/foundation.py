"""Pretrained tabular foundation models: TabICLv2, TabPFN, TabDPT (optional extra ``fm``).

They learn in context: ``fit`` stores the training rows and ``predict`` runs one forward pass
over training and test rows, with no gradient training and no tuning. On small and medium tables
(up to ~50k rows, the size of most QRT challenges) they often beat tuned GBMs, and their errors
differ from GBM errors, so they also help blends. ``make models`` installs them and downloads
the weights; after that they run offline on the Apple-Silicon GPU (MPS).
They run in a separate worker process (qrtc.fm_worker) because torch and LightGBM crash when
loaded into one process on macOS; ``QRTC_FM_INPROCESS=1`` turns that off.

Weight licences: TabICL BSD-3-Clause, TabDPT Apache-2.0, TabPFN v2 Prior Labs License
(Apache-2.0 plus attribution: write "built with TabPFN"). TabPFN 2.5 and later
(``version="v3_5"`` etc.) are non-commercial and need a one-time licence acceptance
(``TABPFN_TOKEN``), so check a competition's rules before using them.

Params (``--params``) besides the backend's own constructor arguments:
  max_rows  context cap: larger training sets are subsampled (stratified for classification)
  n_bags    average this many models, each fit on its own subsample (only when capped)
  version   TabPFN weights: v2 (default), v2_5, v2_6, v3, v3_5, v3_5_fast
"""

from __future__ import annotations

import atexit
import importlib.util
import os
import pickle
import subprocess
import sys
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from qrtc import fm_worker

ISOLATE = True  # run backends in the worker process (False: in this process, for tests)
NAMES = frozenset({"tabicl", "tabpfn", "tabdpt"})
TASKS = frozenset({"binary", "multiclass", "regression"})
DEFAULTS: dict[str, dict[str, Any]] = {
    "tabicl": dict(max_rows=50_000, n_bags=1),
    "tabpfn": dict(max_rows=10_000, n_bags=1, version="v2"),
    "tabdpt": dict(max_rows=50_000, n_bags=1),
}
OWN_PARAMS = frozenset({"max_rows", "n_bags", "version"})
INSTALL_HINT = "install the optional models first: make models (uv sync --all-groups --extra fm)"


class MissingModelError(ValueError):
    """Raised when a foundation model's package is not installed."""


def available(name: str) -> bool:
    """True if the package behind ``name`` is importable."""
    return importlib.util.find_spec(name) is not None


def _categorical(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
            or isinstance(X[c].dtype, pd.CategoricalDtype)]


@dataclass(frozen=True)
class Encoder:
    """Ordinal codes for categorical columns (unknown or missing -> NaN), floats elsewhere."""

    columns: tuple[str, ...]
    categories: dict[str, pd.Index]

    @classmethod
    def fit(cls, X: pd.DataFrame) -> Encoder:
        cats = {c: pd.Index(pd.Series(X[c]).dropna().astype(str).unique()).sort_values()
                for c in _categorical(X)}
        return cls(tuple(X.columns), cats)

    @property
    def categorical_indices(self) -> list[int]:
        return [i for i, c in enumerate(self.columns) if c in self.categories]

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        coded = {c: pd.Categorical(X[c].astype(str).where(X[c].notna()), categories=cats).codes
                 for c, cats in self.categories.items()}
        floats = {c: np.where(codes < 0, np.nan, codes).astype(float)
                  for c, codes in coded.items()}
        return X[list(self.columns)].assign(**floats).to_numpy(dtype=float)


def context_rows(y: np.ndarray, task: str, max_rows: int, seed: int) -> np.ndarray:
    """Sorted row indices of a subsample of at most ``max_rows`` rows (all rows if fewer).

    Classification keeps class shares and at least one row per class."""
    n = len(y)
    if n <= max_rows:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    if task == "regression":
        return np.sort(rng.choice(n, max_rows, replace=False))
    picks = []
    for cls in np.unique(y):
        members = np.flatnonzero(y == cls)
        k = min(len(members), max(1, round(max_rows * len(members) / n)))
        picks.append(rng.choice(members, k, replace=False))
    return np.sort(np.concatenate(picks))


def _tabicl(task: str, p: dict[str, Any], seed: int, cat_idx: list[int]) -> Any:
    from tabicl import TabICLClassifier, TabICLRegressor

    cls = TabICLRegressor if task == "regression" else TabICLClassifier
    return cls(random_state=seed, **p)


def _tabpfn(task: str, p: dict[str, Any], seed: int, cat_idx: list[int]) -> Any:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    version = str(p.get("version", "v2")).upper().replace(".", "_")
    if version not in ModelVersion.__members__:
        raise ValueError(f"tabpfn version must be one of {list(ModelVersion.__members__)}")
    rest = {k: v for k, v in p.items() if k != "version"}
    rest.setdefault("ignore_pretraining_limits", True)
    cls = TabPFNRegressor if task == "regression" else TabPFNClassifier
    return cls.create_default_for_version(ModelVersion[version], random_state=seed,
                                          categorical_features_indices=cat_idx or None, **rest)


def _tabdpt(task: str, p: dict[str, Any], seed: int, cat_idx: list[int]) -> Any:
    from tabdpt import TabDPTClassifier, TabDPTRegressor

    cls = TabDPTRegressor if task == "regression" else TabDPTClassifier
    return cls(**{"verbose": False, **p})


BUILDERS = {"tabicl": _tabicl, "tabpfn": _tabpfn, "tabdpt": _tabdpt}


class Worker:
    """The child process (``python -m qrtc.fm_worker``) that runs every foundation-model call.

    torch never loads into the parent, so LightGBM keeps working there (see qrtc.fm_worker)."""

    def __init__(self) -> None:
        env = {"OMP_NUM_THREADS": str(max(1, (os.cpu_count() or 2) - 2)), **os.environ,
               "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        self.proc = subprocess.Popen([sys.executable, "-m", "qrtc.fm_worker"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def call(self, request: dict[str, Any]) -> np.ndarray:
        try:
            pickle.dump(request, self.proc.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            self.proc.stdin.flush()
            status, payload = pickle.load(self.proc.stdout)
        except (EOFError, BrokenPipeError, OSError) as exc:
            code = self.proc.wait()
            raise RuntimeError(f"foundation-model worker died (exit code {code}); the next "
                               "call starts a new one") from exc
        if status == "error":
            raise payload
        return payload

    def close(self) -> None:
        if not self.alive():
            return
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


_WORKER: Worker | None = None


def _worker() -> Worker:
    global _WORKER
    if _WORKER is None or not _WORKER.alive():
        _WORKER = Worker()
        atexit.register(_WORKER.close)
    return _WORKER


def shutdown_worker() -> None:
    """Stop the worker process (it also stops when Python exits)."""
    global _WORKER
    if _WORKER is not None:
        _WORKER.close()
        _WORKER = None


def _call(request: dict[str, Any]) -> np.ndarray:
    if os.environ.get("QRTC_FM_INPROCESS") == "1" or not ISOLATE:
        return fm_worker.run(request)
    return _worker().call(request)


def _spec(fn: Any) -> str:
    return f"{fn.__module__}:{fn.__qualname__}"


class ICLEstimator:
    """sklearn-style wrapper: encodes categoricals, caps the context, averages bags.

    ``fit`` only stores the encoded context rows; each predict call sends context + query rows
    to the worker, which builds, fits and predicts (in-context models do the work there)."""

    def __init__(self, name: str, task: str, params: dict[str, Any] | None = None,
                 seed: int = 42):
        if name not in NAMES:
            raise ValueError(f"unknown foundation model {name!r}; use one of {sorted(NAMES)}")
        if task not in TASKS:
            raise ValueError(f"{name} supports {sorted(TASKS)}, not {task!r}")
        if not available(name):
            raise MissingModelError(f"{name} is not installed; {INSTALL_HINT}")
        merged = {**DEFAULTS[name], **(params or {})}
        if "version" in merged and name != "tabpfn":
            raise ValueError("'version' is a tabpfn param; tabicl uses checkpoint_version")
        self.name, self.task, self.seed = name, task, seed
        self.max_rows, self.n_bags = int(merged["max_rows"]), int(merged["n_bags"])
        if self.max_rows < 2 or self.n_bags < 1:
            raise ValueError("max_rows must be >= 2 and n_bags >= 1")
        keep = OWN_PARAMS - {"version"} if name == "tabpfn" else OWN_PARAMS
        self.backend_params = {k: v for k, v in merged.items() if k not in keep}
        self.context_: tuple[np.ndarray, ...] = ()

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None
            ) -> ICLEstimator:
        if sample_weight is not None:
            warnings.warn(f"{self.name} ignores sample_weight", UserWarning, stacklevel=2)
        self.y_ = np.asarray(y)
        self.encoder_ = Encoder.fit(X)
        self.X_ = self.encoder_.transform(X)
        bags = self.n_bags if len(self.y_) > self.max_rows else 1
        self.context_ = tuple(context_rows(self.y_, self.task, self.max_rows, self.seed + b)
                              for b in range(bags))
        self.classes_ = np.unique(self.y_) if self.task != "regression" else None
        return self

    def _predict_bags(self, X: pd.DataFrame, method: str) -> np.ndarray:
        Xq = self.encoder_.transform(X)
        preds = []
        for b, rows in enumerate(self.context_):
            kwargs = {"seed": self.seed + b} if self.name == "tabdpt" and method != "predict" \
                else {}
            preds.append(_call(dict(
                builder=_spec(BUILDERS[self.name]), task=self.task,
                params=self.backend_params, seed=self.seed + b,
                cat_idx=self.encoder_.categorical_indices, X=self.X_[rows], y=self.y_[rows],
                X_query=Xq, method=method, kwargs=kwargs)))
        return np.mean(preds, axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "regression":
            raise AttributeError("regression models have no predict_proba")
        return self._predict_bags(X, "predict_proba")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "regression":
            return self.classes_[self.predict_proba(X).argmax(axis=1)]
        return self._predict_bags(X, "predict")


def smoke_test(name: str, task: str, params: dict[str, Any] | None = None,
               n: int = 120, seed: int = 0) -> float:
    """Fit and predict a tiny synthetic table (downloads the weights on first use).

    Returns the in-sample score (accuracy or correlation) as a sanity number."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd")).assign(
        cat=rng.choice(["x", "y", "z"], n))
    signal = X["a"].to_numpy() + 0.5 * (X["cat"] == "x").to_numpy()
    if task == "regression":
        y = signal + 0.1 * rng.normal(size=n)
    elif task == "binary":
        y = (signal > 0).astype(int)
    else:
        y = np.digitize(signal, [-0.5, 0.5])
    pred = ICLEstimator(name, task, params, seed).fit(X, y).predict(X)
    if task == "regression":
        return float(np.corrcoef(pred, y)[0, 1])
    return float((pred == y).mean())
