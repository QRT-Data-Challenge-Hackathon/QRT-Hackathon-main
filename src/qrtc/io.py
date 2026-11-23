"""Load the competition data described by the spec into one immutable ``Dataset``.

Handles the usual Challenge-Data layout (``X_train`` + ``y_train`` joined on an ID column,
``X_test``) as well as a single train file that already contains the target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from qrtc.config import Spec

READERS = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".feather": pd.read_feather,
    ".xlsx": pd.read_excel,
    ".json": pd.read_json,
}


class DataError(ValueError):
    """Raised when data files are missing or inconsistent with the spec."""


@dataclass(frozen=True)
class Dataset:
    """Train features, target(s), test features and the encoders needed to submit."""

    train: pd.DataFrame          # features + id (+ date/group), target columns removed
    y: pd.DataFrame              # target columns, aligned row-by-row with ``train``
    test: pd.DataFrame           # features + id
    classes: tuple | None = None # original class labels for classification tasks
    y_raw: pd.DataFrame | None = None  # real-valued target before binarize_target

    @property
    def y_vector(self) -> np.ndarray:
        """Model-ready target: encoded labels, float values, or (n, 2) [time, event]."""
        return encode_target(self.y, self.classes)


def read_table(path: Path) -> pd.DataFrame:
    """Read csv / parquet / feather / xlsx / json by extension."""
    if not path.exists():
        raise DataError(f"data file not found: {path} (put organiser files in data/raw/)")
    for suffix, reader in READERS.items():
        if path.name.endswith(suffix) or path.name.endswith(suffix + ".gz"):
            return reader(path)
    raise DataError(f"unsupported file type: {path.name}")


def _join_target(x: pd.DataFrame, y: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    if spec.id_col not in y.columns:
        if len(y) != len(x):
            raise DataError("y_train has no id column and a different length than x_train")
        return pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    merged = x.merge(y, on=spec.id_col, how="inner", validate="one_to_one")
    if len(merged) != len(x):
        raise DataError(f"{len(x) - len(merged)} train rows have no target after joining on id")
    return merged


def _class_labels(y: pd.DataFrame, spec: Spec) -> tuple | None:
    if spec.task not in ("binary", "multiclass"):
        return None
    if y.shape[1] > 1:  # one-hot targets (e.g. HOME_WINS / DRAW / AWAY_WINS)
        return tuple(y.columns)
    return tuple(sorted(y.iloc[:, 0].dropna().unique().tolist()))


def encode_target(y: pd.DataFrame, classes: tuple | None) -> np.ndarray:
    """Turn the target frame into model input (see qrtc.metrics conventions)."""
    if classes is None:
        return y.to_numpy(dtype=float) if y.shape[1] > 1 else y.iloc[:, 0].to_numpy(float)
    if y.shape[1] > 1:
        return y.to_numpy().argmax(axis=1)
    lookup = {c: i for i, c in enumerate(classes)}
    return y.iloc[:, 0].map(lookup).to_numpy(dtype=int)


def read_features(spec: Spec, names: str | tuple[str, ...]) -> pd.DataFrame:
    """One feature table, or several joined on the id column (prefixing via x_prefixes)."""
    frames = [read_table(p) for p in spec.paths(names)]
    if spec.x_prefixes:
        frames = [f.rename(columns={c: f"{pre}{c}" for c in f.columns if c != spec.id_col})
                  for f, pre in zip(frames, spec.x_prefixes, strict=True)]
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=spec.id_col, how="left", validate="one_to_one")
    return out


def load_dataset(spec: Spec) -> Dataset:
    """Load train/test according to the spec; validates ids, targets and test columns."""
    x_train = read_features(spec, spec.x_train)
    x_test = read_features(spec, spec.x_test)
    full = x_train
    if spec.y_train:
        full = _join_target(x_train, read_table(spec.path(spec.y_train)), spec)
    missing = [c for c in spec.target_cols if c not in full.columns]
    if missing:
        raise DataError(f"target column(s) {missing} not in training data")
    if spec.id_col not in x_test.columns:
        raise DataError(f"id column {spec.id_col!r} not in test data")
    y = full[list(spec.target_cols)].reset_index(drop=True)
    if y.isna().any().any():
        raise DataError("target has missing values; drop or impute them explicitly")
    y_raw = None
    if spec.binarize_target:
        y_raw, y = y, (y > 0).astype(int)  # zero return counts as class 0 (QRT #167)
    train = full.drop(columns=list(spec.target_cols)).reset_index(drop=True)
    return Dataset(train=train, y=y, test=x_test.reset_index(drop=True),
                   classes=_class_labels(y, spec), y_raw=y_raw)


def feature_columns(df: pd.DataFrame, spec: Spec, keep_date: bool = False) -> list[str]:
    """Model features: everything except id, weight, explicit drops (and date unless kept)."""
    excluded = {spec.id_col, spec.weight_col, *spec.drop_cols, *spec.target_cols}
    if not keep_date:
        excluded.add(spec.date_col)
    return [c for c in df.columns if c not in excluded]


def train_weight(ds: Dataset, spec: Spec) -> np.ndarray | None:
    """Per-row weight for metrics like weighted_accuracy (from weight_col in train or y)."""
    if not spec.weight_col:
        return None
    if spec.weight_col == "abs_target":
        raw = ds.y_raw if ds.y_raw is not None else ds.y
        return raw.iloc[:, 0].abs().to_numpy(dtype=float)
    for frame in (ds.train, ds.y):
        if spec.weight_col in frame.columns:
            return frame[spec.weight_col].to_numpy(dtype=float)
    raise DataError(f"weight_col {spec.weight_col!r} not found in train data")
