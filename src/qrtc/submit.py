"""Build, validate and write submission files. A wrong file wastes a submission slot, so
``write_submission`` always validates first and refuses to write an invalid file.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qrtc import config
from qrtc.config import Spec

LABEL_METRICS = frozenset({"accuracy", "weighted_accuracy", "balanced_accuracy", "f1"})


class SubmissionError(ValueError):
    """Raised when a submission does not match the expected format."""


def _format_values(spec: Spec, pred: np.ndarray, classes: tuple | None,
                   threshold: float) -> dict[str, np.ndarray]:
    pred = np.asarray(pred)
    cols = spec.out_cols
    if spec.task == "multiclass" and len(cols) > 1:  # one-hot output columns
        labels = pred.argmax(axis=1)
        return {c: (labels == i).astype(int) for i, c in enumerate(cols)}
    if spec.task in ("binary", "multiclass") and spec.metric in LABEL_METRICS:
        idx = pred.argmax(axis=1) if pred.ndim == 2 else (pred > threshold).astype(int)
        if spec.signed_labels:  # sign-scored platforms count an exact 0 as wrong
            return {cols[0]: np.where(idx == 1, 1, -1)}
        values = np.asarray(classes, dtype=object)[idx] if classes else idx
        return {cols[0]: np.asarray(values.tolist())}
    if pred.ndim == 2:
        return {f"{cols[0]}_{c}": pred[:, i] for i, c in enumerate(classes or range(pred.shape[1]))}
    return {cols[0]: pred}


def build_submission(spec: Spec, test_ids: np.ndarray, pred: np.ndarray,
                     classes: tuple | None = None, threshold: float = 0.5) -> pd.DataFrame:
    """Submission frame: id column + target column(s) in the format the metric expects."""
    values = _format_values(spec, pred, classes, threshold)
    return pd.DataFrame({spec.id_col: np.asarray(test_ids), **values})


def validate_submission(sub: pd.DataFrame, spec: Spec, test_ids: np.ndarray,
                        sample: pd.DataFrame | None = None) -> list[str]:
    """Return a list of problems (empty list = valid)."""
    problems = []
    if spec.id_col not in sub.columns:
        return [f"missing id column {spec.id_col!r}"]
    if len(sub) != len(test_ids):
        problems.append(f"{len(sub)} rows, expected {len(test_ids)}")
    if sub[spec.id_col].duplicated().any():
        problems.append("duplicate ids")
    if set(sub[spec.id_col]) != set(np.asarray(test_ids).tolist()):
        problems.append("id set differs from test ids")
    if sub.drop(columns=[spec.id_col]).isna().any().any():
        problems.append("NaN predictions")
    if sample is not None:
        if list(sample.columns) != list(sub.columns):
            problems.append(f"columns {list(sub.columns)} != sample {list(sample.columns)}")
        elif len(sample) == len(sub) and not sample[spec.id_col].equals(sub[spec.id_col]):
            problems.append("row order differs from sample submission (reorder to match)")
    return problems


def load_sample(spec: Spec) -> pd.DataFrame | None:
    path = spec.path(spec.sample_submission)
    return pd.read_csv(path) if path and path.exists() else None


def align_to_sample(sub: pd.DataFrame, sample: pd.DataFrame | None, spec: Spec) -> pd.DataFrame:
    """Reorder rows like the sample submission (if one exists)."""
    if sample is None:
        return sub
    return sample[[spec.id_col]].merge(sub, on=spec.id_col, how="left")


def write_submission(sub: pd.DataFrame, spec: Spec, test_ids: np.ndarray, tag: str,
                     out_dir: Path | None = None) -> Path:
    """Validate then write ``submissions/<time>_<tag>.csv``; raises SubmissionError if invalid."""
    sample = load_sample(spec)
    sub = align_to_sample(sub, sample, spec)
    problems = validate_submission(sub, spec, test_ids, sample)
    if problems:
        raise SubmissionError("invalid submission: " + "; ".join(problems))
    directory = out_dir or config.SUBMISSIONS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{tag}.csv"
    if path.exists():
        path = directory / f"{tag}_{datetime.now():%H%M%S}.csv"
    sub.to_csv(path, index=False)
    return path
