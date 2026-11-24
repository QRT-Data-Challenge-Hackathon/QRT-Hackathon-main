"""Post-processing check: re-score a saved run's OOF predictions fold by fold
after centring them within DATE_ID x COUNTRY (the target sums to 0 in each such group).

  uv run python experiments/residual_gbm/pp_check.py <run_id> [<run_id> ...]

Variants: raw (threshold 0.5) | demean (p - group mean + 0.5) | median (split at group median).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from qrtc import config, io, tracking
from qrtc.cv import make_splits
from qrtc.metrics import get_metric


def variants(oof: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    """Raw OOF probabilities and two group-relative versions."""
    frame = pd.DataFrame({"p": oof, "g": groups})
    mean = frame.groupby("g")["p"].transform("mean").to_numpy()
    median = frame.groupby("g")["p"].transform("median").to_numpy()
    return {"raw": oof, "demean": oof - mean + 0.5, "median": oof - median + 0.5}


def fold_scores(pred: np.ndarray, y: np.ndarray, w: np.ndarray, splits: list) -> np.ndarray:
    metric = get_metric("weighted_accuracy")
    return np.array([metric(y[va], pred[va], weight=w[va]) for _, va in splits])


def main(run_ids: list[str]) -> int:
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    y, w = ds.y_vector, io.train_weight(ds, spec)
    splits = make_splits(ds.train, y, spec, test=ds.test)
    groups = (ds.train[spec.date_col].astype(str) + "_"
              + ds.train["COUNTRY"].astype(str)).to_numpy()
    for run_id in run_ids:
        oof = tracking.load_preds(run_id, train_ids=ds.train[spec.id_col].to_numpy())["oof"]
        scores = {k: fold_scores(v, y, w, splits) for k, v in variants(oof, groups).items()}
        raw = scores["raw"]
        for name, s in scores.items():
            d = s - raw
            print(f"{run_id} {name:6s}: {s.mean():.5f} ± {s.std():.5f}  vs raw "
                  f"{d.mean():+.5f} ± {d.std():.5f} ({(d > 0).sum()}/{len(d)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
