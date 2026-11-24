"""Validator gate: paired fold deltas and a calm-day (test-like) slice for two saved runs.

Usage: uv run python experiments/dmr-claude-4/gate.py <candidate_run_id> <baseline_run_id>
Calm days = train days whose median intraday vol is <= the test days' 90th percentile, the
regime the test set lives in (audit: test days are calmer).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from qrtc import config, cv, io, metrics


def oof_by_id(run_id: str) -> pd.Series:
    z = np.load(config.SHARED_ROOT / "runs" / run_id / "preds.npz", allow_pickle=True)
    ids = z["train_ids"]
    return pd.Series(z["oof"], index=ids)


def main(cand: str, base: str) -> None:
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    r = ds.y_raw.iloc[:, 0].to_numpy()
    yb, w = (r > 0).astype(int), np.abs(r)
    ids = ds.train[spec.id_col].to_numpy()
    p = {k: oof_by_id(k).reindex(ids).to_numpy() for k in (cand, base)}
    wa = metrics.get_metric("weighted_accuracy")
    day = ds.train[spec.date_col].to_numpy()
    test = ds.test[~ds.test[spec.id_col].isin(ids)]
    calm_cut = test.groupby(spec.date_col)["vol"].median().quantile(0.9)
    calm = ds.train.groupby(spec.date_col)["vol"].transform("median").to_numpy() <= calm_cut
    splits = cv.make_splits(ds.train, yb, spec)
    for name, mask in (("all", np.ones(len(yb), bool)), ("calm", calm)):
        d = []
        for _, va in splits:
            v = va[mask[va]]
            d.append(wa(yb[v], (p[cand][v] > 0.5).astype(int), weight=w[v])
                     - wa(yb[v], (p[base][v] > 0.5).astype(int), weight=w[v]))
        d = np.array(d)
        print(f"{name:4s} days={len(np.unique(day[mask]))} delta {d.mean():+.5f} ± {d.std():.5f} "
              f"({(d > 0).sum()}/{len(d)} folds up)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
