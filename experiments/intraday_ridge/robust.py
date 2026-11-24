"""How much of a candidate's CV edge is real? Day-bootstrap of OOF weighted accuracy.

Resamples whole DATE_IDs with replacement (days are the independent unit: rows of one day
share the market move) and reports, for each run, the bootstrap mean / 90% interval, and for
each run vs the reference run, the paired difference and P(run beats reference).

  uv run python experiments/intraday_ridge/robust.py REF_RUN RUN [RUN ...] [--n 2000]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from qrtc import config, tracking

DERIVED = config.SHARED_ROOT / "data" / "derived"


def day_sums(run: str, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-day sum of |y| * hit and of |y| for one run's OOF predictions."""
    p = tracking.load_preds(run)
    oof = pd.Series(p["oof"].astype(float), index=p["train_ids"]).reindex(frame["ID"]).to_numpy()
    hit = ((oof > 0.5) == (frame["y"].to_numpy() > 0)).astype(float)
    w = frame["y"].abs().to_numpy()
    g = pd.DataFrame({"d": frame["DATE_ID"].to_numpy(), "wh": w * hit, "w": w})
    agg = g.groupby("d")[["wh", "w"]].sum()
    return agg["wh"].to_numpy(), agg["w"].to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    frame = pd.read_parquet(DERIVED / "X_train.parquet", columns=["ID", "DATE_ID"])
    y = pd.read_parquet(DERIVED / "y_train.parquet").set_index("ID")["Close return"]
    frame = frame.assign(y=frame["ID"].map(y)).sort_values("DATE_ID").reset_index(drop=True)
    sums = {r: day_sums(r, frame) for r in a.runs}
    n_days = len(next(iter(sums.values()))[0])
    idx = np.random.default_rng(0).integers(0, n_days, size=(a.n, n_days))
    boot = {r: wh[idx].sum(1) / w[idx].sum(1) for r, (wh, w) in sums.items()}
    ref = a.runs[0]
    rows = []
    for r in a.runs:
        d = boot[r] - boot[ref]
        rows.append((r, boot[r].mean(), *np.quantile(boot[r], [0.05, 0.95]), d.mean(),
                     (d > 0).mean()))
    cols = ["run", "boot_mean", "p05", "p95", "diff_vs_ref", "P(beats ref)"]
    print(f"{n_days} days, {a.n} bootstrap resamples, reference = {ref}")
    print(pd.DataFrame(rows, columns=cols).round(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
