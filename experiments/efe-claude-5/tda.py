"""Issue #38: topological (0-dim persistent homology) features of the intraday idio path.

Path = stock cumulative log return minus its DATE_ID x COUNTRY equal-weight cumulative path,
in units of the stock's own 5-min idio vol (scale-free). Sublevel persistence pairs local minima
with the maxima that merge them ("dips"); superlevel (on -path) does the same for "peaks".
Per side: max / total persistence, #pairs above 1 and 2 vol units, persistence entropy,
time and depth of the most persistent dip/peak, and where the cutoff sits relative to it.
Base: beta-idio lgs-fw (repro 0.54335).

Usage: uv run python experiments/efe-claude-5/tda.py
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dmr-claude-3"))
import run_lane_a as lane  # noqa: E402

from qrtc import config, io, tracking  # noqa: E402
from qrtc.cv import make_splits  # noqa: E402
from qrtc.train import run_cv  # noqa: E402

RAW = config.SHARED_ROOT / "data" / "raw" / "qrt-x-eth-hackathon-24-11-2026"
NB = 95


def persistence_pairs(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """0-dim sublevel persistence of a 1-D sequence: (persistence, birth index) per finite pair."""
    n = len(f)
    parent = np.full(n, -1)
    birth = np.zeros(n, dtype=int)  # root -> index of its minimum

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pers, bidx = [], []
    for i in np.argsort(f, kind="stable"):
        parent[i], birth[i] = i, i
        for j in (i - 1, i + 1):
            if 0 <= j < n and parent[j] != -1:
                a, b = find(i), find(j)
                if a == b:
                    continue
                young, old = (a, b) if f[birth[a]] > f[birth[b]] else (b, a)
                if young != i or birth[young] != i:  # skip zero-length self pairs
                    pers.append(f[i] - f[birth[young]])
                    bidx.append(birth[young])
                parent[young] = old
    p = np.asarray(pers)
    keep = p > 1e-12
    return p[keep], np.asarray(bidx, dtype=int)[keep]


def side_stats(f: np.ndarray) -> list[float]:
    p, b = persistence_pairs(f)
    n = len(f)
    if len(p) == 0:
        return [0.0, 0.0, 0, 0, 0.0, np.nan, 0.0, f[-1] - f.min()]
    w = p / p.sum()
    k = int(np.argmax(p))
    return [p.max(), p.sum(), int((p > 1).sum()), int((p > 2).sum()),
            float(-(w * np.log(w)).sum()), b[k] / n, p[k], f[-1] - f.min()]


SIDE = ["max", "total", "n1", "n2", "entropy", "t_top", "depth_top", "last_vs_ext"]
COLS = [f"tda_dip_{s}" for s in SIDE] + [f"tda_peak_{s}" for s in SIDE]


def row_feats(path: np.ndarray) -> list[float]:
    if not np.isfinite(path).all():
        path = pd.Series(path).ffill().bfill().fillna(0).to_numpy()
    return side_stats(path) + side_stats(-path)


def idio_paths() -> tuple[np.ndarray, np.ndarray]:
    raw = pd.concat([pd.read_parquet(RAW / f, columns=["ID", "DATE_ID", "COUNTRY", "Price TS"])
                     for f in ("x_train.parquet", "x_test.parquet")], ignore_index=True)
    R = np.full((len(raw), NB), np.nan)
    for i, p in enumerate(raw["Price TS"]):
        r = np.diff(np.log(np.asarray(p, dtype=float)))[-NB:]
        R[i, NB - len(r):] = r
    grp = (raw["DATE_ID"].astype(str) + "_" + raw["COUNTRY"].astype(str)).to_numpy()
    M = pd.DataFrame(R).groupby(grp).transform("mean").to_numpy()
    E = np.nan_to_num(R - M)
    vol = E.std(axis=1) + 1e-9
    return np.cumsum(E, axis=1) / vol[:, None], raw["ID"].to_numpy()


def tda_features() -> pd.DataFrame:
    paths, ids = idio_paths()
    with Pool(6) as pool:
        rows = pool.map(row_feats, list(paths), chunksize=2000)
    return pd.DataFrame(rows, columns=COLS, index=ids)


def main() -> None:
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    largs = lane.parse_args(["--families", "beta,idio", "--base", "spec"])
    X, X_test = lane.features_for(ds, spec, largs, ["beta", "idio"])
    t = tda_features()
    X = pd.concat([X.reset_index(drop=True),
                   t.loc[ds.train[spec.id_col]].reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True),
                        t.loc[ds.test[spec.id_col]].reset_index(drop=True)], axis=1)
    splits = make_splits(ds.train, ds.y_vector, spec, test=ds.test)
    weight = io.train_weight(ds, spec)
    print(f"feats={X.shape[1]} tda={len(COLS)}")
    res = run_cv(spec, X, ds.y_vector, X_test, splits, "lgbm_shallow", None, weight,
                 ds.train[spec.date_col].to_numpy(), fit_weight=weight)
    run_id = tracking.save_run(res, spec, "tda-beta-idio-lgs-fw",
                               ds.train[spec.id_col].to_numpy(),
                               ds.test[spec.id_col].to_numpy(), notes="issue #38")
    print("saved", run_id)
    imp = res.importance
    print(imp[[c for c in imp.index if str(c).startswith("tda_")]].to_string())


if __name__ == "__main__":
    main()
