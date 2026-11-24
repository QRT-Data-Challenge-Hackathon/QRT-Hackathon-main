"""Issue #35: kNN analog model on scale-free path-shape features (new family for the blend).

Neighbours and scaler are fit inside each fold; the test prediction is fit on all training rows.
Prediction = mean real Close return of the k nearest stock-days, mapped to sigmoid(pred / std).

Usage: uv run python experiments/efe-claude-5/knn.py K
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dmr-claude-3"))
import run_lane_a as lane  # noqa: E402

from qrtc import config, io, tracking  # noqa: E402
from qrtc.cv import make_splits  # noqa: E402
from qrtc.train import CVResult, score_predictions  # noqa: E402

WINDOWS = ("w1", "w2", "w3", "w6", "w12", "w24", "w48")
DIRECT = ["idio_day_t", "idio_w3_t", "idio_w12_t", "idio_w48_t", "idio_pos", "idio_autocorr1",
          "beta", "corr_country", "ret_day_rank", "ret_12_rank", "ret_3_rank", "vol_rank"]


def shape_matrix(X: pd.DataFrame) -> np.ndarray:
    iv = X["idio_vol"].to_numpy(dtype=float) + 1e-9
    cols = [X[f"idio_{w}"].to_numpy(dtype=float) / (iv * np.sqrt(int(w[1:]))) for w in WINDOWS]
    cols += [X[c].to_numpy(dtype=float) for c in DIRECT]
    M = np.column_stack(cols)
    return np.nan_to_num(np.clip(M, -50, 50))


def fit_predict(M_fit, y_fit, M_pred, k) -> np.ndarray:
    sc = StandardScaler().fit(M_fit)
    a, b = np.clip(sc.transform(M_fit), -5, 5), np.clip(sc.transform(M_pred), -5, 5)
    return KNeighborsRegressor(n_neighbors=k, n_jobs=6).fit(a, y_fit).predict(b)


def main(k: int) -> None:
    t0 = time.time()
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    largs = lane.parse_args(["--families", "beta,idio", "--base", "spec"])
    X, X_test = lane.features_for(ds, spec, largs, ["beta", "idio"])
    M, M_test = shape_matrix(X), shape_matrix(X_test)
    raw = ds.y_raw.iloc[:, 0].to_numpy(dtype=float)
    lo, hi = np.quantile(raw, [0.01, 0.99])
    y = np.clip(raw, lo, hi)
    splits = make_splits(ds.train, ds.y_vector, spec, test=ds.test)
    weight = io.train_weight(ds, spec)
    groups = ds.train[spec.date_col].to_numpy()
    oof = np.full(len(M), np.nan)
    for tr, va in splits:
        oof[va] = fit_predict(M[tr], y[tr], M[va], k)
    test = fit_predict(M, y, M_test, k)
    scale = float(np.nanstd(oof)) or 1.0
    oof_p, test_p = (1 / (1 + np.exp(-v / scale)) for v in (oof, test))
    folds = [score_predictions(spec, ds.y_vector, oof_p, va, weight, groups) for _, va in splits]
    for i, s in enumerate(folds):
        print(f"  fold {i}: {spec.metric} = {s:.5f}")
    res = CVResult(model=f"knn{k}", params={"k": k}, features=tuple(map(str, range(M.shape[1]))),
                   metric=spec.metric, score=float(np.mean(folds)), score_std=float(np.std(folds)),
                   fold_scores=tuple(folds), oof=oof_p, test_pred=test_p,
                   elapsed_s=time.time() - t0)
    print(f"knn k={k} dims={M.shape[1]}: {res.score:.5f} ± {res.score_std:.5f} ({res.elapsed_s:.0f}s)")
    run_id = tracking.save_run(res, spec, f"knn{k}-shape", ds.train[spec.id_col].to_numpy(),
                               ds.test[spec.id_col].to_numpy(), notes="issue #35")
    print("saved", run_id)


if __name__ == "__main__":
    main(int(sys.argv[1]))
