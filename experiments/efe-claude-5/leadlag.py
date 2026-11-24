"""Issue #41: lead-lag (lagged betas, Lévy area) + Fourier spectrum of the intraday path.

Per stock-day from 5-min returns before the cutoff: r (stock), m (DATE_ID x COUNTRY
equal-weight mean). OLS of r_t on (m_t, m_{t-1}, m_{t-2}) gives the same-bar and lagged betas;
catch-up = lag betas x the market's last bars (the move the stock has not absorbed yet).
Lévy area of (cum m, cum r) = signature level-2 antisymmetric term (>0: market leads).
Fourier: share of idio-path energy in the lowest / highest frequencies.
Base: beta-idio lgs-fw (repro 0.54335).

Usage: uv run python experiments/efe-claude-5/leadlag.py
"""

from __future__ import annotations

import sys
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


def returns() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = pd.concat([pd.read_parquet(RAW / f, columns=["ID", "DATE_ID", "COUNTRY", "Price TS"])
                     for f in ("x_train.parquet", "x_test.parquet")], ignore_index=True)
    R = np.full((len(raw), NB), np.nan)
    for i, p in enumerate(raw["Price TS"]):
        r = np.diff(np.log(np.asarray(p, dtype=float)))[-NB:]
        R[i, NB - len(r):] = r
    grp = (raw["DATE_ID"].astype(str) + "_" + raw["COUNTRY"].astype(str)).to_numpy()
    M = pd.DataFrame(R).groupby(grp).transform("mean").to_numpy()
    return np.nan_to_num(R), np.nan_to_num(M), raw["ID"].to_numpy()


def lag_betas(R: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Row-wise OLS r_t ~ m_t + m_{t-1} + m_{t-2} (t = 2..94), batched via 3x3 normal equations."""
    y = R[:, 2:]
    Xs = np.stack([M[:, 2:], M[:, 1:-1], M[:, :-2]], axis=2)          # (n, T, 3)
    XtX = np.einsum("ntk,ntl->nkl", Xs, Xs) + 1e-10 * np.eye(3)
    Xty = np.einsum("ntk,nt->nk", Xs, y)
    return np.linalg.solve(XtX, Xty[..., None])[..., 0]                # (n, 3)


def leadlag_features() -> pd.DataFrame:
    R, M, ids = returns()
    b = lag_betas(R, M)
    cr, cm = np.cumsum(R, 1), np.cumsum(M, 1)
    levy = 0.5 * np.sum(cm[:, :-1] * np.diff(cr, axis=1) - cr[:, :-1] * np.diff(cm, axis=1), 1)
    E = R - b[:, :1] * M
    vol = E.std(1) + 1e-9
    spec = np.abs(np.fft.rfft(np.cumsum(E, 1) / vol[:, None], axis=1))[:, 1:] ** 2
    tot = spec.sum(1) + 1e-12
    mvol = M.std(1) + 1e-9
    out = pd.DataFrame({
        "ll_beta0": b[:, 0], "ll_beta1": b[:, 1], "ll_beta2": b[:, 2],
        "ll_lag_share": (b[:, 1] + b[:, 2]) / (np.abs(b).sum(1) + 1e-9),
        "ll_levy": levy / (vol * mvol * NB),
        "ll_catchup1": b[:, 1] * M[:, -1] / mvol,
        "ll_catchup2": (b[:, 1] * M[:, -1] + b[:, 2] * M[:, -2] + b[:, 2] * M[:, -1]) / mvol,
        "ll_catchup_3bars": (b[:, 1] + b[:, 2]) * M[:, -3:].sum(1) / mvol,
        "ff_low3": spec[:, :3].sum(1) / tot,
        "ff_high": spec[:, 24:].sum(1) / tot,
        "ff_peak_freq": np.argmax(spec, 1).astype(float),
    }, index=ids)
    return out


def main() -> None:
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    largs = lane.parse_args(["--families", "beta,idio", "--base", "spec"])
    X, X_test = lane.features_for(ds, spec, largs, ["beta", "idio"])
    f = leadlag_features()
    X = pd.concat([X.reset_index(drop=True),
                   f.loc[ds.train[spec.id_col]].reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True),
                        f.loc[ds.test[spec.id_col]].reset_index(drop=True)], axis=1)
    y_raw = ds.y_raw.iloc[:, 0].to_numpy(dtype=float)
    for c in f.columns:  # quick univariate read, train rows only
        print(f"  corr({c}, y) = {np.corrcoef(np.nan_to_num(X[c].to_numpy(float)), y_raw)[0, 1]:+.4f}")
    splits = make_splits(ds.train, ds.y_vector, spec, test=ds.test)
    weight = io.train_weight(ds, spec)
    res = run_cv(spec, X, ds.y_vector, X_test, splits, "lgbm_shallow", None, weight,
                 ds.train[spec.date_col].to_numpy(), fit_weight=weight)
    run_id = tracking.save_run(res, spec, "leadlag-beta-idio-lgs-fw",
                               ds.train[spec.id_col].to_numpy(),
                               ds.test[spec.id_col].to_numpy(), notes="issue #41")
    print("saved", run_id)
    imp = res.importance
    print(imp[[c for c in imp.index if str(c).startswith(("ll_", "ff_"))]].to_string())


if __name__ == "__main__":
    main()
