"""Issue #13: stock-level OOF priors + intraday-shape features vs the baseline lgbm.

Usage: uv run python experiments/efe-claude-5/stock_priors.py <variant> [--save] [--model=lgbm]
  base   reproduce baseline (same features, splits, model)
  te     + OOF target encoding of STOCK_ID (signed Close return), inside the folds
  shape  + shape features from the raw 5-minute Price TS
  both   te + shape
  mkt    + same-day market features and intraday beta (issue #27); combine with "+", e.g. te+mkt
  add "_w" (or use "w") to also train with |Close return| sample weights (issue #24)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from qrtc import config, features, io, tracking
from qrtc.cv import make_splits
from qrtc.train import run_cv

RAW = config.SHARED_ROOT / "data" / "raw" / "qrt-x-eth-hackathon-24-11-2026"


def _shape_row(p: np.ndarray) -> list[float]:
    """Unsupervised shape stats of one price path (last element = cutoff)."""
    p = p[np.isfinite(p)]
    if len(p) < 13:
        return [np.nan] * 7
    r = np.diff(np.log(p))
    hi, lo, last = p.max(), p.min(), p[-1]
    rng = hi - lo
    sd = r.std() + 1e-12
    ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if r[:-1].std() > 0 and r[1:].std() > 0 else 0.0
    mean_px = p.mean()
    return [
        (last - lo) / rng if rng > 0 else 0.5,          # position in day range
        np.log(last / mean_px) / sd,                     # distance from day mean, vol units
        r[-12:].sum() / (sd * np.sqrt(12)),              # last-hour trend z
        r[-3:].sum() / (sd * np.sqrt(3)),                # last-15-min z
        ac1,                                             # 5-min return autocorr
        float(np.argmax(p) / len(p)),                    # when the high happened
        float(np.argmin(p) / len(p)),                    # when the low happened
    ]


SHAPE_COLS = ["sh_range_pos", "sh_dist_mean_z", "sh_last12_z", "sh_last3_z", "sh_ac1",
              "sh_t_high", "sh_t_low"]


def shape_features(ids: np.ndarray) -> pd.DataFrame:
    raw = pd.concat([pd.read_parquet(RAW / f, columns=["ID", "Price TS"])
                     for f in ("x_train.parquet", "x_test.parquet")])
    raw = raw.set_index("ID").loc[ids]
    vals = np.array([_shape_row(np.asarray(p, dtype=float)) for p in raw["Price TS"]])
    return pd.DataFrame(vals, columns=SHAPE_COLS, index=range(len(ids)))


def add_cs_rank(df: pd.DataFrame, dates: np.ndarray, cols: list[str]) -> pd.DataFrame:
    """Per-date cross-sectional rank of the shape stats (target is demeaned per date)."""
    tmp = df[cols].assign(_d=dates)
    ranks = tmp.groupby("_d")[cols].rank(pct=True)
    return df.assign(**{f"{c}_csr": ranks[c].to_numpy() for c in cols})


def main(variant: str, save: bool, model: str = "lgbm") -> None:
    weighted = variant == "w" or variant.endswith("_w")
    parts = set(variant.removesuffix("_w").split("+"))
    if "both" in parts:
        parts |= {"te", "shape"}
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    feats = io.feature_columns(ds.train, spec)
    cats = [c for c in features.auto_categoricals(ds.train[feats]) if c in feats]
    X, X_test = features.to_category(ds.train[feats], ds.test[feats], cats)
    splits = make_splits(ds.train, ds.y_vector, spec, spec.cv)
    weight = io.train_weight(ds, spec)
    groups = ds.train[spec.date_col].to_numpy()

    if "te" in parts:
        y_signed = ds.y_raw.iloc[:, 0].to_numpy(dtype=float) if ds.y_raw is not None else None
        key_tr = ds.train[["STOCK_ID"]].reset_index(drop=True)
        key_te = ds.test[["STOCK_ID"]].reset_index(drop=True)
        tr_e, te_e = features.oof_target_encode(key_tr, key_te, "STOCK_ID", y_signed * 1e3,
                                                splits, smoothing=50.0)
        X = X.reset_index(drop=True).assign(stock_te=tr_e["STOCK_ID_te"].to_numpy())
        X_test = X_test.reset_index(drop=True).assign(stock_te=te_e["STOCK_ID_te"].to_numpy())

    if "shape" in parts:
        sh_tr = shape_features(ds.train[spec.id_col].to_numpy())
        sh_te = shape_features(ds.test[spec.id_col].to_numpy())
        sh_tr = add_cs_rank(sh_tr, groups, SHAPE_COLS)
        sh_te = add_cs_rank(sh_te, ds.test[spec.date_col].to_numpy(), SHAPE_COLS)
        X = pd.concat([X.reset_index(drop=True), sh_tr], axis=1)
        X_test = pd.concat([X_test.reset_index(drop=True), sh_te], axis=1)

    if "mkt" in parts:
        import market_feats
        mk = market_feats.market_features()
        X = pd.concat([X.reset_index(drop=True),
                       mk.loc[ds.train[spec.id_col]].reset_index(drop=True)], axis=1)
        X_test = pd.concat([X_test.reset_index(drop=True),
                            mk.loc[ds.test[spec.id_col]].reset_index(drop=True)], axis=1)

    print(f"variant={variant} weighted={weighted} features={X.shape[1]} folds={len(splits)}")
    fit_w = weight / weight.mean() if weighted else None
    result = run_cv(spec, X, ds.y_vector, X_test, splits, model, None, weight, groups,
                    fit_weight=fit_w)
    if save:
        slug = f"stockpri-{variant.replace('+', '-')}" + ("" if model == "lgbm" else f"-{model}")
        run_id = tracking.save_run(result, spec, slug,
                                   ds.train[spec.id_col].to_numpy(),
                                   ds.test[spec.id_col].to_numpy(), notes="issue #13")
        print("saved", run_id)
    print(result.importance.head(15).to_string())


if __name__ == "__main__":
    mdl = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), "lgbm")
    main(sys.argv[1], "--save" in sys.argv, mdl)
