"""Estimate how runs will score on the private test days, from their OOF predictions.

The public LB is computed on the TRAINING rows (in-sample), so it says nothing about the test.
This script answers "how would this run do on 199 unseen days?" instead:

1. pooled OOF weighted accuracy of each run (matches `qrtc blend RUN --method mean`),
2. day bootstrap: resample whole days (392 = our CV, 199 = private test size): score range,
   gain over a reference run and P(beats reference), paired on the same days,
3. test-like days: training days reweighted by p(test)/p(train) from day-level calm/volatility
   (the test period is ~14% calmer; adversarial AUC 0.66),
4. --adv: adversarial validation train vs test rows (slow-ish, LightGBM).

Usage:
  uv run python experiments/dmr-claude-0/validate_candidates.py RUN [RUN ...] [--base=RUN] [--adv]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from qrtc import config

DATA = config.load_spec().data_dir
DEFAULT_BASE = "1124-190254_dmr-claude-0_blend-rank"     # current best (U003), CV 0.5440
B = 4000
TEST_FIRST_ID = 144225


def load_oof(run: str, ids: np.ndarray) -> np.ndarray:
    z = np.load(config.RUNS_DIR / run / "preds.npz", allow_pickle=True)
    return pd.Series(z["oof"], index=z["train_ids"]).reindex(ids).to_numpy()


def report(runs: list[str], base: str, scores: dict, label: str) -> None:
    print(f"\n{label}: score [95%] | gain vs {base[12:]} [95%] | P(beats it)")
    for r in runs:
        s, d = scores[r], scores[r] - scores[base]
        print(f"  {r[12:60]:<48} {s.mean():.4f} [{np.quantile(s, .025):.4f}, "
              f"{np.quantile(s, .975):.4f}] | {d.mean():+.4f} [{np.quantile(d, .025):+.4f}, "
              f"{np.quantile(d, .975):+.4f}] | {np.mean(d > 0):.2f}")


def resample(day_w: np.ndarray, day_hit: dict, n_days: int, rng, prob=None) -> dict:
    idx = rng.choice(len(day_w), size=(B, n_days), p=prob)
    wsum = day_w[idx].sum(1)
    return {r: hits[idx].sum(1) / wsum for r, hits in day_hit.items()}


def shift_weights(train_days: np.ndarray) -> np.ndarray:
    """p(test)/p(train) per training day from day-level volatility (covariate shift)."""
    from sklearn.linear_model import LogisticRegression
    x = pd.read_parquet(DATA / "prep/x_submit.parquet",
                        columns=["ID", "DATE_ID", "vol", "ret_day", "ret_1"])
    x = x.assign(is_test=x["ID"] >= TEST_FIRST_ID, ad=x.ret_day.abs(), a1=x.ret_1.abs())
    g = x.groupby("DATE_ID").agg(vol=("vol", "mean"), ad=("ad", "mean"), a1=("a1", "mean"),
                                 is_test=("is_test", "max"))
    feats = np.log(g[["vol", "ad", "a1"]])
    p = LogisticRegression().fit(feats, g.is_test).predict_proba(feats)[:, 1]
    ratio = pd.Series(p / (1 - p), g.index).loc[train_days].to_numpy()
    return ratio / ratio.sum()


def adversarial() -> None:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    x = pd.read_parquet(DATA / "prep/x_submit.parquet")
    is_test = (x["ID"] >= TEST_FIRST_ID).astype(int).to_numpy()
    feats = x.drop(columns=["ID", "DATE_ID", "STOCK_ID", "COUNTRY"])
    oof = np.zeros(len(x))
    for tr, va in GroupKFold(n_splits=3).split(feats, is_test, x["DATE_ID"]):
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_threads=4, verbose=-1)
        m.fit(feats.iloc[tr], is_test[tr])
        oof[va] = m.predict_proba(feats.iloc[va])[:, 1]
    print(f"\nadversarial AUC train vs test (grouped by day): {roc_auc_score(is_test, oof):.3f}")


def main(runs: list[str], base: str, adv: bool) -> None:
    y = pd.read_csv(DATA / "y_train.csv")
    keys = pd.read_parquet(DATA / "prep/x_train.parquet", columns=["ID", "DATE_ID"])
    df = keys.merge(y, on="ID")
    w, up = df["Close return"].abs().to_numpy(), df["Close return"].to_numpy() > 0
    days, day = np.unique(df["DATE_ID"], return_inverse=True)
    day_w = np.bincount(day, weights=w)
    day_hit = {}
    print("pooled OOF weighted accuracy:")
    for r in runs:
        hit = (load_oof(r, df["ID"].to_numpy()) > 0.5) == up
        day_hit[r] = np.bincount(day, weights=w * hit)
        print(f"  {r[12:60]:<48} {day_hit[r].sum() / day_w.sum():.5f}")
    rng = np.random.default_rng(42)
    report(runs, base, resample(day_w, day_hit, 392, rng), "392 days (CV sample)")
    report(runs, base, resample(day_w, day_hit, 199, rng), "199 days (private test size)")
    report(runs, base, resample(day_w, day_hit, 199, rng, shift_weights(days)),
           "199 TEST-LIKE days (reweighted to the calmer test period)")
    if adv:
        adversarial()


if __name__ == "__main__":
    args = sys.argv[1:]
    base = next((a.split("=", 1)[1] for a in args if a.startswith("--base=")), DEFAULT_BASE)
    runs = [a for a in args if not a.startswith("--")]
    main([base] + [r for r in runs if r != base], base, "--adv" in args)
