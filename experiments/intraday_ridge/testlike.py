"""Score saved runs on the most test-like training days (day-level adversarial validation).

Test days are a different regime (calmer, more correlated, more stocks per day: day-level
train-vs-test AUC 0.84), so plain date-block CV mixes regimes. This re-scores each run's OOF
predictions on the train days a classifier finds most test-like.

  uv run python experiments/intraday_ridge/testlike.py RUN [RUN ...] [--top 0.3]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qrtc import config, tracking

DERIVED = config.SHARED_ROOT / "data" / "derived"


def day_table(x: pd.DataFrame) -> pd.DataFrame:
    g = x.groupby("DATE_ID")
    return pd.DataFrame({
        "n": g.size(), "vol": g.vol.median(), "range": g.range.median(),
        "open_hour": g.open_hour.mean(), "disp_rday": g.r_day_rel.std(),
        "mkt_absday": g["r_day_DATE_ID_COUNTRY_mean"].apply(lambda s: s.abs().mean()),
        "beta_mean": g.beta_day.median(), "corr_mean": g.corr_day.median()}).fillna(0)


def testlike_prob(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Out-of-fold P(test) for every train day."""
    days = pd.concat([day_table(train).assign(t=0), day_table(test).assign(t=1)])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(clf, days.drop(columns="t"), days.t, method="predict_proba",
                          cv=StratifiedKFold(5, shuffle=True, random_state=0))[:, 1]
    return pd.Series(p, index=days.index)[days.t.to_numpy() == 0]


def weighted_acc(y: np.ndarray, p: np.ndarray) -> float:
    ok = ~np.isnan(p)
    w = np.abs(y[ok])
    return float((w * ((p[ok] > 0.5) == (y[ok] > 0))).sum() / w.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--top", type=float, default=0.3, help="share of most test-like days")
    a = ap.parse_args()
    train = pd.read_parquet(DERIVED / "X_train.parquet")
    test = pd.read_parquet(DERIVED / "X_test.parquet")
    y = pd.read_parquet(DERIVED / "y_train.parquet").set_index("ID")["Close return"]
    p_day = testlike_prob(train, test)
    like = set(p_day[p_day >= p_day.quantile(1 - a.top)].index)
    print(f"test-like days: {len(like)} of {len(p_day)} (top {a.top:.0%})")
    rows = []
    for run in a.runs:
        preds = tracking.load_preds(run)
        frame = pd.DataFrame({"ID": preds["train_ids"], "p": preds["oof"]})
        frame = frame.merge(train[["ID", "DATE_ID"]], on="ID").assign(y=lambda f: f.ID.map(y))
        m = frame.DATE_ID.isin(like).to_numpy()
        yy, pp = frame.y.to_numpy(), frame.p.to_numpy(dtype=float)
        rows.append((run, weighted_acc(yy, pp), weighted_acc(yy[m], pp[m]),
                     weighted_acc(yy[~m], pp[~m])))
    out = pd.DataFrame(rows, columns=["run", "all", "test_like", "other"])
    print(out.round(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
