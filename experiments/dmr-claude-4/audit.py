"""Validator data audit for the QRT x ETH 24/11/2026 Kaggle task (last 30 minutes).

Usage: uv run python experiments/dmr-claude-4/audit.py
Reports split structure (raw test only), trivial scores, adversarial validation, leakage checks
and whether the shuffled DATE_ID order is recoverable from price levels. Report-only: nothing
here is used as a feature.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from qrtc import config, eda, io, metrics

RAW = config.SHARED_ROOT / "data" / "raw" / "qrt-x-eth-hackathon-24-11-2026"


def raw_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = pd.read_parquet(RAW / "x_train.parquet")
    te = pd.read_parquet(RAW / "x_test.parquet")
    y = pd.read_csv(RAW / "y_train.csv")
    return tr, te, y


def split_report(tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    return {
        "train_rows": len(tr), "test_rows": len(te),
        "train_days": tr.DATE_ID.nunique(), "test_days": te.DATE_ID.nunique(),
        "shared_days": len(set(tr.DATE_ID) & set(te.DATE_ID)),
        "train_stocks": tr.STOCK_ID.nunique(), "test_stocks": te.STOCK_ID.nunique(),
        "test_stocks_in_train": float(te.STOCK_ID.isin(tr.STOCK_ID).mean()),
        "date_id_union": sorted(set(tr.DATE_ID) | set(te.DATE_ID)) == list(range(591)),
        "train_date_mean": float(tr.DATE_ID.mean()), "test_date_mean": float(te.DATE_ID.mean()),
    }


def trivial_scores(ds: io.Dataset) -> dict:
    r = ds.y_raw.iloc[:, 0].to_numpy()
    w = np.abs(r)
    wa = metrics.get_metric("weighted_accuracy")
    yb = (r > 0).astype(int)
    last5 = ds.train["ret_1"].fillna(0).to_numpy()
    last5_rel = ds.train["ret_1_rel"].to_numpy()
    return {
        "pos_share": float(yb.mean()),
        "weighted_pos_share": float(w[yb == 1].sum() / w.sum()),
        "always_up": wa(yb, np.ones_like(yb), weight=w),
        "always_down": wa(yb, np.zeros_like(yb), weight=w),
        "benchmark_-ret_1": wa(yb, (-last5 > 0).astype(int), weight=w),
        "reversal_-ret_1_rel": wa(yb, (-last5_rel > 0).astype(int), weight=w),
    }


def date_order_signal(tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    """If DATE_ID were time order, consecutive ids of a stock would have close price levels."""
    df = pd.concat([tr, te])[["DATE_ID", "STOCK_ID", "Price TS"]].copy()
    df["first"] = df["Price TS"].map(lambda a: float(a[0]))
    df["last"] = df["Price TS"].map(lambda a: float(a[-1]))
    df = df.sort_values(["STOCK_ID", "DATE_ID"])
    nxt = df.groupby("STOCK_ID")["first"].shift(-1)
    gap_by_id = np.log(nxt / df["last"]).abs().median()
    rng = np.random.default_rng(42)
    shuffled = df.groupby("STOCK_ID")["first"].transform(lambda s: rng.permutation(s.to_numpy()))
    gap_random = np.log(shuffled / df["last"]).abs().median()
    # nearest-neighbour chaining: for each (stock, day) the best-matching next-day open
    return {"median_|gap|_consecutive_ids": float(gap_by_id),
            "median_|gap|_random_pairing": float(gap_random),
            "ids_look_time_ordered": bool(gap_by_id < 0.5 * gap_random)}


def main() -> None:
    spec = config.load_spec()
    tr, te, _ = raw_frames()
    ds = io.load_dataset(spec)
    out: dict = {"split": split_report(tr, te), "trivial": trivial_scores(ds),
                 "date_order": date_order_signal(tr, te)}
    test_ids = set(te.ID)
    test_feats = ds.test[ds.test["ID"].isin(test_ids)]
    feats = [c for c in ds.train.columns if c not in ("ID", "DATE_ID", "STOCK_ID", "COUNTRY")]
    out["adversarial"] = eda.adversarial_validation(ds.train, test_feats, feats)
    out["leakage_checks"] = eda.leakage_checks(ds.train, ds.y.iloc[:, 0].to_numpy(), spec)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
