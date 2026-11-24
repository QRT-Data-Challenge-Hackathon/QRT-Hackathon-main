"""Prepare the QRT x ETH 24/11/2026 Kaggle data ("predict the last 30 minutes") for qrtc.

Raw rows hold a variable-length 5-minute price array (``Price TS``, last element = cutoff).
This script writes flat numeric tables that competition.toml points at:

  prep/x_train.parquet   train rows: ids, keys and features
  prep/x_submit.parquet  ALL rows Kaggle wants in a submission (train + test ids, 219,157)
  prep/benchmark.csv     the host's benchmark, -(last 5-minute return), for every id

Kaggle scores only the sign; an exact 0 is always wrong, so the benchmark replaces zeros.
Usage: uv run python scripts/prep_last30.py [data_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qrtc import config

DEFAULT_DIR = config.SHARED_ROOT / "data" / "raw" / "qrt-x-eth-hackathon-24-11-2026"
KEYS = ["ID", "DATE_ID", "STOCK_ID", "COUNTRY"]
LAGS = (1, 2, 3, 6, 12, 24, 48)          # returns over the last k bars (5 min each)
RELATIVE = ["ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "ret_24", "ret_48", "ret_day",
            "ret_first6", "vol", "vol_last12", "bar1"]
EPS = 1e-9


def price_matrix(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Right-aligned price matrix (last column = cutoff), forward-filled; and bar counts."""
    arrays = [np.asarray(a, dtype=float) for a in series]
    lengths = np.array([len(a) for a in arrays])
    width = lengths.max()
    out = np.full((len(arrays), width), np.nan)
    for row, (arr, n) in enumerate(zip(arrays, lengths)):
        out[row, width - n:] = arr
    filled = pd.DataFrame(out).ffill(axis=1).to_numpy()
    pad = np.arange(width)[None, :] < (width - lengths)[:, None]
    filled[pad] = np.nan
    return filled, lengths


def row_features(frame: pd.DataFrame) -> pd.DataFrame:
    prices, lengths = price_matrix(frame["Price TS"])
    width = prices.shape[1]
    first_idx = width - lengths
    rows = np.arange(len(frame))
    first = prices[rows, first_idx]
    feats = {"n_bars": lengths,
             "n_missing": frame["Price TS"].map(lambda a: int(pd.isna(np.asarray(a)).sum())),
             "ret_day": prices[:, -1] / first - 1,
             "ret_first6": prices[rows, np.minimum(first_idx + 6, width - 1)] / first - 1}
    for k in LAGS:
        feats[f"ret_{k}"] = prices[:, -1] / prices[:, -1 - k] - 1
    bars = np.diff(np.log(prices), axis=1)
    feats["bar1"] = bars[:, -2]                    # the 5-minute return before the last one
    feats["vol"] = np.nanstd(bars, axis=1)
    feats["vol_last12"] = np.nanstd(bars[:, -12:], axis=1)
    feats["zero_bars"] = np.sum(bars == 0, axis=1)
    feats["open_min"] = _minutes(frame["OPEN_TIME"])
    feats["close_min"] = _minutes(frame["CLOSE_TIME"])
    return pd.DataFrame(feats, index=frame.index)


def _minutes(times: pd.Series) -> np.ndarray:
    parts = times.str.split(":", expand=True).astype(float)
    return (parts[0] * 60 + parts[1]).to_numpy()


def add_relative(feats: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Each feature minus its mean over the same DATE_ID and COUNTRY (the target is demeaned so)."""
    grouped = pd.concat([keys[["DATE_ID", "COUNTRY"]], feats[RELATIVE]], axis=1)
    means = grouped.groupby(["DATE_ID", "COUNTRY"])[RELATIVE].transform("mean")
    rel = (feats[RELATIVE] - means).add_suffix("_rel")
    ranks = grouped.groupby(["DATE_ID", "COUNTRY"])[RELATIVE].rank(pct=True).add_suffix("_rank")
    return pd.concat([feats, rel, ranks], axis=1)


def benchmark(feats: pd.DataFrame) -> np.ndarray:
    """Host benchmark -(last 5-min return); zeros/NaN fall back to older moves, then +EPS."""
    pred = -feats["ret_1"].to_numpy()
    for fallback in (-feats["ret_2"].to_numpy(), -feats["ret_day"].to_numpy()):
        bad = ~np.isfinite(pred) | (pred == 0)
        pred = np.where(bad, fallback, pred)
    bad = ~np.isfinite(pred) | (pred == 0)
    return np.where(bad, EPS, pred)


def build(raw: pd.DataFrame) -> pd.DataFrame:
    feats = add_relative(row_features(raw), raw)
    return pd.concat([raw[KEYS].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)


def main(data_dir: Path) -> None:
    train = pd.read_parquet(data_dir / "x_train.parquet")
    test = pd.read_parquet(data_dir / "x_test.parquet")
    both = pd.concat([train, test], ignore_index=True)
    table = build(both)                    # relative features need whole days: build once
    out = data_dir / "prep"
    out.mkdir(exist_ok=True)
    is_train = table["ID"].isin(train["ID"])
    table[is_train].to_parquet(out / "x_train.parquet", index=False)
    table.to_parquet(out / "x_submit.parquet", index=False)
    bench = pd.DataFrame({"ID": table["ID"], "Close return": benchmark(table)})
    bench.sort_values("ID").to_csv(out / "benchmark.csv", index=False)
    print(f"train {is_train.sum()} rows, submit {len(table)} rows, {table.shape[1]} columns -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR)
