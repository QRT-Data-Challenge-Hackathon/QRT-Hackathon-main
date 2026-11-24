"""Kaggle files -> shared feature tables in <main checkout>/data/derived/ (parquet).

  uv run python experiments/dmr-claude-2/build.py            # defaults match the event files
  uv run python experiments/dmr-claude-2/build.py --history  # + rolling per-stock features

Input (data/raw): x_train.parquet / x_test.parquet with ID, DATE_ID, STOCK_ID, COUNTRY,
OPEN_TIME, CLOSE_TIME and "Price TS" (list of 5-min prices up to the cutoff = close - 30 min);
y_train.csv with ID, "Close return" (idiosyncratic return cutoff -> close, de-meaned per
date x country). Output: X_train / y_train / X_test parquet, one row per ID.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import intraday as I  # noqa: E402

from qrtc.config import SHARED_ROOT  # noqa: E402

KEYS = ["ID", "DATE_ID", "STOCK_ID", "COUNTRY"]
WIDTH = 96


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/derived")
    p.add_argument("--history", action="store_true", help="add rolling per-stock features")
    return p.parse_args()


def load(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(raw / "x_train.parquet")
    test = pd.read_parquet(raw / "x_test.parquet")
    y = pd.read_csv(raw / "y_train.csv")
    return train, test, y


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    prices = I.stack_prices(df["Price TS"], WIDTH)
    meta = df[KEYS].assign(open_hour=df["OPEN_TIME"].str[:2].astype(int))
    return pd.concat([meta, prices], axis=1).reset_index(drop=True)


def main() -> int:
    a = parse()
    start = time.time()
    raw = SHARED_ROOT / a.raw
    train, test, y = load(raw)
    tr, te = flatten(train), flatten(test)
    bars = [c for c in tr.columns if c.startswith("P")]
    X_tr, X_te = I.features_from_rows(tr, te, bars, *KEYS, history=a.history)
    for X, src in ((X_tr, tr), (X_te, te)):
        X["open_hour"] = X["ID"].map(src.set_index("ID")["open_hour"])
    y_tr = y.set_index("ID").loc[X_tr["ID"]].reset_index()
    out = Path(a.out) if Path(a.out).is_absolute() else SHARED_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    X_tr.to_parquet(out / "X_train.parquet", index=False)
    y_tr.to_parquet(out / "y_train.parquet", index=False)
    X_te.to_parquet(out / "X_test.parquet", index=False)
    print(f"wrote {out}: X_train {X_tr.shape}, X_test {X_te.shape} in {time.time() - start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
