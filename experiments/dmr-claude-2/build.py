"""Kaggle files -> shared feature tables in <main checkout>/data/derived/ (parquet).

  uv run python experiments/dmr-claude-2/build.py --train data/raw/train.csv \
      --test data/raw/test.csv --id ID --day DAY --stock STOCK --country COUNTRY \
      --target TARGET [--y-train data/raw/y_train.csv] [--sample data/raw/sample_submission.csv]

Relative paths resolve against the main checkout. Writes X_train / y_train / X_test parquet
(+ the sample submission) so competition.toml can point at data/derived.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import intraday as I  # noqa: E402

from qrtc.config import SHARED_ROOT, resolve_shared  # noqa: E402
from qrtc.io import read_table  # noqa: E402


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    for name in ("train", "test", "id", "day", "stock", "country", "target"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--y-train", help="separate target file joined on --id")
    p.add_argument("--sample", help="sample submission to copy next to the tables")
    p.add_argument("--out", default="data/derived")
    p.add_argument("--extra", nargs="*", default=[], help="non-price columns to keep as-is")
    return p.parse_args()


def main() -> int:
    a = parse()
    train, test = read_table(resolve_shared(a.train)), read_table(resolve_shared(a.test))
    if a.y_train:
        train = train.merge(read_table(resolve_shared(a.y_train)), on=a.id, validate="1:1")
    keys = [a.id, a.day, a.stock, a.country, a.target, *a.extra]
    bars = I.detect_bar_cols(train, keys)
    missing = [c for c in bars if c not in test.columns]
    if missing:
        raise SystemExit(f"price columns missing in test: {missing[:5]}")
    print(f"train {train.shape} test {test.shape} bars={len(bars)} first={bars[:3]} "
          f"last={bars[-3:]}")
    X_tr, X_te = I.features_from_rows(train, test, bars, a.id, a.day, a.stock, a.country)
    if a.extra:
        X_tr = X_tr.merge(train[[a.id, *a.extra]], on=a.id, validate="1:1")
        X_te = X_te.merge(test[[a.id, *a.extra]], on=a.id, validate="1:1")
    y_tr = train[[a.id, a.target]].set_index(a.id).loc[X_tr[a.id]].reset_index()
    out = Path(a.out) if Path(a.out).is_absolute() else SHARED_ROOT / a.out  # shared data/
    out.mkdir(parents=True, exist_ok=True)
    X_tr.to_parquet(out / "X_train.parquet", index=False)
    y_tr.to_parquet(out / "y_train.parquet", index=False)
    X_te.to_parquet(out / "X_test.parquet", index=False)
    if a.sample:
        shutil.copy(resolve_shared(a.sample), out / "sample_submission.csv")
    print(f"wrote {out}: X_train {X_tr.shape}, X_test {X_te.shape}, features "
          f"{X_tr.shape[1] - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
