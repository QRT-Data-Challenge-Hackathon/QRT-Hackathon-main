# data/

Gitignored except this file. The competition data is not redistributed here.

```
data/raw/qrt-x-eth-hackathon-24-11-2026/        the unzipped Kaggle files: x_train.parquet,
                                                x_test.parquet, y_train.csv
data/raw/qrt-x-eth-hackathon-24-11-2026/prep/   flat baseline features (make prep)
data/derived/                                   intraday feature tables (make features)
```

Relative data paths resolve against the main checkout (`qrtc.config.SHARED_ROOT`), so git
worktrees of the repo read the same files.
