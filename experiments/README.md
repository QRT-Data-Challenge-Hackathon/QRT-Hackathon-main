# experiments/

Competition-specific research code. Every script reads the spec through `qrtc.config`, runs the
same `date_blocks` folds and saves its run with `tracking.save_run`, so all numbers are
comparable and every saved run can be blended with `qrtc blend`.

| folder | what it does |
|---|---|
| `intraday_ridge/` | feature builder for the 5-minute price panel (`intraday.py`, `build.py`), the ridge model, neural nets (CNN/GRU/MLP, supervised autoencoder), robustness checks (day bootstrap, fresh fold partitions, test-like days) and the negative-result feature variants |
| `residual_gbm/` | residual-path features (intraday beta, cumulative idiosyncratic path, shape), correlation-clustered sectors, LightGBM runner with seed bagging, ridge/ExtraTrees on the same features |
| `cnn_path/` | 1D CNN on the scale-free relative price path and its blends |
| `validation/` | data audit, shift-aware test-score estimate for candidate runs, paired gain gate |
| `xgboost_gpu/` | XGBoost notebooks for Colab/Kaggle GPUs (metric-matched targets and weights, seed bagging) |

Notebooks are committed without outputs (`make nbstrip`).
