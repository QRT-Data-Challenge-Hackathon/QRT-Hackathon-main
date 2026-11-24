# Method and results

This is the technical write-up of our entry: the data, how we processed it, how we validated it,
what we tried, what worked, and how each CV number maps to a saved run in
[`results/registry/`](../results/registry).

## 1. Task

| | |
|---|---|
| Universe | 386 European stocks in 6 countries (GBR 136, FRA 76, DEU 75, ITA 40, NLD 33, ESP 26) |
| One row | one stock on one day: 5-minute prices from the open up to the cutoff (close minus 30 minutes) |
| Target | `Close return`: the stock's cutoff-to-close return minus the equal-weighted mean of its country that day |
| Metric | weighted accuracy of the predicted sign, weight = \|Close return\| |
| Split | 392 training days (144,225 rows), 199 test days (74,932 rows), all 386 stocks in both |
| Submission | `ID,Close return` with ±1 for every one of the 219,157 ids, train ids included |
| Time | 2 hours on Kaggle |

The target sums to exactly zero within each `DATE_ID x COUNTRY`, so the task is cross-sectional:
does this stock beat or trail its own country over the last half hour? Always-up and always-down
both score 0.5000. The host's benchmark, the negative of the last 5-minute return, scores 0.5227.

## 2. Data audit

Script: [`experiments/validation/audit.py`](../experiments/validation/audit.py).

- **Shuffled days.** `DATE_ID` is a random relabelling of the days. Price levels of consecutive
  ids are no closer than random pairs (median absolute log gap 1.22 vs 1.19), so calendar order
  cannot be recovered.
- **Rescaled price levels.** Prices are rescaled per row. Returns within a day are valid. Levels
  across days are not, so overnight gaps and multi-day rolling features carry no information.
- **Short days.** Most rows hold 96 bars; a few half days hold 48.
- **Distribution shift.** A day-grouped adversarial classifier separates train from test rows
  with AUC 0.657 (0.766 with row-level folds, which leak days). Volatility drives it: the median
  per-day volatility is 0.00119 on test days against 0.00140 on train days, and train has a
  high-volatility tail the test lacks. A day-level classifier on per-day aggregates, including
  stocks per day (test at least 243, train down to 128), reaches AUC 0.84: the test period is
  later, calmer and has a wider universe.
- **No leaks.** The leakage checks in `qrtc eda` (id order, recoverable date order, target-derived
  columns, overlapping lags) came back clean.

## 3. Processing pipeline

Feature builder: [`experiments/intraday_ridge/intraday.py`](../experiments/intraday_ridge/intraday.py),
driven by [`build.py`](../experiments/intraday_ridge/build.py) (`make features`).

1. **Flatten.** The `Price TS` lists become a 96-column matrix right-aligned at the cutoff, so
   column 95 is always the last observed bar and half days are padded on the left.
2. **Clean.** Log prices, forward-filled along the day (a missing bar means no trade).
3. **Per-row features** from the visible bars only: day return, return over the first 6 bars,
   returns over the last 1, 3, 6, 12 and 24 bars, volatility (whole day and last hour), range,
   position in the day's range, bar count, opening hour.
4. **Country-relative features.** Each return minus its `DATE_ID x COUNTRY` mean (the target's
   own reference), the country's mean move, and the within-country rank and clipped z-score.
5. **Intraday beta.** Per row, the beta and correlation of the stock's 5-minute returns to its
   country's equal-weighted path, the residual (idiosyncratic) volatility and the residual return
   over the last hour.
6. **Beta x country move.** `beta * country_return` over the last hour and over the day. These
   two columns let a linear model split a stock's move into the part its beta explains and its
   own part.
7. **Tables.** `X_train`, `y_train`, `X_test` and `X_submit` (train + test ids, in the order
   Kaggle wants) as parquet under `data/derived/`. 45 features.

Train and test rows go through the same functions, and every feature uses only the row's own day,
so nothing crosses the train/test boundary.

## 4. Validation

- **Folds of whole days.** `cv = "date_blocks"` with 5 folds and seed 42
  ([`src/qrtc/cv.py`](../src/qrtc/cv.py)). Rows of one day share the market and country move,
  so splitting a day across folds would leak it. Because the ids are shuffled, the blocks are
  random sets of days.
- **The metric itself.** Every fold is scored with |y|-weighted accuracy on the sign. Regression
  models are mapped to a pseudo-probability `sigmoid(pred / std(pred))`, so they can be scored
  and blended like classifiers.
- **One spec for every model.** All runs read the same `competition.toml`, the same splits and
  the same metric, so CV numbers are comparable and every run's out-of-fold predictions can be
  blended.
- **The public leaderboard scored the training rows.** The LightGBM baseline scored 0.6631 on
  the public leaderboard and exactly 0.6631 in-sample on the training ids, against 0.5329 in CV.
  The host benchmark matched the same way (0.52273 on both). So the public leaderboard measured
  in-sample fit and rewarded memorisation. We ignored it and selected every submission on CV.

## 5. What worked

### 5.1 Ridge on the real return

[`experiments/intraday_ridge/run.py`](../experiments/intraday_ridge/run.py). Ridge regression
on the raw `Close return` (not its sign), with standardisation fitted inside each fold through a
scikit-learn pipeline:

| run | model | CV (mean ± std over 5 folds) |
|---|---|---|
| `1124-184854_baseline-lgbm` | LightGBM, flat baseline features | 0.5329 ± 0.0058 |
| `1124-184705_lgbm-shallow-base-w` | shallow LightGBM, intraday features, \|y\| weights | 0.5372 ± 0.0079 |
| `1124-184652_logreg-base-w` | logistic regression, \|y\| weights | 0.5397 ± 0.0067 |
| `1124-185327_ridge-v1-reg` | **ridge on the real return** | **0.5420 ± 0.0052** |
| `1124-190543_ridge-v1-nobetax-reg` | the same without the beta x country terms | 0.5364 ± 0.0065 |

The signal is **short-horizon idiosyncratic reversal**: a stock that ran ahead of its country
late in the session tends to give some of it back by the close. Only the stock's own move
reverts, not the part its beta explains, which is why removing the two beta x country columns
costs 0.55 points. Linear models beat trees on these features: with a signal this weak and this
close to linear, the trees mostly fit noise.

`make ridge` reproduces the ridge run exactly from this repository (0.54196 ± 0.00521, identical
fold scores).

### 5.2 Residual-path features, recovered sectors, LightGBM

[`experiments/residual_gbm/`](../experiments/residual_gbm). A second feature set built directly
from the price path, in families that can be switched on one at a time:

- `beta`: intraday beta to the equal-weighted country path, residual returns over windows of
  1 to 48 bars;
- `idio`: the cumulative residual path: t-statistics, quarter-day segments, drawdown and run-up,
  position in its range, within-country ranks;
- `shape`: trend slope and R², distance to the day's high and low, autocorrelation;
- **sectors**: the data has no sector column, so we recovered one. Stocks are clustered with
  KMeans (K = 25) on the correlation of their country-residual 5-minute return paths over all
  days (features only, never the target). Each signal is then split into a stock-vs-sector and a
  sector-vs-country part.

Shallow LightGBM with |y| sample weights, averaged over 3 bagging seeds (a single seed moves the
CV by about 0.0015):

| features | CV (3-seed average) |
|---|---|
| base spec only | 0.5388 ± 0.0072 |
| + beta | 0.5407 ± 0.0075 |
| + beta, idio | 0.5416 ± 0.0037 |
| + beta, idio, shape | 0.5422 ± 0.0052 |
| **+ beta, idio, sectors K = 25** (`1124-190809_a-s-beta-idio-lgs-fw-k25-x3`) | **0.5436 ± 0.0056** |

On the same features, logistic regression scores 0.5436 ± 0.0034 and ridge 0.5422 ± 0.0037.

### 5.3 Blend

Equal-weight rank average of three models with different inductive biases: the sector LightGBM,
the sector logistic regression and the intraday ridge (`1124-192031_blend-rank`, out-of-fold
0.5447). The ridge and the sector LightGBM agree on only about 82% of predicted signs, which is
where the blend gain comes from. This was our best private score.

## 6. Overfitting checks

With an edge of about 1 accuracy point and a test set of 199 days, choosing between candidates
is itself a source of overfitting. We ran three checks.

1. **Day bootstrap** ([`robust.py`](../experiments/intraday_ridge/robust.py)): 2,000
   resamples of whole days, paired between runs. The 90% interval of a single run's score is
   about ±0.44 points, wider than any gap between our top four candidates.
2. **Fresh fold partitions** ([`permute_days.py`](../experiments/intraday_ridge/permute_days.py)):
   the same tables with `DATE_ID` relabelled by three random bijections, so the folds become
   three new partitions of the same days. Tweaks chosen on the official folds have to survive
   them:

   | model | official | p1 | p2 | p3 | mean |
   |---|---|---|---|---|---|
   | ridge, untuned | 0.5420 | 0.5428 | 0.5429 | 0.5425 | 0.5425 |
   | ridge, target clipped at 2% | 0.5427 | 0.5425 | 0.5426 | 0.5415 | 0.5422 |
   | ridge, clipped + logits demeaned per country-day | 0.5427 | 0.5424 | 0.5424 | 0.5427 | 0.5426 |
   | shallow LightGBM, \|y\| weights | 0.5372 | 0.5368 | 0.5362 | 0.5363 | 0.5366 |

   The two tweaks only helped on the folds they were picked on. Ridge over LightGBM held on
   every partition.
3. **Test-like days** ([`testlike.py`](../experiments/intraday_ridge/testlike.py),
   [`validate_candidates.py`](../experiments/validation/validate_candidates.py)): an
   adversarial classifier gives each training day its probability of looking like a test day,
   and every candidate is re-scored on the most test-like 30% of days and with importance
   weights. The model ranking held in the calm regime.

## 7. Submissions

Submissions were chosen on CV, never on the public score. Private scores were revealed after the
deadline.

| submission | CV / OOF | public (= train rows) | private |
|---|---|---|---|
| LightGBM baseline | 0.5329 | 0.6631 | 0.5337 |
| rank blend: ridge + residual LightGBM | 0.5440 | 0.5580 | 0.5395 |
| ridge, clipped target | 0.5427 | 0.5462 | 0.5377 |
| residual logistic regression, sectors | 0.5436 | 0.5486 | 0.5413 |
| residual LightGBM, sectors | 0.5441 | 0.5748 | 0.5417 |
| residual LightGBM, sectors, 3 seeds | 0.5438 | 0.5752 | 0.5419 |
| XGBoost GPU notebook, tuned | 0.5405 (before tuning) | 0.5752 | 0.5426 |
| group-rank blend of 5 models | 0.5453 | 0.5453 | 0.5425 |
| rank blend of 3 models, final | 0.5449 | 0.5577 | 0.5430 |
| **rank blend of 3 models (5.3)** | **0.5447** | 0.5571 | **0.5436** |

CV picked the right families. The three-model blends finished on top. The baseline finished last.
The single ridge was ranked below the gradient-boosting models by CV and finished below them too.
The gaps on the test set are within the noise of 199 days: the ridge lost 0.5 points against
its CV, and the two-model blend lost 0.45. The public leaderboard ranked the baseline first.
Final standing: **8th of 37 teams**, 0.5436 against 0.5456 for the winner.

## 8. What did not help

All on the official folds, one change against the run it modifies:

| idea | run | CV |
|---|---|---|
| rank-gauss inputs per day | `1124-190351_ridge-rg-reg` | 0.5381 |
| per-bar relative returns + residual z-scores (v2) | `1124-185233_ridge-v2-reg` | 0.5395 |
| stock-average beta profile (v3) | `1124-190525_ridge-v3-clip-reg` | 0.5370 |
| per-country slopes (v4) | `1124-190714_ridge-v4-cslopes-reg` | 0.5408 |
| reversal x correlation/beta/vol interactions (v5) | `1124-191053_ridge-v5-clip-reg` | 0.5414 |
| spline basis on the strongest signals (v6) | `1124-191228_ridge-v6-spline-clip-reg` | 0.5414 |
| distance to the day's average price (v7) | `1124-191311_ridge-v7-devmean-clip-reg` | 0.5416 |
| KMeans sectors in the ridge (v8) | `1124-191539_ridge-v8-sectors-clip-reg` | 0.5416 |
| up-weighting test-like days | `1124-191017_ridge-v1-clip-tlw-reg-w` | 0.5391 |
| stock id as a categorical | `1124-184813_lgbm-shallow-stockcat-w` | 0.5309 |
| 1D CNN over the return path + features | `1124-192742_cnn-e4` | 0.5353 |
| MLP on the features | `1124-192758_mlp-e4` | 0.5379 |
| GRU over the return path, 3 seeds | `1124-194224_gru-e3-s3` | 0.5372 |
| supervised autoencoder + MLP (Jane Street Market Prediction 1st-place architecture) | `1124-193445_sae-mlp-reg-s1` | 0.5327 |
| 1D CNN on the scale-free relative path | `1124-193353_cnn-path` | 0.5352 |
| ExtraTrees on the residual features | `1124-200252_et-sectors` | 0.5403 |

The neural networks were trained on the Apple-Silicon GPU (PyTorch, MPS) inside the same folds.
The path CNN's out-of-fold predictions were less correlated with the ridge and the sector
LightGBM (0.71 to 0.74, against 0.81 between those two), but it was too weak to lift a blend.

## 9. Engineering notes

- **The toolkit came first.** `src/qrtc` was written and tested before the event on five
  synthetic datasets shaped like past QRT challenges (panel sign prediction, sign-scored
  regression, rank correlation, multiclass, survival). At the event, only `competition.toml` and
  the feature code were new.
- **Every number is a saved run.** `tracking.save_run` stores out-of-fold and test predictions
  under `runs/` and the metadata in `results/registry/`, so any two runs can be compared fold by
  fold or blended.
- **The submission validator refuses to write a malformed file.** It checks the id set, the row
  count, NaNs and the value format: ±1, never 0.
- **A bug found under time pressure.** scikit-learn pipelines do not accept `sample_weight`
  directly. `Model.fit` now routes it to the last step (`<step>__sample_weight`), with a test.
- **torch and LightGBM never share a process.** On macOS their OpenMP runtimes crash in either
  order, so the foundation-model wrappers run in a worker process and the torch scripts never
  import LightGBM.
