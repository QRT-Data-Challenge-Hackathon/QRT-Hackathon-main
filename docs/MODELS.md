# Pretrained models for the event

Research from 23 Nov 2026 on which open pretrained models are worth downloading before the
hackathon, and how they are wired into the toolkit.

## Short version

- **Use `--model tabicl` (TabICLv2) early, next to the LightGBM baseline.** Pretrained tabular
  foundation models need no tuning, and on small and medium tables (up to ~50k rows, the size
  of most QRT challenges) they usually beat tuned GBMs. Their errors differ from GBM errors,
  so they also help blends (`qrtc blend <lgbm_run> <tabicl_run>`).
- **Also available:** `--model tabpfn` (TabPFN v2) and `--model tabdpt` (TabDPT 1.3). All three
  run offline on the M4 GPU and plug into `run_cv`, `save_run`, `blend` and `fit_full`.
- **Install and download before the event:** `make models` on every laptop. It installs the
  `fm` extra (torch, about 1.3 GB), downloads the weights (about 0.6 GB) and checks they run
  with `HF_HUB_OFFLINE=1`. `make setup` and `scripts/agent.sh start` include the extra
  (`QRTC_NO_FM=1` skips it).
- **Ask at the briefing** whether pretrained models are allowed (PLAYBOOK §3, question 10).
  TabICL and TabPFN were pretrained only on synthetic tables, so they bring no external data.
  TabDPT was pretrained on real public tables (OpenML), which a strict "no external data" rule
  may forbid.
- **Decide by CV, as always.** QRT write-ups found that deep nets did not pay on noisy
  return-sign panels; in-context models are a different kind of model (no training, built-in
  regularisation), but their gain there is unproven until our CV shows it.
- **No pretrained autoencoder helps here.** An anonymised table's columns mean nothing to a
  model pretrained elsewhere. The in-context models above are the useful kind of pretraining:
  they were trained on millions of tables to learn *how* to predict from rows. Autoencoders that
  win competitions are trained per dataset (see below).

## How to use

```bash
uv run qrtc baseline --model tabicl                       # CV + run + submission, as usual
uv run qrtc baseline --model tabpfn --params '{"n_estimators": 4}'
uv run qrtc baseline --model tabicl --params '{"max_rows": 20000, "n_bags": 3}'
uv run qrtc blend <lgbm_run> <tabicl_run> --method hill
```

| Param | Meaning | Default |
|---|---|---|
| `max_rows` | context cap; larger training folds are subsampled (stratified for classification) | tabicl 50k, tabpfn 10k, tabdpt 50k |
| `n_bags` | when capped, average this many models, each on its own subsample | 1 |
| `version` | TabPFN weights: `v2`, `v2_5`, `v2_6`, `v3`, `v3_5`, `v3_5_fast` | `v2` |
| anything else | passed to the backend (`n_estimators`, `checkpoint_version`, `device`, ...) | |

Tasks: binary, multiclass and regression (not survival). The models ignore `sample_weight`,
with a warning, and report no feature importance.

**Cost.** `fit` only stores the training rows; each predict call sends them with the query rows
to the model, so time grows with training rows × predicted rows. On large panels (QRT 2020 had
418k rows) cap the context (`max_rows` 10–20k, `n_bags` 2–3) or skip these models.

**Worker process.** LightGBM and torch ship different OpenMP runtimes, and on macOS loading
both into one process segfaults in **either** order (tested on this M4). qrtc therefore runs
every foundation-model call in one long-lived child process (`python -m qrtc.fm_worker`),
started on first use, and keeps LightGBM in the main process. In your own scripts and
notebooks, use the models through qrtc (`make_model`, `run_cv`) and never `import torch` next to
`lightgbm`. The worker runs with `nice 5` and `OMP_NUM_THREADS = cores − 2` to keep the
laptop responsive (`QRTC_FM_NICE` and `OMP_NUM_THREADS` override). The GPU is still busy while
it predicts, so the screen can stutter: run one foundation-model job at a time per laptop.

## What we measured (M4, 24 GB, rehearsal data, 5-fold CV)

Synthetic rehearsal data only, so treat it as a smoke test of the plumbing and of relative
cost, not as proof of the gain on the real problem.

RESULTS_TABLE

## The shortlist (checked on Hugging Face and PyPI, 23 Nov 2026)

| Model | In toolkit | Weights licence | Pretraining data | Size | Sweet spot |
|---|---|---|---|---|---|
| **TabICLv2** (Inria, ICML 2026), `tabicl` 2.2 | `tabicl` | BSD-3-Clause | synthetic | 2 × 110 MB | 300–100k rows, up to 2k features |
| **TabPFN v2** (Prior Labs, Nature 2025), `tabpfn` 10.2 | `tabpfn` | Prior Labs License (Apache-2.0 + "built with TabPFN") | synthetic | ~30–50 MB | ≤ 10k rows |
| **TabDPT 1.3** (Layer 6), `tabdpt` 1.3 | `tabdpt` | Apache-2.0 | real (OpenML) | 252 MB | any size (subsampled context), fastest here |
| TabPFN-3.5 / 3.5-Fast (Sep 2026) | `--params '{"version": "v3_5"}'` | **non-commercial**, needs account + licence | synthetic | 876 / 334 MB | Prior Labs' strongest; check event rules |
| TabPFN 2.5 / 2.6 / 3 | `version` param | non-commercial | synthetic | 43–233 MB | |
| Mitra 2 (Amazon) | no (AutoGluon only) | Apache-2.0 | synthetic | 303 MB | via AutoGluon 1.6 presets |
| LimiX-2 | no | non-commercial | | 1.6 GB | too big for the gain |
| TabM, RealMLP (`pytabkit`) | no | Apache-2.0 / MIT | none (train from scratch) | | top of TabArena when tuned and ensembled; tuning costs time we do not have |

Benchmarks: on TabArena (the living benchmark from the AutoGluon team), tabular foundation
models lead on small and medium datasets; TabICLv2's authors report it beats tuned
XGBoost/CatBoost/LightGBM on ~80% of TabArena datasets without tuning. Ensembles across model
families beat any single model, which is why the toolkit blends them with GBMs instead of
replacing GBMs.

**TabPFN-3.5 licence.** The weights may be used for research and evaluation, but "the model,
its derivatives, and its outputs cannot be used for any commercial or production purpose".
A student contest whose predictions go to a hedge fund is a grey zone: ask the organisers
first, otherwise stay with TabICLv2 / TabPFN v2 / TabDPT. To enable it, a human runs
`uv run python scripts/prefetch_models.py --tabpfn3` in their own terminal and accepts the
licence in the browser (the API key it creates is a credential: agents never handle it).

## Time-series foundation models (not wired in)

Useful only if the event gives raw series to forecast (load, prices) instead of a feature
table. On GIFT-Eval, Chronos-2 (Amazon, Apache-2.0, 478 MB) leads, with TimesFM-2.5 (Apache-2.0),
TiRex-2 (Apache-2.0) and Toto-2.0 (Apache-2.0) close behind; TimesFM-3.0 is non-commercial and
Moirai-2.0 is CC-BY-NC. On daily stock returns, off-the-shelf time-series foundation models
perform poorly (arXiv 2511.18578), so they are not a shortcut for return-sign tasks. `amazon/chronos-2`
and `NeoQuasar/Kronos-base` (a foundation model for OHLCV candles, MIT) are already in this
laptop's Hugging Face cache from other projects; install `chronos-forecasting` only if needed.

## Autoencoders

What worked in finance competitions was not a downloaded autoencoder but one trained on the
competition's own features:

- Jane Street Market Prediction (Kaggle 2021) top solutions used a **supervised autoencoder +
  MLP**: the encoder's output is concatenated to the raw features, reconstruction and target
  losses are trained together, and the autoencoder is fit **inside each CV fold** (fitting it
  on all rows first leaks the target).
- Porto Seguro (Kaggle 2017) was won with denoising autoencoder features (swap noise).

On a 4-hour clock that is a lane for one agent at most, after the GBM + in-context baseline.
Rules: fit it in-fold, report CV vs the baseline run, and treat a large gain as a leak first.

## Sources

- TabPFN model cards and README: huggingface.co/Prior-Labs/tabpfn_3_5, github.com/PriorLabs/TabPFN
- TabICLv2: arXiv 2602.11139 (ICML 2026), github.com/soda-inria/tabicl
- TabDPT: arXiv 2410.18164, huggingface.co/Layer6/TabDPT
- TabArena: arXiv 2506.16791, tabarena.ai
- AutoGluon 1.6 release notes: auto.gluon.ai/dev/whats_new/v1.6.0.html
- Chronos-2: arXiv 2510.15821; Moirai 2.0: arXiv 2511.11698
- Time-series foundation models on returns: arXiv 2511.18578
- Supervised autoencoder MLP for financial forecasting: arXiv 2404.01866
