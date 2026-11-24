# PLAYBOOK — QRT Data Challenge Hackathon, ETH Zürich, Tue 24 Nov 2026, 18:00–22:00

Distilled from [the research report](../reports/QRT%20Data%20Challenge%20hackathon%20format.md)
(sources there). Rules for agents: [AGENTS.md](../AGENTS.md). Fill §9 during the event.

## 1. What to expect

| | Status | Best answer |
|---|---|---|
| Time/place | **Confirmed** | 18:00–22:00, ETH, in person, own laptop, 123 registered, individual sign-up |
| Problem | Inference (~75%) | One anonymised QRT table: `X_train`, `y_train`, `X_test` → `ID,prediction` CSV |
| Metric | Inference | Accuracy on a return sign (most likely) · Spearman · multiclass accuracy · IPCW C-index |
| Platform | Inference | Kaggle-style upload, hidden test, public/private split; few submissions |
| Alternative (~20%) | Inference | **C3**: 2 h HackerRank optimisation set, partial credit, AI encouraged → §7 |
| Teams, AI rules, prizes | Unknown | Ask at the briefing (§3) |

QRT house style: shuffled anonymised dates (`DATE`, `TS`, `DAY_ID`), lag blocks `RET_1..RET_20`
(+ volume), group columns (`SECTOR`, `GROUP`, …), deliberately different train/test (other
dates / later regime / unseen hospital), benchmark ≈ 51% on finance tasks, honest winners
+1–2 points. **The ETH data may be ENS #167 (asset allocation) reused, or a fresh dataset in
the same style.**

## 2. Before the event

- [ ] Every laptop: `git clone`, `make setup`, `export AGENT_NAME=…`, `make rehearse`,
      `make check` green, `QRTC_CONFIG=configs/rehearsal/allocation.toml uv run qrtc baseline`.
- [ ] Every laptop: `make models` (torch + TabICL/TabPFN/TabDPT weights, ends with "all models
      work offline"); see [MODELS.md](MODELS.md).
- [ ] Every laptop: open `notebooks/00_quickstart.ipynb` with kernel *Python (qrtc)*, run all.
- [ ] Each person practises once: `scripts/agent.sh start exp test-run --claim "practice"`,
      commit, push, PR, `scripts/safe_merge.sh <n>` (on a throwaway docs change), `finish`.
- [ ] Optional, high value: register on challengedata.ens.fr, download **#167** and run the kit on
      it (write a `configs/ens167.toml`): `ROW_ID`, `TS`, `ALLOCATION`, `GROUP`, `TARGET`
      (real return, `binarize_target = true`), metric accuracy.
- [ ] Logistics: chargers, power strip, phone hotspot, agent logins tested, `brew install libomp`.
- [ ] Upload rehearsal (Submitter laptop): `make rehearse`, then `uv run qrtc baseline`,
      `uv run qrtc upload submissions/<file>.csv -m test`, `uv run qrtc uploads --reveal`
      (local backend: public vs hidden private score). If using Claude in Chrome: extension
      installed, Chrome profile ready for a human login.

## 3. Briefing questions (Captain asks, Validator writes answers in §9 verbatim)

1. Final ranking on a **private** split? Public share?
2. Submissions per team? Does the **best, last, or selected** count? Exact cutoff time?
3. Team formation and max size?
4. **AI assistants / agents** allowed (which, disclosure)?
5. Pre-written code allowed (incl. work on public ENS QRT datasets)?
6. **External data** allowed? (default: no)
7. If we find **hidden structure** (recoverable date order, overlapping lags, ID order) — may we use it? (default: no)
8. Presentation / code review? Prizes?
9. **How do we submit** (web form, Kaggle, API)? Who logs in, one account per team? Is a
   submission that fails validation counted? → fill `[upload]` (backend, url/slug,
   `max_total`, `max_per_day`, `reserve_final = 2`, ONE uploader) in the spec PR.
10. **Pretrained models** allowed? (TabICL/TabPFN were pretrained on synthetic tables only,
    TabDPT on public real tables; TabPFN-3.5 has a non-commercial licence.) Default: only
    `tabicl`/`tabpfn` v2 if the answer is unclear; never TabPFN-3.5 without a clear yes.

## 4. The clock (contest ≈ 18:20–21:20; anchor start to data release, end to the deadline)

| Clock | T | What | Done when |
|---|---|---|---|
| 17:30–18:00 | before | Sit together near power; `make check` + quickstart green on every laptop | all green |
| 18:00–18:20 | briefing | §3 questions; Plan A (data) or B (C3, §7) | §9 filled |
| 18:20–18:30 | +0–10 | Read statement, metric, sample submission. **Nobody codes.** Captain puts files in `data/raw/` | one sentence: "predict X per ID, scored by Y; test differs by Z" |
| 18:30–18:40 | +10–20 | Validator: `competition.toml` (incl. `[upload]`) from closest rehearsal config → PR → merge. `qrtc eda`; adversarial AUC | spec merged |
| 18:40–18:50 | +20–30 | Submitter: `qrtc upload` a **trivial valid entry** (majority / benchmark) | scored entry in `qrtc uploads` |
| 18:50–19:15 | +30–55 | `qrtc baseline` lgbm + logreg/ridge; Submitter uploads the better (score lands in `qrtc lb`) | CV ≈ LB within noise (else fix folds!) |
| 19:15–19:20 | +55–60 | Stand-up: four lanes, written tasks | |
| 19:20–20:35 | +60–135 | **Lanes A–D** (§5); check-ins at +90 and +120; merge gains > 1 std | runs logged |
| 20:35–20:55 | +135–155 | Final features, 3–5 seeds, `qrtc blend` (rank/hill); pick **two finals**: best-CV blend + simplest near-best | 2 candidates |
| 20:55–21:00 | +155–160 | `check-sub`, row count, id set, class balance, corr vs last good file | both pass |
| **21:00** | **T−20** | **Captain picks; Submitter `qrtc upload --final` both; Captain selects them on the platform. CODE FREEZE.** | both scored |
| 21:00–21:20 | T−20…0 | Verify; resubmit safety net if needed; 3 slides if judged | nothing new started |

Longer clock → stretch only the lanes. Shorter → cut lanes, never the first entry, the
baseline or the freeze.

## 5. Lanes (one written task per agent; never "improve the score")

- **A — cross-sectional features:** per-date ranks / z-scores of `RET_1..5`, window sums
  (3/5/10/20), de-meaned by date × group, date means (market move), volume ranks.
- **B — target & entity:** entity id (`ALLOCATION`, `STOCK`) as categorical; regress the real
  target and threshold vs classify; rank target for rank metrics; domain encodings
  (football: per-position aggregates, goal-difference proxy; survival: clinical risk score).
- **C — model diversity:** `lgbm_shallow`, `catboost_shallow`, `xgb_shallow`, logreg/ridge on
  ranks, 3 seeds each → saved runs for the blend. Pretrained in-context models (if §3 Q10
  allows): `tabicl`, `tabpfn`, `tabdpt`, one run each, early; cap large panels with
  `--params '{"max_rows": 20000, "n_bags": 2}'` ([MODELS.md](MODELS.md)). One such job per
  laptop at a time (they use the GPU).
- **D — Validator audit:** leak flags, `eda.lag_chain_links`, adversarial AUC, test-like CV
  subset, review every claimed gain.

Good agent prompt: *"In `experiments/$AGENT_NAME/03_csrank.ipynb`, add per-date percentile ranks
of RET_1..RET_5 via `qrtc.features.cs_rank`, run `run_cv` with `lgbm_shallow` on the standard
splits, save the run, and report CV mean ± std vs run `<baseline id>`. Do not modify src/."*

## 6. Recipes by problem shape

| Shape (rehearsal config) | CV | Start with | Then | Post-process |
|---|---|---|---|---|
| Return sign, panel (`stock_sign`, `allocation`) | `date_blocks` (or `time` if test is later) | `logreg`, `lgbm_shallow`; entity as category | per-date features (§5A), regress real target + threshold | threshold 0.5 vs per-date median split (`postprocess.median_split_within`); check base rate (always-long ≈ 50.6% in #167) |
| Regression, Spearman / IC (`electricity`) | `time`/`date_blocks` | `ridge` on raw | `features.rank_target` (+0.04 Spearman in #97), lgbm, catboost | rank-average blend |
| Multiclass (`football`) | `stratified` | `logreg`, `lgbm` | domain aggregates; draws are rarely predicted | argmax; `prior_adjust` if test prior differs |
| Survival, C-index (`survival`) | `group` by centre/hospital | `coxph` / `coxnet` | `rsf`, `gbsa`, `xgb` (cox); clinical risk features | output `risk_score` (higher = earlier death) |
| Weighted accuracy (|y|-weighted, #44) | `date_blocks` | regress the real target | `weight_col = "abs_target"` | sign |

Small tables (≤ ~50k training rows): run one pretrained model (`tabdpt` is fastest, or
`tabicl`) next to `logreg`/`ridge` in the first hour; it needs no tuning. On our synthetic
rehearsals it tied the linear models and beat default LightGBM by ~4 points, but never beat
logreg/ridge ([MODELS.md](MODELS.md)): decide by CV like everything else.

What did **not** pay in QRT write-ups: PCA, autoencoders, deep nets, per-entity models, heavy
stacking, Optuna tuning, de-meaning the target per day (throws away market drift).

## 7. Plan B — C3 optimisation format

Everyone reads every problem (10 min) → one owner per problem → **valid answer on every problem
by ~minute 40** (invalid = 0) → local scorer identical to the judge → greedy, then
`qrtc.optim.anneal` / `hill_climb` / `beam_search` (`noisy_minimise` for noisy black boxes) →
several independent agent attempts, keep the best → stop new approaches at T−20, confirm best
submissions at T−5. Watch bonus gates (Edinburgh: bonus only if > 70% on every main problem).
Templates: `templates/optim/solve_template.py` (stdin/stdout), `templates/optim/sa_template.cpp`.

## 8. Reflexes: leaks and noise

| You see | It usually means | Do |
|---|---|---|
| CV far above the public LB | same-date rows across folds; preprocessing fitted outside folds | regroup folds by date; fit inside folds |
| One feature adds > ~1 pt on a sign task | leak (ID order, row index, target-derived column, overlapping lags) | stop, Validator audits, ask organisers |
| LB jumps, CV does not | noise (±0.3–0.5 pt per 10k rows) | ignore; select on CV |
| Adversarial AUC ≫ 0.5 | test is another regime / entity / period | CV on the test-like subset; group folds |
| All one class, NaN, wrong row count | broken pipeline | `qrtc check-sub`; compare with sample submission |
| z-score explodes on a day | zero cross-sectional spread | use ranks (`cs_rank`) or clipped `cs_zscore` |

Final selection rule: **best CV that is stable across folds**, plus the simplest model within
1 std of it. Never the best public score.

## 9. Event log (fill in during the event, via small PRs)

- Briefing answers (verbatim, as relayed by the Captain): "the challenge will take 2 hours, data: price every 5 minutes for 286 sticks, goal is to predict whether each stock will beat or trail its country over the last 30 minutes of the day. Score the share of correct directions, large moves count more than small ones. Split: train on the 392 days, then get scored on later 199 days" (later: "329 training days"). The data has 386 stocks and 392 train days; the data wins.
- Data (Kaggle `qrt-x-eth-hackathon-24-11-2026`, deadline 21:00): `x_train.parquet` 144,225 rows, `x_test.parquet` 74,932 rows, `y_train.csv` (`ID, Close return`). Columns `ID, DATE_ID, STOCK_ID, COUNTRY, OPEN_TIME, CLOSE_TIME, Price TS` (list of 5-min prices up to the cutoff, mostly 96 bars, a few 48). 6 countries: GBR 136, FRA 76, DEU 75, ITA 40, NLD 33, ESP 26 stocks (DEU/ITA/NLD open 08:00, others 07:00). Target is already demeaned per DATE_ID x COUNTRY (mean exactly 0). Base rate 49.7% up; weighted base rate 50.0%; always-up = always-down = 0.5000 weighted accuracy. Host benchmark -(last 5-min return) 0.5227; -(last 5-min country-relative return) 0.5264 on train. Kaggle submission must hold all 219,157 ids (train + test).
- Split: 392 train / 199 test DATE_IDs, disjoint, union = 0..590; all 386 stocks in both. DATE_ID is shuffled: price levels of consecutive ids are no closer than random pairs (median abs log gap 1.22 vs 1.19), so day order is NOT recoverable from levels. CV `date_blocks` (random whole days) is the closest match available; kept.
- Spec PR #9 merged 18:37 (dmr-claude-0); Kaggle limit #12 merged 18:52 (10/day). Validator audit: dmr-claude-4, `experiments/dmr-claude-4/audit.py`.
- Shift: adversarial AUC train vs test 0.766 with row folds, **0.657 with day-grouped folds** (the honest number). Driver: volatility (`vol`, `vol_rank`, `ret_day`). Test days are calmer: per-day median vol 0.00119 (test) vs 0.00140 (train), train has a high-vol tail to 0.0037 (test max 0.0025). Consequence: prefer scale-free features (vol-scaled, ranks, country-relative); be wary of gains that come from the high-vol train days; check claimed gains on the calm train days too.
- Models (dmr-claude-2, PR #7, official CV): **linear beats trees**. Best is ridge on the real `Close return` (fit target clipped at the 2% tails, sign; logits demeaned per DATE_ID x COUNTRY) 0.5427 ± 0.0046 vs lgbm 0.533-0.538. The one large effect is daily beta x country-move terms (-0.55 pts when removed): they let ridge split a stock's move into its beta part and its own part, and only the own part reverts. No gain from rank-gauss inputs, stock-average beta, per-country slopes, per-bar returns, interactions, blends or test-like day weights (all within ±0.2 pts or worse). Day-level adversarial check on per-day aggregates (incl. stocks per day: test ≥ 243, train down to 128 = later period) gives AUC 0.84; re-scoring OOF on the 118 most test-like train days (`experiments/dmr-claude-2/testlike.py`) keeps the model ranking (best 0.5477 there vs baseline 0.5391).
- Leaks/quirks: `qrtc eda` leakage checks clean. Hidden structure: none found (day order not recoverable from price levels; no lag blocks). No organiser ruling needed so far.
- **PUBLIC LB = TRAIN IDS (Validator, 19:15).** The public leaderboard scores the train rows of the 219,157-id file: host benchmark 0.52273 on train ids = 0.52273 public; baseline LGBM 0.66308 on train ids = 0.66307 public (U002); ridge-v1 0.5459 on train ids. So the public score is in-sample fit and rewards overfitting (LGBM memorises train; ridge does not). The private score is presumably the 74,932 test ids. **Ignore the public LB entirely; select finals on CV only.** Do not exploit this (e.g. writing true train labels into train ids): it is not allowed without an organiser ruling, and it cannot move the private score.
- practice run by can
