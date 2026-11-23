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
- [ ] Every laptop: open `notebooks/00_quickstart.ipynb` with kernel *Python (qrtc)*, run all.
- [ ] Each person practises once: `scripts/agent.sh start exp test-run --claim "practice"`,
      commit, push, PR, `scripts/safe_merge.sh <n>` (on a throwaway docs change), `finish`.
- [ ] Optional, high value: register on challengedata.ens.fr, download **#167** and run the kit on
      it (write a `configs/ens167.toml`): `ROW_ID`, `TS`, `ALLOCATION`, `GROUP`, `TARGET`
      (real return, `binarize_target = true`), metric accuracy.
- [ ] Logistics: chargers, power strip, phone hotspot, agent logins tested, `brew install libomp`.

## 3. Briefing questions (Captain asks, Validator writes answers in §9 verbatim)

1. Final ranking on a **private** split? Public share?
2. Submissions per team? Does the **best, last, or selected** count? Exact cutoff time?
3. Team formation and max size?
4. **AI assistants / agents** allowed (which, disclosure)?
5. Pre-written code allowed (incl. work on public ENS QRT datasets)?
6. **External data** allowed? (default: no)
7. If we find **hidden structure** (recoverable date order, overlapping lags, ID order) — may we use it? (default: no)
8. Presentation / code review? Prizes?

## 4. The clock (contest ≈ 18:20–21:20; anchor start to data release, end to the deadline)

| Clock | T | What | Done when |
|---|---|---|---|
| 17:30–18:00 | before | Sit together near power; `make check` + quickstart green on every laptop | all green |
| 18:00–18:20 | briefing | §3 questions; Plan A (data) or B (C3, §7) | §9 filled |
| 18:20–18:30 | +0–10 | Read statement, metric, sample submission. **Nobody codes.** Captain puts files in `data/raw/` | one sentence: "predict X per ID, scored by Y; test differs by Z" |
| 18:30–18:40 | +10–20 | Validator: `competition.toml` from closest rehearsal config → PR → merge. `qrtc eda`; adversarial AUC | spec merged |
| 18:40–18:50 | +20–30 | Captain uploads a **trivial valid entry** (majority / benchmark) after `qrtc check-sub` | scored entry |
| 18:50–19:15 | +30–55 | `qrtc baseline` lgbm + logreg/ridge; Captain submits the better; `record-lb` | CV ≈ LB within noise (else fix folds!) |
| 19:15–19:20 | +55–60 | Stand-up: four lanes, written tasks | |
| 19:20–20:35 | +60–135 | **Lanes A–D** (§5); check-ins at +90 and +120; merge gains > 1 std | runs logged |
| 20:35–20:55 | +135–155 | Final features, 3–5 seeds, `qrtc blend` (rank/hill); pick **two finals**: best-CV blend + simplest near-best | 2 candidates |
| 20:55–21:00 | +155–160 | `check-sub`, row count, id set, class balance, corr vs last good file | both pass |
| **21:00** | **T−20** | **Captain uploads both, selects them. CODE FREEZE.** | both scored |
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
  ranks, 3 seeds each → saved runs for the blend.
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

- Briefing answers (verbatim):
- Data: files, rows, columns, id/date/group, split structure, base rate:
- Spec PR / merged at:
- Submissions (time, run_id, CV, public LB):
- Leaks/quirks found (and organiser ruling):
