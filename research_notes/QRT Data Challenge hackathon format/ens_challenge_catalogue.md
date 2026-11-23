# Catalogue of QRT data challenges on ENS "Challenge Data", plus other QRT-sponsored challenges (researched 23 Nov 2026)

Method note: I downloaded every public page `https://challengedata.ens.fr/challenges/N` for N = 1..185 and searched them for the provider "QRT"/"Qube". Exactly seven pages are "by QRT": **23, 44, 72, 97, 143, 162, 167**. No page exists above 167. Some IDs return 404 (unpublished or removed), so an unlisted QRT challenge cannot be fully ruled out. The leaderboards (`/participants/challenges/N/ranking/public|private`) need a login, so I could not read them directly. Winning and leaderboard scores below come from participants' GitHub write-ups and are marked as such.

## Q1. Which QRT challenges exist? (full catalogue with specs)

### Takeaway
QRT has posted seven ENS Challenge Data challenges, one per season since 2020: stock residual-return sign (2020), liquid-asset reconstruction (2021), Stiefel-manifold factor learning (2022), electricity-price "explanation" (2023), football match outcome (2024), myeloid-leukemia survival with Gustave Roussy (2025 season), and asset-allocation return sign (2026). The 2026 dataset was first used at the in-person "QRT Grand Data Challenge 2025" in Paris (October 2025). I found no QRT challenge on FX, commodities, volatility or ETFs. The ETF/portfolio theme shows up only in the 2026 asset-allocation task.

### Cited Findings

**Summary table**

| # | Season / start date | Title | Task type | Target | Metric | Official benchmark | Benchmark score |
|---|---|---|---|---|---|---|---|
| 23 | 2020, started 6 Jan 2020 | Stock Return Prediction | binary classification | sign of next-day residual return of a US stock (`RET`) | accuracy | Random Forest (500 trees, max depth 8) on 5 lags + STOCK + one conditional-mean feature | **51.31%** public — [C23](https://challengedata.ens.fr/challenges/23) |
| 44 | 2021, started 4 Jan 2021 | Reconstruction of Liquid Asset Performance | sign classification | sign of same-day liquid-asset return (`RET_TARGET`, ±1) | \|y\|-weighted accuracy | 100 univariate linear regressions (max-abs-correlation illiquid asset, OAS-shrunk beta) | not published on page — [C44](https://challengedata.ens.fr/challenges/44) |
| 72 | 2022, started 5 Jan 2022 | Learning factors for stock market returns prediction | constrained regression (parameter estimation) | output a 2510-vector (A: 250×10 on Stiefel manifold, β: 10) | mean cosine similarity between predicted and realized cross-sections (−1 if not orthonormal) | best of 1000 random Gram–Schmidt orthonormal A + OLS β | not published on page — [C72](https://challengedata.ens.fr/challenges/72) |
| 97 | 2023, started 6 Jan 2023 | Can you explain the price of electricity? (FR/DE) | regression | daily change of 24H baseload futures price (`TARGET`) | Spearman correlation | linear regression, NaN→0, COUNTRY dropped | **15.86%** public — [C97](https://challengedata.ens.fr/challenges/97) |
| 143 | 2024, started 10 Jan 2024 | Football: Can you guess the winner? | 3-class classification | one-hot [`HOME_WINS`,`DRAW`,`AWAY_WINS`] | accuracy | XGBoost on team data predicting AWAY_WINS vs not (never predicts draw) | ~47.5% "on the training set" (holdout 0.4742) — [C143](https://challengedata.ens.fr/challenges/143) |
| 162 | 2025 season (page now says started 13 Jan 2026; see conflict below) | Overall Survival Prediction for patients diagnosed with Myeloid Leukemia (with Institut Gustave Roussy) | survival ranking | `risk_score` (higher = shorter survival) from `OS_YEARS`, `OS_STATUS` | IPCW C-index, τ = 7 years (scikit-survival `concordance_index_ipcw`) | Cox PH on BM_BLAST, HB, PLT + mutation count (ranking benchmark); LightGBM on clinical data as example | public score not on page; notebook holdout C-index 0.68 — [C162](https://challengedata.ens.fr/challenges/162) |
| 167 | 2026, started 22 Jan 2026 | Asset Allocation Performance forecasting ("Trust or Short?") | binary classification | sign of next-day allocation return (`TARGET` given as real return) | accuracy | LightGBM (mse, depth 3, lr 0.01, 500 rounds) + rolling-mean/vol and cross-sectional date-mean features; 8-fold CV over dates | **0.5079** public — [C167](https://challengedata.ens.fr/challenges/167) |

**Seasons and platform context**
- Challenge Data is run by the Institut Louis Bachelier with Collège de France, PSL and ENS. Seasons usually launch in January (in Stéphane Mallat's Collège de France course) and end with an award ceremony. The 2025 season closed on 15 Dec 2025 and was then "extended through all of 2026". QRT's "Predicting asset allocation performance" was announced as the new challenge, and a normal season is planned for 2027. QRT is called a "historical partner" — [ILB, 2 Feb 2026](https://www.institutlouisbachelier.org/en/on-the-program/actualites/data-challenge-a-2025-season-extended-to-2026-and-a-new-challenge-signed-by-qrt)
- QRT runs a dedicated forum for its ENS challenges at challengedata.qube-rt.com. Challenges 23, 44, 97 and 143 link to it, and the 2021 benchmark notebook points to it — [C23](https://challengedata.ens.fr/challenges/23), [C143](https://challengedata.ens.fr/challenges/143), [2021 benchmark notebook](https://github.com/medelbechir/QRT-Data-Challenge-2021)
- The football page calls its task "this year's QRT data challenge", and the leukemia page says the same. QRT brands each yearly ENS challenge as "the QRT data challenge" — [C143](https://challengedata.ens.fr/challenges/143), [C162](https://challengedata.ens.fr/challenges/162)

**#23 — Stock Return Prediction (2020)** — source: [C23](https://challengedata.ens.fr/challenges/23)
- Goal: predict the sign of a US stock's **residual** return (return "without the market impact") on day t from the previous 20 business days. The daily return is R_j^t = P_j^t / P_j^{t-1} − 1, with prices adjusted for dividends and splits.
- Relative volume: V̄_j^t = V^t / median(V_j^{t-1..t-20}), then 𝒱_j^t = V̄_j^t − (1/n)Σ_i V̄_i^t (de-meaned across stocks on the same day).
- Inputs: 47 columns, `ID` plus `DATE` ("randomized and anonymized so there is no continuity or link between any dates"), `STOCK`, `INDUSTRY`, `INDUSTRY_GROUP`, `SUB_INDUSTRY`, `SECTOR`, `RET_1`…`RET_20` (RET_1 = previous day), `VOLUME_1`…`VOLUME_20`.
- Output: `ID`, `RET` (binary sign). Rows: 418,595 train, 198,429 test. The train/test split method is not stated.
- Benchmark: Random Forest, 500 trees, max depth 8. Features are the 5 previous returns and relative volumes, `STOCK`, and "the mean of RET_1 conditionally to RANK and SECTOR". NaN→0. Public score **51.31%**.
- Winners: 1 Romain Poncet; 3 Agnès François and Jessy Idez (2nd shown as "-").
- A later participant reports a 7-model stacked ensemble at **LB 0.5151 vs baseline 0.5131** — [george-hutchings/qrt-challenge](https://github.com/george-hutchings/qrt-challenge)

**#44 — Reconstruction of Liquid Asset Performance (2021)** — source: [C44](https://challengedata.ens.fr/challenges/44)
- Goal: map the same-day returns of 100 illiquid assets to the **sign** of the returns of 100 liquid assets, so that signals on illiquid assets can be projected onto liquid ones.
- Inputs: 103 columns: `ID`, `ID_DAY` (randomized and anonymized), `RET_i` (100 illiquid assets; column names such as RET_216 or RET_0 are asset IDs, not lags), `ID_TARGET` (which liquid asset the row asks about). Each day is repeated 100 times, once per target.
- Output: `ID`, `RET_TARGET` (predicted sign in {−1, 1}). A supplementary file gives `ID_asset` and `CLASS_LEVEL_1..4` sector/industry hierarchy.
- Rows: train 267,100 (2,748 days), test 114,468 (1,177 days). "The train/test split has been performed randomly along the day variable… no day is shared."
- Metric (weighted accuracy): f(y, ŷ) = (1/‖y‖₁) Σ_i |y_i| · 1{ŷ_i = sign(y_i)}.
- Benchmark: for each liquid asset, pick the illiquid asset with maximum absolute correlation, then predict sign(β̂·R_k). No score is given on the page.
- The benchmark notebook loads `X_train.parquet`, `y_train.csv`, `X_test.parquet`, estimates the covariance with sklearn `oas` shrinkage, sets beta = cov/diag(cov), fills NaN with the day-mean prediction, and outputs `np.sign` as int — [Benchmark 2021 QRT Challenge.ipynb](https://github.com/medelbechir/QRT-Data-Challenge-2021)

**#72 — Learning factors for stock market returns prediction (2022)** — source: [C72](https://challengedata.ens.fr/challenges/72)
- Model: S_{t+1} = Σ_{ℓ=1}^{F} β_ℓ F_{t,ℓ}, with factors F_{t,ℓ} = Σ_{k=1}^{D} A_{kℓ} R_{t+1−k}. The columns of A must be orthonormal (AᵀA = I_F, the Stiefel manifold). D = 250, F = 10.
- Data: `X_train` holds the cleaned daily returns of **50 stocks × 754 days** (rows = stocks, columns = days). `Y_train` (the returns to predict) is "also contained in X_train". Scoring uses a different set of 50 stocks over the 3-year period.
- Output: one vector of length 2510 = [A_1; …; A_10; β].
- Metric: Metric(A, β) = (1/504) Σ_{t=250}^{753} ⟨S̃_t, R̃_t⟩ / (‖S̃_t‖‖R̃_t‖). It is set to −1 unless |⟨A_i, A_j⟩ − δ_ij| ≤ 1e-6 for all i, j. Range [−1, 1].
- Benchmark: repeat 1000 times (sample a 250×10 N(0,1) matrix, Gram–Schmidt it with `randomA`, fit β by OLS with `fitBeta`, score on train with `metricTrain`) and keep the best. No score is given on the page.
- Winners: 1 New_try, 2 lp505, 3 Nicolas Raoult.
- Participants report 6th of 413 — [Diego-Acuna](https://github.com/Diego-Acuna/stock-returns-factor-learning-qrt) — and large public→private shake-ups — [PirashanthR](https://github.com/PirashanthR/QRT-Data-Challenge-2022)

**#97 — Can you explain the price of electricity? (2023)** — source: [C97](https://challengedata.ens.fr/challenges/97)
- Goal: "explain" the daily variation of 24H baseload electricity futures for France and Germany from **same-day** weather, energy and commodity variables. The page says this is not a forecasting problem.
- Inputs: 35 columns: `ID` (day × country), `DAY_ID` (anonymised; "all data corresponding to a specific day is consistent"), `COUNTRY` (FR/DE), `GAS_RET`, `COAL_RET`, `CARBON_RET`. Per country x ∈ {DE, FR}: `x_TEMP`, `x_RAIN`, `x_WIND`, `x_GAS`, `x_COAL`, `x_HYDRO`, `x_NUCLEAR`, `x_SOLAR`, `x_WINDPOW`, `x_LIGNITE`, `x_CONSUMPTON` (sic), `x_RESIDUAL_LOAD`, `x_NET_IMPORT`, `x_NET_EXPORT`. Also `DE_FR_EXCHANGE`, `FR_DE_EXCHANGE`.
- Output: `ID`, `TARGET`. Rows: 1,494 train, 654 test. X_train and X_test cover "two different time periods".
- Metric: Spearman correlation between predictions and true daily price changes on the test set.
- Benchmark: linear regression, NaN→0, COUNTRY dropped (one model for both countries). Public score **15.86%**.
- Winners: 1 Mohamed Ali Ben Amara, 2 mb, 3 Amine Abdelkader.
- Leaderboard reports: Spearman 0.2749 was rank 77 of 866 — [talhajamal11](https://github.com/talhajamal11/QRT---Electricity-Prices---2023). Later, **0.5658 was rank 6 of 1,157**, using a leak: `ID` keeps calendar order while `DAY_ID` is shuffled — [ryan-racer](https://github.com/ryan-racer/QRT-Electricity-Forecasting). Also rank 352 of 1,070 — [alexandremathiasdonnat](https://github.com/alexandremathiasdonnat/Data-Challenges)

**#143 — Football: Can you guess the winner? (2024)** — source: [C143](https://challengedata.ens.fr/challenges/143)
- Data from Sportmonks for dozens of leagues. Files: `X_train.zip`, `X_test.zip` (4 CSVs each: home/away × team/player), `Y_train.csv`, `Y_train_supp.csv` (with `GOAL_DIFF_HOME_AWAY`). Statistics summarise the last 5 games plus season-to-date, aggregated by sum, average and std.
- Team files: `ID`, `LEAGUE`, `TEAM_NAME` (LEAGUE and TEAM_NAME removed in test) plus 25 stats: TEAM_ATTACKS, TEAM_BALL_POSSESSION, TEAM_BALL_SAFE, TEAM_CORNERS, TEAM_DANGEROUS_ATTACKS, TEAM_FOULS, TEAM_GAME_DRAW, TEAM_GAME_LOST, TEAM_GAME_WON, TEAM_GOALS, TEAM_INJURIES, TEAM_OFFSIDES, TEAM_PASSES, TEAM_PENALTIES, TEAM_REDCARDS, TEAM_SAVES, TEAM_SHOTS_INSIDEBOX, TEAM_SHOTS_OFF_TARGET, TEAM_SHOTS_ON_TARGET, TEAM_SHOTS_OUTSIDEBOX, TEAM_SHOTS_TOTAL, TEAM_SUBSTITUTIONS, TEAM_SUCCESSFUL_PASSES, TEAM_SUCCESSFUL_PASSES_PERCENTAGE, TEAM_YELLOWCARDS.
- Player files: `ID`, `LEAGUE`, `TEAM_NAME`, `POSITION`, `PLAYER_NAME` (names removed in test) plus 52 stats. "All variables have been standardized."
- Output: `ID`, `HOME_WINS`, `DRAW`, `AWAY_WINS` (one-hot). Metric: accuracy of the vector. Row counts are not stated. The data is tagged ">1GB".
- Benchmarks: (1) always HOME win, ≈44% on train; (2) "the one you will see in the leaderboard": gradient boosting on team data only, predicting AWAY_WINS, ≈47.5% on the training set.
- File and column names come from the benchmark notebook: `train_home_team_statistics_df.csv`, `train_away_team_statistics_df.csv` (and player equivalents), with columns like `TEAM_SHOTS_TOTAL_season_sum` — [benchmark_notebook.ipynb](https://github.com/ayoub-mg/QRT-Football-Challenge)
- Leaderboard reports: "0.49–0.50 is the top of the leaderboard", with ±0.003 uncertainty on ~25,000 test matches — [louisrodembourg](https://github.com/louisrodembourg/qrt-football-prediction). 4th of 800+ — [hugopuybareau](https://github.com/hugopuybareau/QRT_DATA_CHALLENGE). In a "QRT Data Challenge × Télécom Business Finance" sub-event, the top private score was 0.4875 — [ayoub-mg](https://github.com/ayoub-mg/QRT-Football-Challenge)

**#162 — Overall Survival Prediction for Myeloid Leukemia (2025 season, with Institut Gustave Roussy)** — source: [C162](https://challengedata.ens.fr/challenges/162)
- Data from 24 clinical centres. Files: `X_train.zip`, `X_test.zip` (clinical + molecular), `Y_train.csv`. Patients: 3,323 train, 1,193 test.
- Clinical data, one row per patient: `ID`, `CENTER`, `BM_BLAST`, `WBC`, `ANC`, `MONOCYTES`, `HB`, `PLT`, `CYTOGENETICS` (ISCN karyotype string, e.g. "46,XX").
- Molecular data, one row per mutation: `ID`, `CHR`, `START`, `END`, `REF`, `ALT`, `GENE`, `PROTEIN_CHANGE`, `EFFECT`, `VAF`.
- Targets: `OS_YEARS`, `OS_STATUS` (1 = death, 0 = alive at last follow-up). Submission: index column `ID` plus a `risk_score` column. Only the ordering matters.
- Metric: IPCW C-index from scikit-survival `concordance_index_ipcw`, "clipped at… a maximum horizon of 7 years". C = concordant pairs / comparable pairs.
- Benchmarks: LightGBM on clinical data ignoring censoring (example only), and **Cox PH (clinical + limited mutation info)**, which is used for the ranking. The page gives no score.
- Winners: 1 Arthur De Rouck and Ruben Barata (tie), 2 Octave Cerclé, 3 Ryan Mathieu.
- Date conflict: the live page says "Started on Jan. 13, 2026" — [C162](https://challengedata.ens.fr/challenges/162). But a benchmark write-up dated 10 Feb 2025 — [Moudiki](https://thierrymoudiki.github.io/blog/2025/02/10/python/Benchmark-QRT-Cube) — and repos titled "QRT-Challenge-Data-2025" — [carlmonnaert](https://github.com/carlmonnaert/QRT-Challenge-Data-2025) — show it ran in the 2025 season.
- Test patients all come from one unseen centre: test IDs are `KYW1…KYW1193` in the benchmark notebook output — [Benchmark_nqBJ7fO.ipynb](https://github.com/chenenen13/QRT-Leukemia-DataChallenge). A participant states 3,173 labelled training patients from 23 hospitals, and all test patients from a 24th (`KYW`) — [lucaspaulien](https://github.com/lucaspaulien/qrt-leukemia-survival)
- Reported winner score "0.7744" — [chenenen13 README](https://github.com/chenenen13/QRT-Leukemia-DataChallenge). This is a single secondary source and I could not verify it on the leaderboard.

**#167 — Asset Allocation Performance forecasting (2026)** — source: [C167](https://challengedata.ens.fr/challenges/167)
- Each row is (date, allocation). An allocation is a long/short portfolio with Σ|w| = 1, rebalanced daily. Its return is r_{S,t+1} = Σ_i w_{S,t,i} r_{i,t+1}.
- Columns: `ROW_ID` (index across files), `TS` (DATE_0001…, "anonymized and shuffled so there is no guarantee of continuity"), `ALLOCATION` (ALLOCATION_01…, consistent across dates), `RET_1..RET_20`, `SIGNED_VOLUME_1..20` (V_{S,t} = Σ w·V, rescaled on a rolling basis), `MEDIAN_DAILY_TURNOVER` (median of Σ|w_t − w_{t−1}| over 20 days), `GROUP` (anonymised), and `TARGET` (true next-day return, train only).
- Files: `X_train.csv`, `y_train.csv`, `X_test.csv`, `sample_submission.csv`, `benchmark_submission.ipynb`. Rows: **527,073 train, 31,870 test**.
- Metric: Accuracy = (1/N) Σ 1[sign(r̂_i) = sign(r_i)], where sign(x) = 1 if x > 0 else 0. Submit 1 (long) or 0 (short).
- Benchmark: features AVERAGE_PERF_{3,5,10,15,20}, their date-wise means ALLOCATIONS_AVERAGE_PERF_*, STD_PERF_20 and ALLOCATIONS_STD_PERF_20. Models are Ridge(α=1) and LightGBM. Public score of the LGBM model is **0.5079**.
- Benchmark code details: the date-wise means use `groupby('TS').transform('mean')`. LightGBM uses objective mse, lr 1e-2, max_depth 3, 500 rounds, regressing the real-valued TARGET and thresholding at 0. KFold(8, shuffle) **over unique TS dates**, with fold accuracies 51.95–52.48% and mean **52.22% ±0.20**. Output is `(preds>0).astype(int)` indexed by ROW_ID — [benchmark_submission.ipynb copy](https://github.com/phillipe-BAGUEKA/QRT-Asset-Allocation-Forecasting)
- Data facts from participants:
  - About 278 allocations; always-up baseline 50.6% — [martindelamare](https://github.com/martindelamare/qrt-asset-allocation-forecasting)
  - Positive rate 0.5073 — [phillipe-BAGUEKA](https://github.com/phillipe-BAGUEKA/QRT-Asset-Allocation-Forecasting)
  - Dates DATE_0001–DATE_2642. The universe grows from 65 to 276 allocations. The test set resembles the most recent regime — [najibnasr](https://github.com/najibnasr/qrt-challenge-167)
- Leaderboard reports:
  - A legitimate blend reached **51.22% public vs 50.79% benchmark** — [najibnasr](https://github.com/najibnasr/qrt-challenge-167)
  - Public scores "near 81%" exist and "cannot come from predicting daily return signs" — [najibnasr](https://github.com/najibnasr/qrt-challenge-167)
  - A historical reconstruction from overlapping windows and public prices scored **77.41% public** — [RyanBalech](https://github.com/RyanBalech/qrt-asset-allocation-reconstruction)

**Other QRT-sponsored challenges outside the ENS platform**
- **QRT Grand Data Challenge 2025 (Paris, in person).** This was the first public use of the asset-allocation task.
  - 104 teams and 100+ students from CentraleSupélec, Polytechnique, ENSAE, Télécom, Mines and Ponts, selected by CV. 180,245 training observations with 20-day windows of allocation returns, volume-weighted liquidity and turnover. Metric: accuracy on the next-day sign — [malerbe/QRT_challenge](https://github.com/malerbe/QRT_challenge)
  - Described as an "8-day hackathon organised by QRT" against ~200 students from X, CS, ENSAE and Mines. Task: classify whether to short a portfolio's allocations, on "synthetic data", scored by accuracy — [iliasbendali](https://github.com/iliasbendali/QRT-Grand-Data-Challenge)
  - Results: ranked 13 of 200 — [leclercrafael](https://github.com/leclercrafael/QRT-Asset-Allocation-Performance-forecasting); "would have placed… 3 out of more than 100 teams" after the event — [pradosth0](https://github.com/pradosth0/QRT_Asset_Allocation_Performance_forecasting_2025)
  - QRT's LinkedIn post (snippet via web search): "QRT's Quantitative Researchers hosted our very first in-person Kaggle Competition in Paris. The competition kicked off online earlier this month and culminated in a high-stakes in-person finale" — [LinkedIn post linked from malerbe README](https://www.linkedin.com/posts/qube-rt_last-weekend-qrts-quantitative-researchers-activity-7383847260226342914-XbBW)
- **QRT × Imperial "Code. Compute. Compete" hackathon (2026).** Solo 1st of 68 teams (about 240 participants). A search snippet describes a 2-hour HackerRank test with four optimisation problems plus a bonus. That makes it an algorithmic coding contest, not a data challenge — [xRyann2255/qrt-hackathon-2026](https://github.com/xRyann2255/qrt-hackathon-2026)
- **QRT × CompSoc C3 Coding Hackathon (Edinburgh/St Andrews).** Per the search snippet, the top three teams were invited to "QRT's Grand Final in London" against Oxford, Cambridge, Imperial and Warwick — [UK Hackathons listing](https://ukhackathons.com/events/qrt-x-compsoc-c3-coding-hackathon)
- **QRT Quant Challenge India 2026, Phase 2.** Notebooks build a US-equity universe (top 5,000 by market cap), compute technical features, test their predictive power and construct portfolios, with an in-person workshop — [yszanwar/phase2_qrt_challenge](https://github.com/yszanwar/phase2_qrt_challenge)
- **Imperial Algothon 2024 "QRT Challenge"** — [wowthecoder/ic-algothon-24](https://github.com/wowthecoder/ic-algothon-24), and **HackTheBurgh 2025 "QRT Challenge"**, won by an AI crypto-portfolio-manager app — [syntax/htb2025](https://github.com/syntax/htb2025). Details of these tasks were not available.
- Non-QRT neighbours on the same platform, sometimes confused with QRT: #16 "Prediction of daily stock movements on the US market" is by **CFM** (2019; sign of 15:30–16:00 return; LightGBM benchmark ≈51.8%) — [C16](https://challengedata.ens.fr/challenges/16). #146 "Guess the Stock" is CFM 2024 — [egardn/Guess_The_Stock](https://github.com/egardn/Guess_The_Stock)

### Inferences
- The candidate "US equity daily return sign ≈2020-21" is challenge 23 (2020). The 2021 QRT challenge is a different one: #44, liquid/illiquid sign reconstruction with a weighted accuracy metric.
- Inferred timing of the Grand Data Challenge 2025: the LinkedIn activity ID 7383847260226342914 decodes to 14 Oct 2025. The post says "last weekend", so the in-person finale was probably 11–12 Oct 2025, after an online phase that started earlier that month. The same task then went public on ENS as #167 on 22 Jan 2026. That suggests QRT reuses its ENS-style datasets for live student events. An ETH hackathon in Nov 2026 could plausibly reuse a similar shape (anonymised panel, 20-lag returns, sign accuracy). This is a pattern-based guess, not something announced.
- The page date for #162 (13 Jan 2026) is probably a re-listing artefact of the season extension. Its content and winners belong to the 2025 season.
- In #23, "RANK" in the benchmark text is probably a typo for `DATE`, i.e. a sector-by-date mean of RET_1. This is an unverified guess: the notebook was not available.
- The 2024 football and 2025 leukemia tasks are not finance problems. QRT deliberately varies the domain, so the team should prepare generic tabular, multi-table and survival templates, not only return-prediction code.

### Gaps
- Official winning and leaderboard scores for #23, #44, #72, #97, #143 and #162 (private and public) are behind the login-gated leaderboard. Only participant-reported figures are given above.
- Benchmark scores for #44 and #72, and the public score of the #162 Cox benchmark, are not on the challenge pages.
- The train/test split method for #23 (by date vs random) and the row counts for #143 are not stated on the pages.
- I could not open the QRT LinkedIn post or the UK hackathon pages directly. Their details come from search snippets and repo READMEs.
- I found no official QRT webpage describing an ETH Zürich "QRT Data Challenge Hackathon" format.

## Q2. What do the benchmark notebooks do, and how big is the gap to winners?

### Takeaway
QRT benchmarks are deliberately simple and beatable, but only by small margins in the finance tasks. Examples: a RF at 51.31% (#23), a LightGBM at 50.79% (#167), linear regression at 15.86% Spearman (#97), XGBoost on team stats at ~47.5% (#143), and Cox on 4 features at ~0.68 holdout C-index (#162). Legitimate improvements in the market-sign tasks are about +0.2 to +1 accuracy point. The large gaps appear only in #97, #162 and leak-exploiting #167 entries.

### Cited Findings
- **#23:** RandomForest(500 trees, max_depth 8), NaN→0, features RET_1..5, VOLUME_1..5, STOCK and a conditional mean of RET_1. Public 51.31% — [C23](https://challengedata.ens.fr/challenges/23). A participant's 7-model stack reached 0.5151 on the leaderboard (+0.2 pt) — [george-hutchings](https://github.com/george-hutchings/qrt-challenge)
- **#44:** per liquid asset, the single most-correlated illiquid asset; OAS-shrunk covariance; beta = cov/var; sign of beta × return. Missing values are filled with the cross-sectional mean, then the day mean of predictions — [benchmark notebook](https://github.com/medelbechir/QRT-Data-Challenge-2021); [C44](https://challengedata.ens.fr/challenges/44). No score was found.
- **#72:** random search over 1000 uniformly sampled Stiefel matrices with an OLS β — [C72](https://challengedata.ens.fr/challenges/72). No score was found. One participant reports that public and private rankings diverged massively (1st public → 81st private; 99th public → 1st private) — [PirashanthR](https://github.com/PirashanthR/QRT-Data-Challenge-2022)
- **#97:** OLS, NaN→0, COUNTRY dropped, public 15.86% — [C97](https://challengedata.ens.fr/challenges/97). Participant figures: LightGBM 27.49% (rank 77/866) — [talhajamal11](https://github.com/talhajamal11/QRT---Electricity-Prices---2023); 0.5658 using per-country models, a rank-transformed target and recovered day order via `ID` — [ryan-racer](https://github.com/ryan-racer/QRT-Electricity-Forecasting)
- **#143:**
  - Notebook method: XGBoost binary (AWAY_WINS vs not), `max_depth 8`, `learning_rate 0.025`, `tree_method hist`, early stopping 100 rounds, 80/20 then 80/20 random splits. Zeros are replaced with NaN. HOME vs AWAY is chosen by argmax with the draw probability forced to 0. Holdout accuracy is **0.4742** vs 0.4401 for always-home — [benchmark_notebook.ipynb](https://github.com/ayoub-mg/QRT-Football-Challenge)
  - The official page says ≈47.5% — [C143](https://challengedata.ens.fr/challenges/143)
  - Top of the leaderboard is 0.49–0.50 (about +2 pts) — [louisrodembourg](https://github.com/louisrodembourg/qrt-football-prediction)
- **#162:**
  - Benchmark notebook: features `['BM_BLAST','HB','PLT']`, median imputation, `train_test_split(test_size=0.3, random_state=42)`.
  - LightGBM on `OS_YEARS`, ignoring censoring and using negative predictions as risk: test IPCW-C 0.65. Cox PH on the 3 features: 0.66. Cox adding `Nmut` (mutation count per patient): **0.68 train / 0.68 test**.
  - Metric call: `concordance_index_ipcw(y_train, y_test, pred, tau=7)[0]`.
  - Source for the three points above — [Benchmark_nqBJ7fO.ipynb](https://github.com/chenenen13/QRT-Leukemia-DataChallenge)
  - Another reproduction got Cox 0.6606 on the holdout — [Moudiki](https://thierrymoudiki.github.io/blog/2025/02/10/python/Benchmark-QRT-Cube)
  - Leave-centre-out CV: benchmark 0.6847, IPSS-M 0.7086, ensemble 0.7238 — [lucaspaulien](https://github.com/lucaspaulien/qrt-leukemia-survival)
  - Reported winner 0.7744 (≈ +0.09 over the benchmark; unverified) — [chenenen13](https://github.com/chenenen13/QRT-Leukemia-DataChallenge)
- **#167:** Ridge + LightGBM as above, with CV over dates at 52.22% local and **0.5079** public (a ~1.4 pt local→public drop) — [C167](https://challengedata.ens.fr/challenges/167); [benchmark notebook](https://github.com/phillipe-BAGUEKA/QRT-Asset-Allocation-Forecasting). Legitimate best reports are 51.22% public — [najibnasr](https://github.com/najibnasr/qrt-challenge-167) — and 52.4% on a 504-date holdout vs 50.6% always-up — [martindelamare](https://github.com/martindelamare/qrt-asset-allocation-forecasting)

### Inferences
- Every QRT benchmark uses standard sklearn, LightGBM, XGBoost or scikit-survival APIs, with NaN filled by 0 or the median and few features. A 4-hour team that has ready templates for (a) LightGBM/XGBoost with date-grouped CV, (b) Ridge/logistic, (c) Spearman and IPCW-C-index scorers and (d) a correct submission writer should match the benchmark within the first hour.
- In the market tasks the benchmark is near the noise floor. Leaderboard noise (≈±1.4 pt for #167, ±0.3 pt for #143) is larger than most model differences. Ranking will be driven by feature insight (cross-sectional/date-grouped features, structure or leak discovery), not model choice.

### Gaps
- No public benchmark score for #44, #72 or the #162 Cox model on the official leaderboard.
- The #23 benchmark notebook was not retrieved, so the exact "conditional mean" feature definition is unconfirmed.

## Q3. Recurring patterns across QRT challenges

### Takeaway
QRT's house style:
- Anonymised, shuffled date IDs and entity IDs.
- Lagged windows `RET_1..RET_20` (1 = most recent) and volume analogues.
- A sign-accuracy target, or a rank-based metric (Spearman, C-index, cosine).
- A two-column submission CSV (`ID`, target) with an example random submission and a benchmark notebook.
- Train/test splits by disjoint dates, periods or entities.

Anonymisation has repeatedly leaked ordering (#97 `ID`, #167 overlapping windows).

### Cited Findings
- **Shuffled, anonymised dates:**
  - #23: `DATE` "randomized and anonymized so there is no continuity" — [C23](https://challengedata.ens.fr/challenges/23)
  - #44: `ID_DAY` likewise — [C44](https://challengedata.ens.fr/challenges/44)
  - #97: `DAY_ID` anonymised — [C97](https://challengedata.ens.fr/challenges/97)
  - #167: `TS` DATE_xxxx "anonymized and shuffled" — [C167](https://challengedata.ens.fr/challenges/167)
- **Lag-window features:**
  - #23 `RET_1..RET_20`, `VOLUME_1..VOLUME_20` — [C23](https://challengedata.ens.fr/challenges/23)
  - #167 `RET_1..RET_20`, `SIGNED_VOLUME_1..20`, `MEDIAN_DAILY_TURNOVER`, `GROUP` — [C167](https://challengedata.ens.fr/challenges/167)
  - #72 uses a D = 250 lag window — [C72](https://challengedata.ens.fr/challenges/72)
- **Cross-sectional normalisation and grouping by date:**
  - #23 relative volume is de-meaned across stocks each day — [C23](https://challengedata.ens.fr/challenges/23)
  - The #167 benchmark builds `groupby('TS').transform('mean')` market-average features — [benchmark notebook](https://github.com/phillipe-BAGUEKA/QRT-Asset-Allocation-Forecasting)
  - #72 scores the cross-sectional cosine per day — [C72](https://challengedata.ens.fr/challenges/72)
- **Categorical hierarchies:** INDUSTRY/INDUSTRY_GROUP/SUB_INDUSTRY/SECTOR in #23 — [C23](https://challengedata.ens.fr/challenges/23); CLASS_LEVEL_1..4 in #44 — [C44](https://challengedata.ens.fr/challenges/44); GROUP in #167 — [C167](https://challengedata.ens.fr/challenges/167)
- **Metrics:**
  - Accuracy: #23, #143, #167
  - |return|-weighted accuracy: #44
  - Spearman: #97
  - Mean cosine similarity with a hard constraint: #72
  - IPCW C-index, τ = 7: #162
  - Sources: the respective pages [C23](https://challengedata.ens.fr/challenges/23), [C44](https://challengedata.ens.fr/challenges/44), [C72](https://challengedata.ens.fr/challenges/72), [C97](https://challengedata.ens.fr/challenges/97), [C143](https://challengedata.ens.fr/challenges/143), [C162](https://challengedata.ens.fr/challenges/162), [C167](https://challengedata.ens.fr/challenges/167)
- **Submission formats:**
  - #23: `ID,RET` (binary sign) — [C23](https://challengedata.ens.fr/challenges/23)
  - #44: `ID,RET_TARGET` (±1) — [C44](https://challengedata.ens.fr/challenges/44)
  - #97: `ID,TARGET` — [C97](https://challengedata.ens.fr/challenges/97)
  - #143: `ID,HOME_WINS,DRAW,AWAY_WINS` — [C143](https://challengedata.ens.fr/challenges/143)
  - #162: `ID` index + `risk_score` — [C162](https://challengedata.ens.fr/challenges/162). The benchmark notebook itself writes a Series named `OS_YEARS`, which conflicts with the page — [notebook](https://github.com/chenenen13/QRT-Leukemia-DataChallenge)
  - #167: ROW_ID-indexed 0/1 predictions, following `sample_submission.csv` — [C167](https://challengedata.ens.fr/challenges/167), [notebook](https://github.com/phillipe-BAGUEKA/QRT-Asset-Allocation-Forecasting)
  - Each page provides "an example submission file containing random predictions" — e.g. [C23](https://challengedata.ens.fr/challenges/23), [C97](https://challengedata.ens.fr/challenges/97)
- **Train/test split types:**
  - Random by day with no shared days: #44 — [C44](https://challengedata.ens.fr/challenges/44)
  - Different time periods: #97 — [C97](https://challengedata.ens.fr/challenges/97)
  - Unseen hospital: #162 — [lucaspaulien](https://github.com/lucaspaulien/qrt-leukemia-survival)
  - Strong train/test shift: #143 (adversarial AUC 0.98) — [louisrodembourg](https://github.com/louisrodembourg/qrt-football-prediction)
  - Test = most recent regime: #167 — [najibnasr](https://github.com/najibnasr/qrt-challenge-167)
- **Platform rules:**
  - Maximum **2 submissions per 24 h** per participant or team. Each submission is scored on public and private parts of the test set.
  - Ranking uses the best public submission by default. Private leaderboards are revealed 15 June and 15 December. The December private leaderboard decides awards.
  - The split fraction is not disclosed.
  - Source — [Challenge Data FAQ](https://challengedata.ens.fr/FAQ)
- **Leakage and structure discoveries:**
  - #97: the `ID` column preserves calendar order while `DAY_ID` is shuffled. FR and DE rows of the same day have identical features; only COUNTRY and TARGET differ — [ryan-racer](https://github.com/ryan-racer/QRT-Electricity-Forecasting)
  - #167: `RET_2` of day d matches `RET_1` of day d−1, so dates can be re-chained (4 chains of 1,780/348/256/130 days). The missing `SIGNED_VOLUME_1` follows a 5-day weekday cycle — [najibnasr](https://github.com/najibnasr/qrt-challenge-167)
  - #167: a date's target reappears as the next date's `RET_1` (88% sign agreement) — [martindelamare](https://github.com/martindelamare/qrt-asset-allocation-forecasting)
  - #167: a 1-NN on `ROW_ID` "scores 100% in-sample", which is memorisation — [martindelamare](https://github.com/martindelamare/qrt-asset-allocation-forecasting)

### Inferences
- Template checklist for a QRT-style 4-hour task:
  1. Parse the ID/date/entity columns and confirm which IDs are shuffled (test whether `ID` order or lag overlaps reveal chronology).
  2. Build cross-sectional per-date features (means, ranks, z-scores by date and group).
  3. Use GroupKFold over the date column, or over the entity/centre column when the test set is a new entity.
  4. Prepare scorers for accuracy, weighted accuracy, Spearman, IPCW C-index and cosine.
  5. Output a submission matching the sample file's index and column names exactly.
- Whether exploiting leaks is acceptable in a live QRT hackathon is unknown. Prepare the detection code, but check the event rules before relying on it.

### Gaps
- Public vs private split proportions are not published for any QRT challenge.
- Daily submission limits may differ at an in-person hackathon (unknown).

## Q4. What the challenge descriptions say about overfitting, cross-validation and external data

### Takeaway
The official pages say little. Only the football challenge explicitly bans external data ("can lead to disqualification"). Benchmarks show the preferred validation style: date-grouped KFold in #167, plain random holdouts in #143 and #162. Participants consistently report severe public/private and CV→leaderboard gaps.

### Cited Findings
- #143: "We expect from you to not use any external data, which can lead to disqualification." The data is also "exclusively intended for use in this challenge" under Sportmonks terms — [C143](https://challengedata.ens.fr/challenges/143)
- #23, #44, #72, #97, #162 and #167 pages contain no external-data clause — [C23](https://challengedata.ens.fr/challenges/23), [C44](https://challengedata.ens.fr/challenges/44), [C72](https://challengedata.ens.fr/challenges/72), [C97](https://challengedata.ens.fr/challenges/97), [C162](https://challengedata.ens.fr/challenges/162), [C167](https://challengedata.ens.fr/challenges/167). The FAQ is also silent on external data — [FAQ](https://challengedata.ens.fr/FAQ)
- #167 benchmark comment: "A quite large number of trees with low depth to prevent overfits". It uses `KFold(n_splits=8, shuffle=True)` applied to unique `TS` dates (date-grouped CV) — [benchmark notebook](https://github.com/phillipe-BAGUEKA/QRT-Asset-Allocation-Forecasting)
- #72 notes that random Stiefel sampling plus OLS is a "brute force" baseline. Participants saw extreme public/private reversals ("1st public LB → 81st private LB… 99th public LB → 1st private LB") — [PirashanthR](https://github.com/PirashanthR/QRT-Data-Challenge-2022)
- #167: participants found splitting a date across train/validation leaks the common market factor — [martindelamare](https://github.com/martindelamare/qrt-asset-allocation-forecasting). The benchmark's local 52.22% vs public 50.79% gap comes from non-stationarity; with purged recent-regime folds, local and public rankings matched (Spearman 0.82) — [najibnasr](https://github.com/najibnasr/qrt-challenge-167)
- #143: CV gains did not carry over to the online score because of train/test shift — [louisrodembourg](https://github.com/louisrodembourg/qrt-football-prediction)
- #162: leave-centres-out CV is wider but more honest than random K-fold (0.7220 ± 0.0077 vs 0.7246 ± 0.0033) — [lucaspaulien](https://github.com/lucaspaulien/qrt-leukemia-survival)
- Grand Data Challenge 2025 teams report that early validation scores "didn't translate to the leaderboard" because of leakage in preprocessing, and point to the importance of time-series CV — [malerbe](https://github.com/malerbe/QRT_challenge)

### Inferences
- For a 4-hour event with few submissions, validate on date-grouped (or entity-grouped) folds that mimic the test split. Do not chase the public leaderboard, and prefer shallow, regularised models.
- Assume external data is banned unless told otherwise, following QRT's only explicit statement (#143).

### Gaps
- No QRT page states the public/private percentage, a limit on final selected submissions, or rules on team size or code sharing.
- I found no published rules for QRT's in-person data hackathons (Grand Data Challenge 2025 or the ETH event).
