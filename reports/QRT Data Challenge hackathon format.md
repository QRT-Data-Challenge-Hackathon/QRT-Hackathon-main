# Expect a tabular sprint; validate by date

The QRT x Analytics Club ETH evening (Tue 24 Nov 2026, 18:00–22:00) will most likely be a team prediction sprint on one anonymised table built by QRT. You get `X_train`, `y_train` and `X_test` files and upload a two-column CSV (an `ID` plus a prediction), scored by a single metric on a hidden test set. The metric is most likely accuracy on the sign of next-day returns; the alternatives are Spearman correlation, 3-class accuracy and a C-index. Expect about three hours of actual competing inside the four-hour slot. That is an inference, not a published fact. The only official page confirms logistics and nothing about the problem, team size, platform, AI rules or prizes. QRT's whole autumn-2026 UK campus series also used a different format: "C3", a 2-hour HackerRank optimisation contest sponsored by OpenAI. The inference rests on QRT's seven ENS Challenge Data competitions. They share a stable house style: shuffled anonymous dates, `RET_1..RET_20` lag windows, group columns, and a benchmark near 51% accuracy. QRT has also reused those datasets at live student events. In the finance tasks, legitimate top entries beat the benchmark by only 1–2 accuracy points, which is less than public-leaderboard noise. Placings are therefore decided by process, not model choice. The process steps that matter are: validation that keeps each date's rows together, a few per-date features, an early valid submission, and suspicion of any big gain. For a first-time team with several AI coding agents, the plan that follows from this has four parts:

- One human owns the metric, the folds and the submit button.
- Agents work in separate lanes and only write predictions to disk.
- A trivial entry is scored on the leaderboard within 30 minutes, and a real date-grouped baseline within the first hour.
- Code freezes 20 minutes before the deadline.

In this report, cited statements are documented facts. Anything labelled *inference* or *my judgement* is reasoning from those facts.

## The official page confirms the logistics, not the problem

The event listing confirms the following ([Luma](https://luma.com/4k3p0fm5); [ACE events](https://www.analytics-club.org/upcoming-events)):

- **Format and time:** an in-person "Data Challenge Hackathon", **18:00–22:00 Zurich time**, at an ETH room still "to be confirmed" on the public page.
- **Registration:** **full, with 123 registered**. Every registrant commits to attending in person with their own laptop.
- **Data sharing:** registrants consent to their name, email, field of study and LinkedIn being shared with QRT "for recruiting purposes".
- **Food:** the form asks for dietary restrictions, so food is very likely served.
- **Teams:** sign-up is individual. There is no team-name field.
- **The challenge itself:** the description says only "a hands-on challenge" that tests "analytical skills". It names no dataset, metric, team size, submission platform, AI policy or prize.

No earlier QRT event at ETH appears anywhere in the research.

Two QRT campus formats ran in 2024–2026, and the ETH event could be either.

**Format A: data challenges.** Short tabular prediction contests on QRT datasets:

- A Paris event on 6 June 2024 used the football files of the then-live ENS challenge. One participant describes it as 5 hours in a team of three ([rwitt74](https://github.com/rwitt74/QRT_hackathon)). Another describes it as 4 hours, giving the team size as both six and three ([iljay](https://github.com/iljay/QRT_DataChallenge)).
- An IIT Delhi event in March 2025 asked for an alpha optimised for Sharpe ([manogyasingh](https://github.com/manogyasingh/qrt-hackathon)).
- An 8-day "Grand Data Challenge" in October 2025 had about 200 French engineering students predicting whether to short portfolio allocations, scored by accuracy ([iliasbendali](https://github.com/iliasbendali/QRT-Grand-Data-Challenge)).

**Format B: C3, "Code. Compute. Compete."** QRT's autumn-2026 UK programme at Imperial, Edinburgh and Warwick:

- A **2-hour HackerRank set of four optimisation problems plus a bonus**, in teams of 3–4.
- Every participant gets **$25 of OpenAI credits**, and the problems were "designed by QRT with the help of an LLM" ([UK Hackathons](https://ukhackathons.com/events/qrt-x-compsoc-c3-coding-hackathon); [UWCS](https://events.uwcs.co.uk/2026/1/7/codesoc+x+uwcs+x+qrt+hackathon/)).

*My judgement:* about three in four that ETH is Format A, about one in five that it is C3, and the remainder an open-ended analysis judged on a presentation.

- **For Format A:** the title says "Data Challenge" rather than "C3 Coding Hackathon". The host is an analytics club. The page mentions neither OpenAI credits nor teams of 3–4, and every C3 listing advertises both.
- **For C3:** it is QRT's current campus product. A 4-hour slot also fits the C3 pattern of briefing, contest, then results.

The table separates what is known from what is guessed.

| Question | Status | Best answer now |
|---|---|---|
| When and where | **Confirmed** | Tue 24 Nov, 18:00–22:00, ETH Zürich, in person, own laptop ([Luma](https://luma.com/4k3p0fm5)) |
| Who sees your data | **Confirmed** | QRT, for recruiting ([Luma](https://luma.com/4k3p0fm5)) |
| Problem type | Inference | Tabular prediction on an anonymised QRT dataset; C3-style optimisation is the main alternative |
| Metric | Inference | Accuracy on a sign or class; otherwise Spearman, C-index or Sharpe |
| Platform | Inference | CSV upload to a Kaggle-style page with a hidden test set. QRT called its Oct 2025 Paris finale its "very first in-person Kaggle Competition" ([QRT LinkedIn post, via search snippet](https://www.linkedin.com/posts/qube-rt_last-weekend-qrts-quantitative-researchers-activity-7383847260226342914-XbBW)) |
| Public/private leaderboard split | Inference, likely | The Paris QRT × TBF football edition ranked teams on both public and private scores ([ayoub-mg](https://github.com/ayoub-mg/QRT-Football-Challenge)) |
| Team size | Unknown | Past events used 3–4 (C3) and 3 (Paris); teams may be formed on the night |
| AI tools | Unknown, likely allowed | Sponsored at C3; ETH page silent |
| Prizes | Unknown | C3 gave AirPods, QRT poker sets, a £100 voucher, a London Grand Final invite and an interview fast-track ([UK Hackathons](https://ukhackathons.com/events/qrt-x-compsoc-c3-coding-hackathon)) |

The evening will probably run like Edinburgh's C3 night: **18:00 arrival and briefing, 18:30–20:30 competition, 20:30–21:00 results, prizes and networking with QRT** ([UK Hackathons](https://ukhackathons.com/events/qrt-x-compsoc-c3-coding-hackathon)). Stretched to four hours, that suggests roughly a 20-minute briefing, a contest of about three hours, and 30–40 minutes of results and recruiting conversation. With 123 registrants in teams of three or four, expect 30–40 teams. Both points are inference.

## QRT's seven ENS challenges reveal one house style

QRT has published one data challenge per season on the ENS/Institut Louis Bachelier "Challenge Data" platform since 2020. Each comes with:

- a benchmark notebook,
- an example submission,
- a two-column-style CSV output format.

The 2025 season was extended through all of 2026 with QRT's asset-allocation task, and a normal season is planned for 2027 ([Institut Louis Bachelier](https://www.institutlouisbachelier.org/en/on-the-program/actualites/data-challenge-a-2025-season-extended-to-2026-and-a-new-challenge-signed-by-qrt)). The full catalogue is the best map of what QRT puts in front of students.

| ENS # (year) | Task | Metric | Official benchmark | Best reported legitimate score |
|---|---|---|---|---|
| [23](https://challengedata.ens.fr/challenges/23) (2020) | Sign of a US stock's next-day residual return | Accuracy | 51.31% (Random Forest) | 0.5287 winner, participant-reported ([lbrilh](https://github.com/lbrilh/QRT-StockReturnPrediction)) |
| [44](https://challengedata.ens.fr/challenges/44) (2021) | Sign of liquid-asset returns from 100 illiquid assets | \|return\|-weighted accuracy | Most-correlated-asset regression (no score published) | None published |
| [72](https://challengedata.ens.fr/challenges/72) (2022) | Learn 10 orthonormal return factors | Mean cosine similarity (−1 if not orthonormal) | Best of 1,000 random matrices | ≈0.149 training metric ([mrmrob003](https://github.com/mrmrob003/qrt-data-challenge-2022)) |
| [97](https://challengedata.ens.fr/challenges/97) (2023) | "Explain" daily FR/DE electricity futures moves | Spearman | 15.86% (linear regression) | 0.5658, rank 6 of 1,157, using a recovered-date-order trick ([ryan-racer](https://github.com/ryan-racer/QRT-Electricity-Forecasting)) |
| [143](https://challengedata.ens.fr/challenges/143) (2024) | Football: home win / draw / away win | Accuracy | ≈47.5% (gradient boosting) | 0.49–0.50 top of leaderboard ([louisrodembourg](https://github.com/louisrodembourg/qrt-football-prediction)) |
| [162](https://challengedata.ens.fr/challenges/162) (2025) | Leukaemia patient survival ranking | IPCW C-index, τ = 7 years | Cox model, ~0.68 on a holdout | 0.7541 on submission ([coquetl](https://github.com/coquetl/Myeloid-Leukemia-Prediction)) |
| [167](https://challengedata.ens.fr/challenges/167) (2026) | Sign of a long/short allocation's next-day return | Accuracy | 0.5079 (LightGBM) | ≈0.512 public ([najibnasr](https://github.com/najibnasr/qrt-challenge-167)); 52.4% on a 504-date holdout ([martindelamare](https://github.com/martindelamare/qrt-asset-allocation-forecasting)) |

The house style is consistent enough to plan around. Its main features:

- **Shuffled, anonymised date IDs.** #23's `DATE` is "randomized and anonymized so there is no continuity". #167's `TS` values (DATE_0001…) are "anonymized and shuffled".
- **Wide lag windows.** `RET_1..RET_20` and a volume analogue, where `RET_1` is yesterday ([C23](https://challengedata.ens.fr/challenges/23); [C167](https://challengedata.ens.fr/challenges/167)).
- **Group columns.** `SECTOR`/`INDUSTRY` hierarchies in #23, `CLASS_LEVEL_1..4` in #44, `GROUP` in #167.
- **Train/test splits that deliberately differ.** Disjoint days (#44), different time periods (#97), a hospital never seen in training (#162), and a test set resembling the most recent market regime (#167) ([C44](https://challengedata.ens.fr/challenges/44); [lucaspaulien](https://github.com/lucaspaulien/qrt-leukemia-survival); [najibnasr](https://github.com/najibnasr/qrt-challenge-167)).
- **Accuracy edge cases.** In #167 a zero return counts as class 0, and the target is given as the real-valued return, not just its sign ([C167](https://challengedata.ens.fr/challenges/167)).
- **External data.** QRT banned it once, in #143, warning that using it "can lead to disqualification" ([C143](https://challengedata.ens.fr/challenges/143)).

Datasets also migrate between QRT's public challenges and live events:

- The Paris 2024 hackathon notebook loads the exact #143 files (`train_home_team_statistics_df.csv`, `Y_train_1rknArQ.csv`) while that ENS challenge was still running ([rwitt74](https://github.com/rwitt74/QRT_hackathon)).
- The October 2025 Paris finale used the asset-allocation data that went public as ENS #167 in January 2026 ([malerbe](https://github.com/malerbe/QRT_challenge)).
- A March 2026 "QRT hackathon" repo shows the same `ROW_ID, ALLOCATION, TS, RET_1..RET_20` layout ([ferdinandfd](https://github.com/ferdinandfd/QRT-Hackathon)).

*Inference:* the ETH task is therefore one of two things:

1. **A reuse of a public dataset.** #167 is the obvious candidate because it is live, with #23 or #143 as alternatives.
2. **A fresh dataset in the same style.** Possibly the one QRT needs for the 2027 ENS season, debuted live the way the Oct 2025 data was.

In order of likelihood, expect:

1. A cross-sectional return-sign table (rows = date × asset, `RET_1..20`, a group column, accuracy).
2. A regression or ranking table scored by Spearman or C-index.
3. A non-finance multi-table problem like football or survival.

On platform rules: ENS itself allows **2 submissions per 24 hours** and scores each on public and private parts of the test set, with the private part deciding awards ([Challenge Data FAQ](https://challengedata.ens.fr/FAQ)). No source documents the submission cap, the public/private share or tie-breakers for any in-person QRT data event. Edinburgh C3 teams could not find these rules in advance either; they wrote down "Does the judge count the best submission or the last one?" and "Is there a limit on submissions?" as briefing questions ([CooperMcCloskey](https://github.com/CooperMcCloskey/QRT-Hackathon)).

## Winners added one to two points with features and validation, not fancier models

In every QRT challenge with a documented top solution, the edge came from understanding how the data was built and from validation that mirrors the test split. The model was ordinary: shallow gradient-boosted trees, ridge or logistic regression, and an equal-weight blend. How big the edge is depends on the domain.

**Finance tasks: gains are small.**

- In 2020 the reported winner scored **0.5287 against a 0.5131 benchmark** ([lbrilh](https://github.com/lbrilh/QRT-StockReturnPrediction)).
- In #167 the best documented legitimate entries sit **0.4–1.5 points above a 0.5079 benchmark**, and an always-long baseline already scores about 50.6% ([martindelamare](https://github.com/martindelamare/qrt-asset-allocation-forecasting)).

**Non-finance tasks: structure and domain features move scores a lot.**

- Electricity Spearman went from **0.16 to 0.57** ([ryan-racer](https://github.com/ryan-racer/QRT-Electricity-Forecasting)).
- Leukaemia C-index went from **~0.68 to ~0.72–0.75** ([lucaspaulien](https://github.com/lucaspaulien/qrt-leukemia-survival); [coquetl](https://github.com/coquetl/Myeloid-Leukemia-Prediction)).

Five techniques recur across the winning write-ups.

**1. Group cross-validation folds by date.** QRT's own benchmarks do this: the 2020 and 2026 notebooks assign whole dates to folds, so rows from one day never sit in both training and validation ([JeremyKulcsarDS](https://github.com/JeremyKulcsarDS/QRT_Challenge_2020_Stock_Return_Prediction); [#167 benchmark copy](https://github.com/phillipe-BAGUEKA/QRT-Asset-Allocation-Forecasting)). Assets on the same day share a market move. A random split lets the model learn that day's move from its neighbours, so the validation score is inflated.

**2. Build per-date cross-sectional features.** These compare each row with the other rows on the same date:

- The conditional mean of past returns by (sector, date) is part of the 2020 benchmark itself ([C23](https://challengedata.ens.fr/challenges/23)).
- A 12th-of-207 entry turned predicted probabilities into labels **per date**: it predicted "up" for the top half of each day ([JeremyKulcsarDS](https://github.com/JeremyKulcsarDS/QRT_Challenge_2020_Stock_Return_Prediction)).
- The strongest documented #167 pipeline used **18 within-date features**: rank, distance to the day's median, and robust z-score ([martindelamare](https://github.com/martindelamare/qrt-asset-allocation-forecasting)).

**3. Treat the entity ID as a category.** In #167 the largest single gain was declaring `ALLOCATION` a categorical feature instead of parsing the number out of "ALLOCATION_123": **+0.44 points, significant on 11 of 12 folds**. Optuna hyperparameter tuning in the same study added a non-significant +0.05 points ([hobbit-joufflu RESULTS](https://github.com/hobbit-joufflu/Data-Challenge-2026/blob/main/RESULTS.md)).

**4. Shape the target to the metric.**

- The #167 benchmark regresses the real-valued return and takes its sign, rather than training a classifier ([benchmark copy](https://github.com/phillipe-BAGUEKA/QRT-Asset-Allocation-Forecasting)).
- For electricity's Spearman metric, rank-transforming the target lifted cross-validated correlation **from 0.236 to 0.290** at no cost ([ryan-racer](https://github.com/ryan-racer/QRT-Electricity-Forecasting)).
- What hurt: in #167, de-meaning the target by day lost 0.34–0.43 points, because it threw away the market drift that the always-long baseline captures ([hobbit-joufflu RESULTS](https://github.com/hobbit-joufflu/Data-Challenge-2026/blob/main/RESULTS.md)).

**5. In non-finance problems, encode the domain.**

- Football: top entries aggregated player statistics by position and added a predicted goal difference ([hugopuybareau](https://github.com/hugopuybareau/QRT_DATA_CHALLENGE); [louisrodembourg](https://github.com/louisrodembourg/qrt-football-prediction)).
- Leukaemia: a re-implemented clinical risk score (IPSS-M) alone reached **0.7086**, close to the 10-model ensemble's **0.7238** ([lucaspaulien](https://github.com/lucaspaulien/qrt-leukemia-survival)).

**What did not pay.** Elaborate machinery rarely helped:

- PCA caused "a significant drop" in football ([ms2621](https://github.com/ms2621/QRT_Football)).
- A deep survival network lost to Cox and random survival forests ([MaxCoppa](https://github.com/MaxCoppa/Myeloid-Leukemia-Survival-Prediction)).
- Per-allocation models and blending were dropped as "less performant" at the 2025 Grand Data Challenge ([iliasbendali](https://github.com/iliasbendali/QRT-Grand-Data-Challenge)).
- On electricity's raw features, "every flexible model lost to ridge" until better features existed ([ryan-racer](https://github.com/ryan-racer/QRT-Electricity-Forecasting)).

**What did help a little.** Blending gives small, reliable gains:

- In #167, an equal-weight average of logistic regression, ridge and LightGBM beat the best single model by **+0.16 points** ([martindelamare](https://github.com/martindelamare/qrt-asset-allocation-forecasting)).
- In electricity, adding LightGBM and then CatBoost to ridge raised Spearman from 0.558 to 0.572 ([ryan-racer](https://github.com/ryan-racer/QRT-Electricity-Forecasting)).

The in-person results tell the same story:

- At the Paris 2024 event a team of three used LightGBM multiclass with light tuning ([rwitt74](https://github.com/rwitt74/QRT_hackathon)).
- At IIT Delhi, **5th of 190+** came from a one-line correlation-weighted formula ([manogyasingh](https://github.com/manogyasingh/qrt-hackathon)).
- Grand Data Challenge placings ranged from 13th of 200 with LightGBM to 31st with plain logistic regression ([leclercrafael](https://github.com/leclercrafael/QRT-Asset-Allocation-Performance-forecasting); [iliasbendali](https://github.com/iliasbendali/QRT-Grand-Data-Challenge)).

*Inference:* a team that, within three hours, (a) reproduces the benchmark, (b) validates by date and (c) adds per-date features, an entity category and a three-model blend has done what the documented top finishers did.

## Leaks and leaderboard noise decide more placings than models

Most lost placings in QRT-style contests come from fooling yourself, not from weak models. Four mechanisms recur.

**1. Validation that leaks.** A participant who used random stratified folds on the 2020 data reported a cross-validated **0.5408**, well above the **0.5287** winning leaderboard score ([chrxs18](https://github.com/chrxs18/qrt_challenge)). Preprocessing leaks do the same damage:

- A Grand Data Challenge team saw "impressive validation scores that didn't translate to the leaderboard" because of "subtle data leakage issues in our preprocessing pipeline" ([malerbe](https://github.com/malerbe/QRT_challenge)).
- A 2026 hackathon team computed target encoding before splitting folds and flagged its own CV number as untrustworthy ([ferdinandfd](https://github.com/ferdinandfd/QRT-Hackathon)).
- Even QRT's honest #167 benchmark drops from **52.22% locally to 50.79% public**, because the test set comes from a later regime ([C167](https://challengedata.ens.fr/challenges/167); [benchmark copy](https://github.com/phillipe-BAGUEKA/QRT-Asset-Allocation-Forecasting)).
- One participant fixed that gap: scoring cross-validation only on "test-like" dates (those with as many allocations as test days) predicted the private score **within 0.18 points** ([hobbit-joufflu RESULTS](https://github.com/hobbit-joufflu/Data-Challenge-2026/blob/main/RESULTS.md)).

**2. Leaderboard noise and shake-ups.** Public-leaderboard noise is about **±0.3 points** on 25,000 football matches ([louisrodembourg](https://github.com/louisrodembourg/qrt-football-prediction)) and about **±0.5 points** for #167 ([hobbit-joufflu RESULTS](https://github.com/hobbit-joufflu/Data-Challenge-2026/blob/main/RESULTS.md)). Those are the same size as real model differences. Shake-ups between public and private rankings have been brutal:

- In #72, "1st public LB → 81st private LB" and "99th public LB → 1st private LB" ([PirashanthR](https://github.com/PirashanthR/QRT-Data-Challenge-2022)).
- An electricity entry fell from 0.295 public to 0.251 private ([chantomkit](https://github.com/chantomkit/QRT2023)).

My own arithmetic: an accuracy near 50% measured on n test rows has a standard error of about √(0.25/n). For a 10,000-row public split that is ±0.5 points, and correlated same-day rows make it worse. **A public-leaderboard gain under about one point is not evidence on its own.**

**3. Structure leaks in QRT's anonymisation.**

- In electricity, `DAY_ID` was shuffled but `ID` kept calendar order. Recovering that order lifted one entry from 0.29 to 0.52 ([ryan-racer](https://github.com/ryan-racer/QRT-Electricity-Forecasting)).
- In #167, one date's `RET_2` equals the previous date's `RET_1`, so dates can be re-chained ([najibnasr](https://github.com/najibnasr/qrt-challenge-167)).
- Also in #167, a date's target reappears as the next date's `RET_1` with **88% sign agreement**, and a nearest-neighbour model on `ROW_ID` scores 100% in-sample ([martindelamare](https://github.com/martindelamare/qrt-asset-allocation-forecasting)).
- Public #167 scores near **81%** "cannot come from predicting daily return signs" ([najibnasr](https://github.com/najibnasr/qrt-challenge-167)). One entry reached **77.41%** by reconstructing history from overlapping windows and public prices ([RyanBalech](https://github.com/RyanBalech/qrt-asset-allocation-reconstruction)).

Top entrants disagreed on whether such tricks are fair: one built on recovered order, another deliberately excluded it. At a recruiting event judged by QRT researchers, ask before you exploit anything.

**4. Process failures.**

- An Edinburgh C3 team logged, with minutes left, that "nothing [was] confirmed as submitted yet" ([CooperMcCloskey](https://github.com/CooperMcCloskey/QRT-Hackathon)).
- QRT formats require exact columns, such as exactly `ID` and `RET` in #23 ([C23](https://challengedata.ens.fr/challenges/23)).
- Documented #167 bugs ([hobbit-joufflu RESULTS](https://github.com/hobbit-joufflu/Data-Challenge-2026/blob/main/RESULTS.md)):
  - imputing missing values before building features, which made the NaN-aware aggregations useless;
  - a robust z-score blowing up to ~1e9 on a day with zero spread;
  - `import torch` before LightGBM crashing the process.
- On macOS, LightGBM needs `brew install libomp` ([TomRodger](https://github.com/TomRodger/qrt-hackathon)).

The table turns these lessons into reflexes for a novice team.

| What you see | What it usually means | What to do |
|---|---|---|
| CV far above the leaderboard | Same-date rows split across folds, or a scaler/encoder fitted outside the folds | Regroup folds by date; refit every preprocessing step inside each fold |
| One feature adds more than ~1 point on a sign task | A leak (ID order, row index, a target-derived column, overlapping lags) | Stop, audit, ask the organisers before using it |
| The leaderboard jumps but CV does not | Noise | Ignore it; select on CV |
| A train-vs-test classifier separates the sets easily (AUC ≫ 0.5; football hit 0.98 per [louisrodembourg](https://github.com/louisrodembourg/qrt-football-prediction)) | The test set comes from a different regime, entity or period | Validate on the test-like subset or leave whole groups out |
| Predictions are all one class, or contain NaN | A broken pipeline | Check against the sample submission before every upload |

## Spend the evening on a fixed clock, with agents in lanes

### Use agents as fast hands, never as the judge

AI coding agents demonstrably help in QRT's own events:

- The solo Imperial C3 winner placed **1st of 68 teams (~240 people)**. He ran **15 Claude Code and 7 Codex sessions in parallel**, "wrote little code by hand", and built a fast local scorer first ([iamagoodacc C3 brief](https://github.com/iamagoodacc/qrt-hackathon-2026)).
- He found that "several independent attempts at a problem, keeping the best, beat one bigger attempt". Each agent followed a fixed protocol: restate the problem, verify the answer, and replace the saved best only on improvement ([iamagoodacc C3 brief](https://github.com/iamagoodacc/qrt-hackathon-2026)).
- On Kaggle, a March 2026 competition was won with three LLMs writing 850 experiments, each saving its out-of-fold and test predictions to disk ([NVIDIA](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/)).

Unsupervised agents are weak, though:

- On MLE-bench the best agent reached a Kaggle bronze medal in only **16.9%** of 75 competitions ([MLE-bench](https://arxiv.org/pdf/2410.07095)).
- A 2025 review found no published case of autonomous agents succeeding in live competitions ([ML Contests](https://mlcontests.com/state-of-machine-learning-competitions-2025/)).

Agents also make exactly the mistakes in the leak table above, and they make them faster than people can audit.

*Recommendation*, for a team of three or four:

- **Captain.** Owns `competition.toml`, the submission check, the run log and final selection, and is **the only person who uploads**.
- **Validator.** Owns folds and the metric, runs the leak and shift audits, and must approve any claimed gain before it is merged.
- **One or two lane owners.** Each steers two or three agents in separate git worktrees.

Rules for agents:

- They never edit the fold, metric or submission code.
- They get small, testable tasks such as "add per-date ranks of RET_1..5 and report 5-fold CV mean ± std against run X". Never "improve the score".
- Each run saves out-of-fold and test predictions.
- They never write the final CSV.
- Any gain more than about twice the typical step counts as a leak until the validator clears it.

### Tonight's preparation

The repo's kit already covers the ENS shapes. It has rehearsal configs for `stock_sign`, `allocation`, `electricity`, `football` and `survival`. The CLI provides `eda`, `baseline`, `blend` (mean, rank or hill-climb), `check-sub`, `record-lb` and `lb`, and a synthetic LightGBM baseline is already logged. Before tomorrow:

1. Run every rehearsal shape end-to-end on **every laptop**: `uv run qrtc synth <kind>`, point `competition.toml` at it, then run `qrtc eda`, `qrtc baseline`, `qrtc check-sub` and `qrtc blend`.
2. Fix or ignore the `make rehearse` / `make synth` references in `competition.toml` and `data/README.md`. The repo has no Makefile, so those commands will fail.
3. If time allows, the highest-value extra step: register on challengedata.ens.fr and run the kit on the real #167 data. That shows the team what genuine 52%-accuracy noise feels like, and it is a dress rehearsal in case ETH reuses that dataset.
4. Keep a tiny C3 fallback: a local-scorer template plus a greedy and simulated-annealing skeleton.
5. Logistics: chargers and a power strip, a phone hotspot, agent logins tested, and `libomp` installed on Macs.

### Minute-by-minute plan

The plan assumes the data drops at about 18:20 and the deadline is about 21:20, a 180-minute contest. Anchor the opening to data release and the ending to the announced deadline.

- **If the clock is longer,** stretch only the lane window.
- **If it is shorter,** cut lane time. Never cut the first submission, the baseline or the freeze.

| Clock | Contest time | What happens | Done when |
|---|---|---|---|
| 17:30–18:00 | Before | Arrive early and sit together near power. Kit opens and `qrtc baseline` runs on synthetic data on every laptop; agents logged in | Every machine green |
| 18:00–18:20 | Briefing | Captain asks the briefing questions (below); validator writes the rules down verbatim. Decide Plan A (one dataset + CSV) or Plan B (C3) | Rules file written |
| 18:20–18:30 | T+0–10 | Everyone reads the statement, metric definition, sample submission and rules. **Nobody codes.** Captain puts the data in `data/raw/` | One agreed sentence: "we predict X per ID, scored by Y; test differs from train by Z" |
| 18:30–18:40 | T+10–20 | Validator copies the closest rehearsal config into `competition.toml` (ID, target, date, group, metric, CV) and merges it. Lane owner 1 runs `qrtc eda` (rows, dates, entities per date, base rate, NaN, train/test date overlap, ID ordering). Lane owner 2's agent runs a train-vs-test classifier | Spec merged; data structure understood |
| 18:40–18:50 | T+20–30 | Captain uploads a trivial entry (majority class, or the benchmark replicated if a notebook is given) through `qrtc check-sub` | **A scored entry exists** |
| 18:50–19:15 | T+30–55 | `qrtc baseline` for LightGBM and ridge/logistic with date-grouped CV. Captain submits the better one and records its leaderboard score | Safety net on the board; CV within noise of the leaderboard (if not, fix folds first) |
| 19:15–19:20 | T+55–60 | Stand-up: assign four lanes, each with a written task | Each lane knows its task |
| 19:20–20:35 | T+60–135 | **Lanes run in parallel.** **A:** per-date ranks, z-scores, de-meaned and group-relative features. **B:** entity as a category, rolling volatility, momentum and reversal, target treatment (regress vs classify; rank target for rank metrics; domain encodings if non-finance). **C:** model diversity (CatBoost or XGBoost, ridge on ranks, 3 seeds). **D:** validator's leak and shift audit, plus review of every claimed gain | Each run logged with CV mean ± std versus baseline |
| 19:50 and 20:20 | T+90, T+120 | Three-minute check-ins. Captain merges features that beat the baseline by more than ~1 standard error on most folds. Submit a milestone only if CV improved and the cap allows | Kept features merged |
| 20:35–20:55 | T+135–155 | Consolidate: final feature set, 3–5 seeds of the best GBDT, `qrtc blend` (rank or hill-climb) over saved runs. Pick **two finals**: the best-CV blend and the simplest near-best model | Two candidates chosen |
| 20:55–21:00 | T+155–160 | Generate both files. Run `qrtc check-sub`, then check row count, ID set, dtype and class balance, and correlate against the last good file | Both pass |
| 21:00 | T+160 (T−20) | **Captain uploads both and selects them if the platform asks. Code freeze.** | Both scored on the board |
| 21:00–21:15 | T−20 to T−5 | Verify. Resubmit the safety net if anything failed. If there is judging, prepare three slides: validation design, what worked, a CV-vs-leaderboard table | Nothing new started |
| 21:15–21:20 | T−5 | Final confirmation of the selected submissions | Laptops closed |
| 21:20–22:00 | After | Results. Talk to QRT researchers about how the dataset was built | — |

### If the briefing reveals C3 instead

Switch to the C3 playbook that Edinburgh teams wrote after studying the Imperial winner:

- Everyone reads every problem in the first 10 minutes, then each problem gets **one owner**.
- Build a local scorer per problem before optimising.
- Get a **valid answer on every problem by about minute 40**, because an invalid answer scores 0.
- Improve with greedy search, then simulated annealing or large-neighbourhood search, running several independent agent attempts and keeping the best.
- Stop starting new approaches 20 minutes before the end, and confirm best submissions 5 minutes before ([CooperMcCloskey](https://github.com/CooperMcCloskey/QRT-Hackathon); [iamagoodacc](https://github.com/iamagoodacc/qrt-hackathon-2026)).
- Watch the bonus gate: at Edinburgh the bonus counted only if the team scored **strictly above 70% on each** main problem ([CooperMcCloskey](https://github.com/CooperMcCloskey/QRT-Hackathon)).

### Questions to ask at the briefing

| Ask | Why it matters | How the answer changes the plan |
|---|---|---|
| Is the final ranking on a private part of the test set, and what share is public? | QRT contests have had 1st→81st shake-ups | If private, select on CV and ignore small leaderboard moves |
| How many submissions per team, and does the final count the best, the last or the ones we select? What is the exact cutoff? | Edinburgh teams could not find this in advance | Sets the milestone-submission budget and when the captain uploads |
| How are teams formed, and what is the maximum size? | Sign-up was individual | Assigns the captain, validator and lane roles |
| Are AI coding assistants and agents allowed, from any provider, and must use be disclosed? | Sponsored at C3; ETH is silent | If banned, run the kit by hand and cut to two lanes |
| May we use code written before tonight, including work done on public ENS QRT datasets? | The kit and any #167 rehearsal depend on it | Decides whether pre-built pipelines are usable |
| Is external data (for example public market prices) allowed? | #143 banned it; a #167 exploit relied on it | Default to no |
| If we find hidden structure, such as recoverable date order or overlapping lags, may we use it? | Such leaks recur in QRT data | If no, the validator uses leak detection only to keep CV honest |
| Is there any presentation or code review, and what do top teams win (for example an interview fast-track)? | An Algothon team topped QRT's score but "messed up the final presentation" ([MorganREN](https://github.com/MorganREN/Algothon_IC_2023)) | Whether to assign a presenter at T−20 |

## Conclusion

The research changes what "winning" means for a first-time team. In QRT's finance problems the achievable edge is one or two accuracy points, smaller than the leaderboard's own noise. So the field's ordering will be set mostly by mistakes: leaky folds, late or malformed submissions, and finals chosen from a lucky public score. That is good news for a novice team. Avoiding those mistakes is a matter of discipline the team can rehearse tonight, not expertise it lacks. With several AI agents the main risk is not speed. It is that agents can produce convincing, leaky gains faster than any human can check them, so the validator's veto is worth more than an extra agent.

Two strategic implications follow. First, the dataset may well be a preview of QRT's next public ENS challenge or a reuse of #167. A practised pipeline, plus a clear answer at the briefing on whether hidden structure may be exploited, could matter more than any modelling idea. Second, the event is explicitly a recruiting channel, with registrant data going to QRT. A clean method the team can explain (date-grouped validation, why a given feature works, why the final was chosen on CV) is worth having ready for the networking half-hour, whatever the final rank.
