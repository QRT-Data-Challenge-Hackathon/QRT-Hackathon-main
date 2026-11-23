# Playbook for Short (3–6 h) Quant/Finance Data-Science Hackathons (prep for the QRT Data Challenge Hackathon, ETH Zürich, 24 Nov 2026)

Research date: 23 Nov 2026. Sources are mainly winning write-ups, organiser pages and practitioner blogs. Several primary pages (Kaggle discussion threads, Kaggle docs, Citadel careers page) could not be fetched because they are JS-rendered or return 403. Where a point comes from my own background knowledge rather than a fetched source, it sits under **Inferences** and is labelled "(unverified / from background knowledge)".

---

## 1. How comparable events are run (duration, team size, scoring, submission limits, AI rules)

### Takeaway
QRT's public data challenges have used the same pattern for years: an anonymised tabular dataset (date and asset IDs are indices, and dates are often shuffled), an `ID` + prediction CSV submission, one simple metric (accuracy of the return sign, Spearman, or IPCW C-index), and a weak benchmark (about 51% accuracy). A 4-hour QRT hackathon will most likely be a scaled-down version of this, ranked on a leaderboard. Citadel/Correlation One datathons are a different format: judged reports and presentations. Kaggle finance competitions give the technical lessons but run for months.

### Cited Findings

**QRT's own data challenges (hosted on ENS ChallengeData), the closest proxy for tomorrow's dataset**
- QRT "Stock return prediction" (ChallengeData #23, 2020): predict the sign of a US stock's one-day *residual* return from its last 20 days. The page says the signal-to-noise ratio is "extremely low". Columns: `ID, DATE, STOCK, INDUSTRY, INDUSTRY_GROUP, SUB_INDUSTRY, SECTOR, RET_1..RET_20, VOLUME_1..VOLUME_20`. Train has 418,595 rows and test 198,429. `DATE` is "randomized and anonymized" with "no continuity or link between any dates". Volume is normalised by its 20-day median and then **demeaned across all stocks on the same day**, so it is cross-sectional. Metric: accuracy of the predicted sign. Benchmark: Random Forest with 500 trees and max depth 8, using 5 recent returns/volumes plus sector-conditional mean features, NaN filled with 0. Benchmark public score: **51.31%**. Submission: exactly two columns, `ID` and `RET`. — [ChallengeData #23](https://challengedata.ens.fr/challenges/23)
- QRT "asset allocation" challenge (ChallengeData #167, started 22 Jan 2026): predict the sign of an allocation's next-day return (1 = long, 0 = short). Multi-index (date, allocation) with a unique `ROW_ID`. Columns: `TS`, `ALLOCATION`, `RET_1..RET_20`, `SIGNED_VOLUME_1..20`, `MEDIAN_DAILY_TURNOVER`, `GROUP`, and `TARGET` as a real-valued return. "Dates were anonymized and shuffled so there is no guarantee of continuity." X_train has 527,073 rows and X_test 31,870. Metric: plain accuracy with sign(x)=1 if x>0 else 0, so a **zero return counts as class 0**. Benchmark: **LightGBM with CV = 0.5079** public, plus a Ridge model. Benchmark features: per-allocation average past performance over several windows, the **same averages across all allocations** (cross-sectional), 20-day volatility, and cross-allocation average volatility. — [ChallengeData #167](https://challengedata.ens.fr/challenges/167)
- QRT "Can you explain the price of electricity?" (ChallengeData #97): target is the daily price variation of 24h baseload futures (France/Germany). Metric is **Spearman correlation**. Output is `ID, TARGET`. About 1,494 train rows and 654 test rows with 35 input columns. A linear-regression benchmark scored 15.86% public. One participant scored 0.295 public but **0.251 private**. Another noted it is an *explanation* problem (contemporaneous variables), not forecasting. — [ChallengeData #97](https://challengedata.ens.fr/challenges/97); [chantomkit/QRT2023 GitHub](https://github.com/chantomkit/QRT2023)
- QRT survival-prediction challenge (ChallengeData #162, myeloid leukaemia, with Gustave Roussy): data from 24 clinical centres. QRT provides a Cox Proportional Hazards benchmark. The metric is **IPCW C-index**, with about 3,323 train and 1,193 test patients. There is no cash prize. — [eFinancialCareers](https://www.efinancialcareers.com/news/the-unusual-hedge-fund-quant-challenge-with-no-trading-involved); [ChallengeData #162](https://challengedata.ens.fr/challenges/162); [Thierry Moudiki benchmark blog](https://thierrymoudiki.github.io/blog/2025/02/10/python/Benchmark-QRT-Cube)
- ChallengeData platform rule: the metric cannot be changed once a challenge has started. — [ChallengeData technical annex](https://challengedata.ens.fr/static/Annexe_tech_ChallengeData_en.pdf)
- There appears to have been an Imperial SIF "QRT Quantitative Competition" (LinkedIn post), but I could not find its format or rules. — [LinkedIn search result](https://gr.linkedin.com/in/politis-george-39aa845)

**Citadel / Correlation One Datathons (judged, report-based)**
- Format: teams get a large, complex dataset and present findings to a panel of judges. Correlation One handles recruiting, including a 60-minute entry assessment. — [Citadel datathons page (search snippet; page returned 403 on fetch)](https://www.citadel.com/careers/programs-and-events/datathons)
- Deliverables have included a ~20-page technical report with graphs and code plus a PowerPoint. In 2018 a panel of **ten judges** chosen by Citadel and Correlation One questioned students. Assessment covered the application, the entry exam, collaboration, creativity, rigour and technical ability. — [The Hedge Fund Journal](https://thehedgefundjournal.com/citadel-s-data-open-showcases-innovative-recruitment-approach/); [Citadel PDF](https://www.citadel.com/wp-content/uploads/2018/03/Citadels_Data_Open_showcases_innovative_recruitment-approach.pdf)
- One European regional event was described as teams of 4 producing the best report they can in about a week. The grand prize at the 2017 Data Open final was $100K. — [search summary of participant blog (page 404 on fetch)](https://anonymous3141.github.io/2022/03/14/Winning-The-European-Regional-Datathon.html); [CMU news](https://www.cmu.edu/dietrich/news/news-stories/2017/march/datathon-results.html)

**Kaggle / sponsor finance competitions (months-long, leaderboard-scored)**
- G-Research Crypto Forecasting (Nov 2021–May 2022): 14 coins, **weighted Pearson correlation** metric. Each team **selected its two best models**, which were then run on **3 months of live data**. The $125K prize pool was shared by the top 10. **All top 3 used LightGBM.** "All of our top competitors believed it was their feature engineering that contributed most to their wins." The leaderboard "saw many big jumps and precipitous drops". — [G-Research wrap-up](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/)
- Jane Street Real-Time Market Data Forecasting was the most popular competition of the year in ML Contests' review: >3,700 teams, $120K prize pool. — [ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)
- Across all 2025 winning solutions: XGBoost appeared 14 times, LightGBM 14, CatBoost 8. More than half of winning teams were **solo** for the third year running, and teams of more than 5 were rare. — [ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)

**Other short student hackathons (format reference)**
- Hi! PARIS Hackathon 2025 (48 h, PISA score prediction, R² metric, 1.17M rows × 307 features): the third-place team's first submission was a tuned CatBoost at R² = 0.76, and their final was 0.78. — [lycoriolis blog](https://lycoriolis.github.io/posts/HiParis2025)
- At a Rotman (U Toronto) on-campus datathon the winners had 6.5 hours, and the instructions and data structure were released a week in advance. Their advice was to use the week before. — [Rotman MMA blog (search snippet; fetched page content had been replaced)](https://rotman.utoronto.ca/Degrees/MastersPrograms/SpecializedProgramsBlog/MMA/MMA-On-Campus-Datathon-2019-The-First-Place-Experience)

**AI-assistant rules**
- Event rules on AI differ. Example: Impact X 2025 allows generative AI but requires its use to be disclosed in the submission. Codeforces prohibits relying on AI for core algorithmic logic. — [Impact X rules](https://kelaenderkoe.pages.dev/Using-https-impactx2025.devpost.com/rules); [Codeforces rule](https://mirror.codeforces.com/topic/134567/en1)
- A Kaggle Playground competition (March 2026) was won by a team using three LLMs (GPT-5.4 Pro, Gemini 3.1 Pro, Claude Opus 4.6), so LLM-assisted coding is accepted in at least that setting. — [NVIDIA blog](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/)

### Inferences
- The two most recent QRT public challenges (#23 and #167) share one template: wide `RET_1..20` / volume lag columns, anonymised and shuffled dates, a cross-sectional group column (sector or `GROUP`), a binary sign target scored by accuracy, and a benchmark barely above 50%. **The team should prepare starter code for exactly this shape**: GroupKFold by date, per-date cross-sectional features, and a sign-accuracy metric. They should also prepare a fallback for regression/Spearman and survival/C-index variants.
- With benchmarks around 50.8–51.3% accuracy, realistic gains in 4 hours are tenths of a percentage point. Leaderboard noise will be about as large as the real differences between models (see Section 2).
- A 4-hour on-site event will most likely be ranked mainly on a leaderboard, possibly with a short presentation or a QRT staff review for recruiting. **Confirm at the briefing**: (a) whether ranking uses the public or a private split, (b) the daily/total submission cap, (c) whether you must select final submissions, (d) AI tool policy, (e) whether external data is allowed, and (f) whether code must be submitted.
- Kaggle defaults (from background knowledge, unverified because the Kaggle docs page did not render): max team size is often 5, about 5 submissions per day, and **2 final submissions** selected, with best-public auto-selected if none are chosen. G-Research's "two best models" rule fits this pattern.

### Gaps
- I found **no public description** of the format of the QRT ETH Zürich hackathon itself, or of any previous on-campus QRT hackathon (duration, team size, AI policy, submission cap).
- I found no public format or rules pages for Jane Street, Two Sigma, Optiver, Squarepoint, HRT or IMC *campus data challenges* (IMC Prosperity is a trading-algorithm competition, not a tabular datathon). Citadel/Correlation One AI-assistant policy: not found.
- Some sources mention a "weighted accuracy" QRT metric (for example weights ∝ |return|), but I could not confirm any QRT challenge that used it. Both #23 and #167 use plain accuracy.

---

## 2. Technical practice for noisy financial tabular data

### Takeaway
Build validation that matches how the test set is constructed. If dates are shuffled, as in QRT's sets, use **GroupKFold by date**. If time order is real, use a time-series split with a purge gap. Engineer **per-date cross-sectional features**, start with **LightGBM and ridge baselines**, average seeds and models, and **choose final submissions by CV rather than public LB**. In G-Research, Jane Street and Optiver, winners got most of their gains from feature engineering and validation discipline, not from model tuning.

### Cited Findings

**Validation that tracks the private LB**
- "Match your CV strategy to how the test data is structured": TimeSeriesSplit for time-dependent data, GroupKFold for grouped data. — [NVIDIA Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data)
- Grandmaster workflow: explore the data, build the standard baseline, set up "a robust local validation", record the baseline score, then iterate "as fast as possible". Use CV schemes that account for leakage and time splits, and **simulate the public/private split with multiple CV folds** to estimate likely shake-up. — [NVIDIA: Grandmasters unveil winning strategies](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)
- Jane Street 2024 (Volkova, top solution): time-series CV with **2 folds of 200 dates each**, matching the public test size, which "correlated well with the public LB". She also tested on the **last 200 dates with a 200-date gap** to simulate the private set. Without online learning and auxiliary targets, the score in the gap fold fell from 0.0161 (fold 0) to **0.0011**, which shows how fast signal decays. — [Volkova solution.md](https://github.com/evgeniavolkova/kagglejanestreet/blob/main/solution.md)
- Purged k-fold (López de Prado): drop training rows whose label horizon overlaps the test fold (purging), and drop a buffer of rows *after* the test fold (embargo). Suggested embargo size differs by source: **1–2% of samples** in one, a **5%** example in Wikipedia. — [Wikipedia: Purged cross-validation](https://en.wikipedia.org/wiki/Purged_cross-validation); [quant-lessons: purging & embargo](https://5x5x5x5.github.io/quant-lessons/backtest/purging-embargo/)
- Jane Street 2021 competitors widely used `PurgedGroupTimeSeriesSplit`, where validation always comes after training "with a small gap in between". — [BulletTech Jane Street summary](https://bullettech.github.io/BulletTech/Main_Course/Machine_Learning/2021-09-02-Kaggle-janestreet)
- Optiver Trading at the Close (mid-table write-up): purged K-fold with **5 folds over 480 date_ids and a purge gap of 2 dates** on each side. Switching to ordinary K-fold made the public score worse (5.3389 vs 5.3341 MAE). — [fan2goa1 blog](https://fan2goa1.github.io/mkdocs-material/blog/2023/12/24/kaggle-optiver---trading-at-the-close)
- Numerai: sklearn's `TimeSeriesSplit` "takes a groups argument, but doesn't actually use it", so a custom era-group splitter is needed. The example used 5 splits (6 era blocks) with an **expanding window**. A Numerai participant argued that if the data contains no overlapping eras, purging is optional "added safety". — [Numerai forum: era-wise TS CV](https://forum.numer.ai/t/era-wise-time-series-cross-validation/791)
- Numerai's "super massive" data release: if you do not subsample eras, purge overlapping eras from CV folds. — [Numerai forum deep dive (search snippet)](https://forum.numer.ai/t/super-massive-data-release-deep-dive)
- Numerai's own scoring computes correlation **per era**, using `numerai_corr`. — [Numerai docs: CORR](https://docs.numer.ai/numerai-tournament/scoring/correlation-corr)

**Leakage traps**
- Jane Street 2021 winner: training the autoencoder *first* and then cross-validating leaks information. The winner optimised the autoencoder and the MLP **jointly inside each fold** to avoid this. — [BulletTech 2022-01-30](https://bullettech.github.io/BulletTech/Main_Course/Machine_Learning/2022-01-30-Kaggle-exp-s2)
- Optiver Realized Volatility: the winner **reverse-engineered the order of the shuffled time_ids from the price tick size**. Because test data for all time_ids arrives in one batch, one can "use aggregated information across all time_ids when making predictions". — [Chris Miles 91st-place write-up](https://chrisrichardmiles.github.io/chrisrichardmiles/projects/optiver/index_optiver.html)
- Hi! PARIS 2025: the third-place team's pipeline began with "remove leaky columns". — [lycoriolis blog](https://lycoriolis.github.io/posts/HiParis2025)
- Compare train and test feature distributions, and plot the target over time, to catch drift. — [NVIDIA Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data)

**Cross-sectional normalisation and market-relative features**
- QRT #23 already demeans volume per day across stocks. Its benchmark adds a sector-conditional mean of RET_1. — [ChallengeData #23](https://challengedata.ens.fr/challenges/23)
- QRT #167 benchmark features include averages across all allocations per date and cross-allocation average volatility. — [ChallengeData #167](https://challengedata.ens.fr/challenges/167)
- Jane Street 2024 winner: **market averages per (date_id, time_id)**, rolling means/stds over the last 1,000 time_ids per symbol, and time_id as a feature together gave **+0.002 CV**. Preprocessing was simple standardisation with NaN → 0; "other methods didn't provide any improvement". — [Volkova solution.md](https://github.com/evgeniavolkova/kagglejanestreet/blob/main/solution.md)
- Optiver Trading at the Close: the target was already market-relative, (stock WAP return − index WAP return) × 10,000 over 60 s, with the index approximated by a regression-weighted combination of stocks. — [fan2goa1 blog](https://fan2goa1.github.io/mkdocs-material/blog/2023/12/24/kaggle-optiver---trading-at-the-close)
- Groupby aggregations are called the most powerful feature-engineering technique in an NVIDIA Grandmaster's winning workflow. — [NVIDIA cuDF feature engineering post (search snippet)](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas)

**Robust baselines and concrete parameters**
- Build quick benchmarks, a linear model and a basic tree, to gauge how much signal exists. Start tabular work with XGBoost/LightGBM. — [NVIDIA: Grandmasters unveil winning strategies](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)
- Build diverse baselines across linear models, GBDTs and small NNs. In one Kaggle Playground competition (Rainfall), an ensemble of GBDT + NN + SVR with **no feature engineering** came 2nd. — [NVIDIA Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data)
- Test several GBDT libraries, because differences can be large. One 2025 winner found XGBoost best, LightGBM useful for diversity, and CatBoost weaker. AutoGluon (2-hour limit, "best" preset) beat a hand-tuned XGBoost/LightGBM/TabM ensemble in one 2025 win. — [ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)
- Concrete LightGBM settings seen in a finance write-up: `LGBMRegressor`, n_estimators 5,000–7,000, `subsample=0.7`, `colsample_bytree=0.7`. — [fan2goa1 blog](https://fan2goa1.github.io/mkdocs-material/blog/2023/12/24/kaggle-optiver---trading-at-the-close)
- Numerai example CV grid: XGBoost with `n_estimators=2000`, a grid of 5 learning rates × 5 colsample_bytree × 3 max depths (75 models). — [Numerai forum](https://forum.numer.ai/t/era-wise-time-series-cross-validation/791)
- QRT #23 benchmark: RF with 500 trees, max_depth 8. QRT #167 benchmark: LightGBM + Ridge. — [#23](https://challengedata.ens.fr/challenges/23); [#167](https://challengedata.ens.fr/challenges/167)
- Hi! PARIS: CatBoost stayed best, LightGBM was slow, XGBoost added nothing, and Ridge/Lasso did poorly. Lesson: "Simple, well-tuned models can beat poorly implemented complex ones". — [lycoriolis blog](https://lycoriolis.github.io/posts/HiParis2025)

**Ensembling and seed averaging**
- Hill climbing: start from the strongest single model, add others with weights, keep an addition only if it improves validation, and repeat. Stacking: use out-of-fold (OOF) predictions as level-2 inputs. Train **multiple seeds** and average. In one example, 100 XGBoost seeds gave about 0.379 vs about 0.376 MAP@3 for an average single seed. Then **retrain on 100% of the data** after choosing hyperparameters. — [NVIDIA Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data)
- Jane Street 2024: 2 architectures × 3 seeds, simple unweighted average. LB rose from 0.0105 to **0.0112**, and CV from 0.0214 (best single) to 0.0222. For the final fit, retrain on all data with **epochs = average optimal epochs from CV**. — [Volkova solution.md](https://github.com/evgeniavolkova/kagglejanestreet/blob/main/solution.md)
- Jane Street 2021 (scaomath): blend by taking the median of the middle 60% of predictions. Winner: Gaussian-noise augmentation plus sample weights. — [vankyle00 research note (secondary)](https://vankyle00.github.io/trading-models/docs/concepts/02-jane-street-key-concepts.html)
- Always save OOF and test predictions to disk for every experiment, good or bad, so they can be ensembled later. — [NVIDIA: LLM-assisted Kaggle win](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/)

**Public-LB overfitting and final selection**
- People who tune heavily to the public LB tend to fall on the private LB. Treat the public set as a final holdout and weigh your own validation. — [Kaggle Book summary](https://alibakh62.github.io/pages/content/book_summary/the_kaggle_book/designing_good_validation.html)
- Rule of thumb: local CV beats the public LB "about 90% of the time". If unsure, blend the two, for example 50/50. (Taken from a search-result summary of a Kaggle-tips slide deck that I did not open.) — [SlideShare: Tips and tricks to win Kaggle competitions](https://pt.slideshare.net/slideshow/tips-and-tricks-to-win-kaggle-data-science-competitions/72909469)
- Santander case: teams that trusted CV saw much smaller rank swings. — [MachineLearningMastery: model selection tips](https://www.machinelearningmastery.com/model-selection/)
- QRT electricity challenge example of shake-up: 0.295 public → 0.251 private Spearman. — [chantomkit/QRT2023](https://github.com/chantomkit/QRT2023)

### Inferences
- **The core validation rule for QRT-style data:** if dates are shuffled and anonymised (#23, #167), there is no temporal order to respect, so use **GroupKFold(n_splits=5) on the date column**. Never split rows from the same date across train and validation. Same-day rows share market moves, so a random KFold leaks the cross-section and inflates CV. If a time order *is* given (an integer date that is clearly sequential), use an expanding-window split with a gap of at least the label horizon (≥1 date for 1-day targets). Purging only matters when label windows overlap.
- **Possible hidden-order leak (inference; check rules before exploiting):** in lag-window datasets like #23 and #167, `RET_1` of a row on date d+1 for asset s *is* the target of the same asset on date d. Matching shifted return vectors (RET_k of one date equals RET_{k+1} of another) could rebuild the shuffled date order, as the Optiver RV winner did with tick size. Check this in 5 minutes with a merge on rounded values. If it works, (a) the CV may be leaky in ways the organisers did not intend, and (b) ask staff whether exploiting it is allowed. Do not build the final submission on an exploit the event might disqualify.
- **Cross-sectional features to build first** (from background knowledge plus the cited patterns): per-date rank or z-score of each RET_k/volume column; return minus date mean and minus date×sector/GROUP mean; per-date market return (mean RET_1..k) and dispersion; per-asset rolling vol (std of RET_1..20) and return/vol ratios; short-horizon reversal (−RET_1) and momentum (sum RET_1..5, 1..20). Compute per-date statistics **within each split's own rows** (train dates from train, test dates from test). Only use test rows to compute same-date statistics for test predictions, which is legitimate because all test rows are given at once and the Optiver RV notes describe the same thing. Never pool target values across splits.
- **Target transformation:** for sign-accuracy tasks, regressing the real-valued target (if provided, as `TARGET` is in #167) and taking the sign often beats a pure classifier, because the model gets magnitude information. Try both. Winsorise or clip extreme targets (for example at the 1st/99th percentile) or rank-transform per date before training to tame fat tails. *(background knowledge; unverified)*
- **Starter LightGBM params for noisy finance tabular data (background knowledge; tune only if time allows):** `objective` matched to the metric; `learning_rate=0.03–0.05`; `num_leaves=15–63` with `max_depth=4–8`; `min_child_samples=200–1000` (large leaves resist noise); `feature_fraction=0.5–0.8`; `bagging_fraction=0.7–0.8, bagging_freq=1`; `lambda_l2=1–10`; up to 5,000 rounds with `early_stopping=100` on the grouped CV; then retrain on all data with about mean best_iteration × 1.1. Ridge: standardise, `alpha` in {1, 10, 100, 1000}.
- **Leaderboard noise estimate (my arithmetic):** for accuracy near 0.5 on n rows, SE ≈ √(0.25/n). For #167's 31,870 test rows, SE ≈ 0.28 pp on the full test set, and about 0.4 pp if the public LB is half of it. Same-date rows are correlated, so the effective n is lower and the true noise is larger. So **public-LB differences under about 0.5–1 pp of accuracy are not evidence**. For Spearman on 654 rows (#97), SE ≈ 1/√651 ≈ 0.04, which matches the 0.295 → 0.251 drop.
- **Final selection rule for the event:** submit (1) the best-CV ensemble (rank- or probability-averaged GBDT seeds + ridge) and (2) a second, *different* robust candidate, for example the simplest model with near-best CV. Never select a submission because of a public-LB jump that CV does not reproduce.

### Gaps
- I could not fetch the 1st-place write-ups for Optiver Trading at the Close, Optiver RV (nyanp) or Ubiquant (Kaggle discussions are JS-rendered). Their exact features and params are unverified.
- I found no source that quantifies the optimal number of folds for 4-hour events. 5 folds is a convention.

---

## 3. Metric-specific tricks

### Takeaway
Optimise for exactly what is scored. Rank-based metrics (Spearman, C-index, Numerai CORR) care only about ordering. Correlation metrics ignore scale. Zero-mean R² punishes confident wrong predictions, so shrink toward zero. Accuracy on a sign target is decided by the threshold and the class balance, and zero returns count as class 0 in QRT's definition.

### Cited Findings
- **Numerai CORR:** rank predictions (ties averaged), scale as (rank − 0.5)/n, gaussianise with the inverse normal CDF, centre the target, apply a signed power of 1.5 to both "to accentuate the tails", then take Pearson correlation. "You are only evaluated on your prediction's ranks." — [Numerai docs](https://docs.numer.ai/numerai-tournament/scoring/correlation-corr)
- Era-wise Spearman (0.0255) and Numerai's metric (0.0254) are "virtually identical", but pooled correlation (0.0250) differs. Compute your CV metric **exactly as the grader does** (per era vs pooled). — [Numerai forum](https://forum.numer.ai/t/era-wise-time-series-cross-validation/791)
- **Weighted Pearson (G-Research):** student write-ups trained directly on a negative weighted-Pearson minibatch loss, noting that predictions at different scales give the same score, which matters when ensembling. — [G-Research wrap-up](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/); [HKUST project PDF (secondary)](https://aifin-hkust.github.io/project3/project3_Crypto_M5.pdf)
- **Jane Street 2024 (weighted zero-mean R²):** the winner trained directly on the weighted zero-mean R² loss, summed over auxiliary responders. Auxiliary targets added about +0.001 CV/LB, and online learning added +0.008. — [Volkova solution.md](https://github.com/evgeniavolkova/kagglejanestreet/blob/main/solution.md)
- **Jane Street 2021 (utility score):** profit × a capped Sharpe-like term, so "it's also important that the profit is evenly distributed across time". Solutions used utility-function regularisers ("train on the metric you're paid on"). — [BulletTech JS summary](https://bullettech.github.io/BulletTech/Main_Course/Machine_Learning/2021-09-02-Kaggle-janestreet); [vankyle00 note](https://vankyle00.github.io/trading-models/docs/concepts/02-jane-street-key-concepts.html)
- **RMSPE (Optiver RV):** each error is divided by the true value. — [search summary of Optiver RV write-ups](https://bullettech.github.io/BulletTech/Main_Course/Machine_Learning/2021-08-27-Kaggle-optiver)
- **C-index:** QRT's survival challenge uses IPCW C-index. Harrell's, Uno's and other C-index estimators can disagree because of choices such as tie handling. — [Thierry Moudiki blog](https://thierrymoudiki.github.io/blog/2025/02/10/python/Benchmark-QRT-Cube); [The C-index Multiverse (arXiv 2508.14821)](https://arxiv.org/html/2508.14821v1)
- **Sign-accuracy (QRT #167):** sign(x)=1 if x>0 else 0, so exact zeros are class 0. — [ChallengeData #167](https://challengedata.ens.fr/challenges/167)
- "Understand the competition metric and optimize the solution directly for it." — [NVIDIA](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)

### Inferences (background knowledge unless cited; verify formulas against the event's metric file)
- **Accuracy / sign:** (1) Check the base rate. If P(y=1) ≠ 0.5, for example because zeros count as 0, a constant "all 0" or "all 1" may already beat 0.5, so report CV *against the majority-class baseline*. (2) Pick the probability threshold on OOF predictions. With calibrated probabilities 0.5 is optimal for plain accuracy, but per-date class imbalance can move it. A per-date median threshold (predict 1 for the top half of each date) is a robust option when the cross-section is balanced. (3) If the metric is **weighted** accuracy, for example weights ∝ |return|, train with `sample_weight=|y|` or regress y and take the sign. Big-move days dominate the score.
- **Spearman / rank correlation:** only ranks matter. Rank-transform or gaussianise the target before regression. Ensemble by **averaging ranks**, not raw predictions. If the grader scores per date, rank per date. If it is pooled (like #97), make sure predictions are comparable across dates. A Pearson-trained model on a rank-transformed target is a strong default.
- **Weighted zero-mean R² (Jane Street):** R² = 1 − Σwᵢ(yᵢ−ŷᵢ)² / Σwᵢyᵢ² *(formula from background knowledge; the official page was not fetched)*. Predicting 0 scores exactly 0, and noisy predictions score below 0. So **multiply predictions by a shrink factor in (0, 1) tuned on OOF**, train with MSE plus sample weights, and clip extremes.
- **RMSPE:** train with `sample_weight = 1/y²`, or fit log(y) and exponentiate. Never let predictions approach 0.
- **C-index:** only the order of risk scores matters, and **sign convention is critical**: a higher score must mean higher risk or earlier event, depending on the scorer. Flipping the sign turns C into 1 − C. Fit Cox PH, `xgboost`/`lightgbm` with a Cox objective, or random survival forests, then rank-average them. Validate with the same IPCW estimator (for example `sksurv.metrics.concordance_index_ipcw`), and stratify folds by event status.
- **MAE (Optiver TatC, background knowledge):** use an L1/median objective. Top solutions reportedly post-processed predictions so the index-weighted sum is zero, because the target is relative to the index *(unverified)*.
- **Pearson / weighted correlation:** demean predictions per time slice if the target is market-relative, and do not worry about scale. Blends can use z-scored components.

### Gaps
- I could not open the official formula pages for the Jane Street 2024 metric, Optiver TatC (MAE) or the post-processing of the 1st-place solutions.
- I found no QRT-specific source using "weighted accuracy".

---

## 4. Time management for a 4-hour event, team split, and using LLM coding agents

### Takeaway
Get a **valid submission within the first 30–45 minutes** (benchmark or constant). Have a CV-validated LightGBM/ridge baseline submitted by about 1:00–1:15. Run parallel feature and model work from 1:15 to 2:45. Ensemble from 2:45 to 3:20. **Freeze code at about 3:30** and spend the rest on verification and final selection. One person owns the shared CV harness, metric and submission writer. LLM agents produce experiments that write OOF/test predictions to disk, and they never touch the submission path.

### Cited Findings
- Grandmaster process: baseline, then local validation, then record score, then iterate "as fast as possible". Simple MVP first, then feedback loops. Use lightweight scripts for debugging, save intermediate features and predictions, and treat the notebook like a production pipeline. — [NVIDIA](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)
- Hackathon planning template (24 h): 1 hour of scoping, a "red line" for the **first working end-to-end slice**, and agree on the data interface between stages in the first hour so teammates can work against a fixed sample file. Pair on the first running version before splitting up, and keep a buffer at the end for hardening. — [Builder's Guide ch. 10 (weak source)](https://intercom.help/builderbase/en/articles/14836121-builder-s-guide-ch-10-the-first-hour)
- Front-load preparation: the Rotman winners (6.5 h event) used the week before the event. — [Rotman MMA blog (search snippet)](https://rotman.utoronto.ca/Degrees/MastersPrograms/SpecializedProgramsBlog/MMA/MMA-On-Campus-Datathon-2019-The-First-Place-Experience)
- "Know when to stop": the Hi! PARIS team plateaued at R² 0.78 despite a two-stage design, kernel PCA and tuning. Their first submission, plain CatBoost, was already at 0.76. — [lycoriolis blog](https://lycoriolis.github.io/posts/HiParis2025)
- **LLM-agent workflow that won a Kaggle competition (Mar 2026):** three LLM agents wrote EDA, baselines (k-fold, saving **OOF and test predictions as .npy**), feature-engineering variants and ensembling code (hill climbing and ridge/logistic/NN/GBDT stackers). They produced >600,000 lines of code and **850 experiments**, and the final entry was a 4-level stack of 150 models. The human ran the code, fed outputs back, chose ideas and directed the work. Advice: start with EDA and simple baselines, **ask agents to summarise all experiments**, and keep OOF/test predictions for every run. — [NVIDIA: winning with generative-AI-assisted coding](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/)
- Autonomous agents alone are weak. On MLE-bench, the best agent (o1-preview + AIDE) won at least a bronze medal in **16.9%** of 75 Kaggle competitions, Claude 3.5 Sonnet + AIDE in 7.6%. Later systems reached 47.7–55% on the 22-competition Lite split. — [MLE-bench (arXiv 2410.07095)](https://arxiv.org/pdf/2410.07095); [DeepLearning.AI summary](https://www.deeplearning.ai/the-batch/openais-mle-bench-tests-ai-coding-agents); [AIRA paper](https://arxiv.org/html/2507.02554v2)
- ML Contests 2025 found no publications showing autonomous agents succeeding in *live* competitions. — [ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)
- Experiment tracking: the Jane Street 2024 winner used WandB "to keep track of scores and easily revert to an older version of the code", and ran a script that monitored Kaggle submissions. — [Volkova solution.md](https://github.com/evgeniavolkova/kagglejanestreet/blob/main/solution.md)
- Hardware: Colab crashed for lack of RAM at Hi! PARIS, and the team moved to Kaggle Notebooks with 32 GB. — [lycoriolis blog](https://lycoriolis.github.io/posts/HiParis2025)

### Inferences (synthesised playbook)

**Tonight (prep, before the clock starts)**
- Prepare a repo template: `data/` (read-only), `src/cv.py` (fold assignment saved to `folds.csv`, GroupKFold by date), `src/metric.py` (accuracy / Spearman / R² / C-index, each a function with the grader's conventions), `src/make_submission.py` (asserts: same row count as test, same ID set, no NaN, correct column names and dtype, sorted like the sample submission), `runs/<run_id>/{oof.npy,test.npy,config.json,cv.json}`, and `LOG.md` or a shared sheet: run_id, owner, description, CV mean ± std, public LB, file name.
- Pre-write and test the starter pipelines on a QRT public dataset (#23 or #167 if accessible): LightGBM + ridge with per-date features. Install the packages and check RAM.
- Write the agent instructions file (CLAUDE.md / AGENTS.md) for the event repo: never edit `cv.py`, `metric.py` or `make_submission.py`; never fit anything on test labels; every experiment writes to its own `runs/` folder; report CV in a fixed format.

**T+0:00–0:15: read, don't code**
- Everyone reads the task statement, **metric definition**, submission format, submission cap and rules on AI and external data. Ask organisers any questions in the first 10 minutes.
- The captain assigns roles. Someone loads the data and prints shapes, dtypes, ID uniqueness, number of dates/assets, whether train and test dates overlap, and the target distribution and base rate.

**T+0:15–0:45: first valid submission**
- Submit the provided benchmark (or a constant/majority-class or sign(−RET_1) heuristic) **through your own `make_submission.py`**. This checks the format and ID alignment end to end and sets the LB reference point.
- Freeze `folds.csv`. Check that the CV of a trivial model matches its LB (±noise).

**T+0:45–1:15: real baseline**
- LightGBM with the default-ish params above, plus ridge on standardised features, using raw columns and 5-fold GroupKFold by date. Submit the better one. This is the safety net: if all else fails, this is the final.

**T+1:15–2:45: parallel exploration (where agents earn their keep)**
- Each person or agent owns one lane and writes OOF/test preds:
  (a) per-date cross-sectional features (ranks, demeaning, sector/GROUP-relative);
  (b) per-asset history features (vol, momentum/reversal, volume-return interactions) and target transforms (regress vs classify, clipping, per-date ranking);
  (c) model diversity (CatBoost/XGBoost, ridge on ranks, small MLP if time);
  (d) leak/structure check (hidden order, duplicate rows, train/test drift).
- Rule: a feature family is kept only if it improves **mean CV by more than about 1 SE across folds**. Submit to the LB only for milestone models, not every tweak.

**T+2:45–3:20: consolidate**
- Hill-climb or rank-average the saved OOFs. Do 3–5 seeds of the best GBDT. Retrain on full train. Generate the final candidates.

**T+3:20–3:40: select and verify**
- Choose 2 finals: best-CV ensemble plus a robust simpler model. Re-run `make_submission.py` assertions. Diff against the previous good file (correlation of predictions should be high). Upload with ≥15 min to spare.

**T+3:40–4:00: buffer and presentation.** No new modelling.

**Team split for 3–5 people**
- **Captain / integrator:** owns the repo, submission writer, log and final selection, and is the only person who submits.
- **Validation and metric owner:** builds folds and the metric, reviews every claimed CV gain for leakage, and runs the leak/structure checks.
- **1–2 feature engineers:** lanes (a) and (b).
- **Modeller / ensembler:** lane (c), then the blend.
- (If judged) **Presenter**, from about T+3:00: writes 3–5 slides covering validation design, key features, CV vs LB table and what didn't work.

**LLM-agent rules of engagement**
- One agent per lane, each in a **separate git worktree or folder**, all importing the same frozen `cv.py`/`metric.py`.
- Give agents small, testable tasks ("add per-date rank features for RET_1..5, report 5-fold CV in this table format"). Do not hand them open-ended "improve the score" goals.
- Treat any CV jump above about 2× the typical gain as a **leak until proven otherwise**. Common agent mistakes are fitting scalers or encoders on train+test with targets, random KFold, using the target column as a feature, and computing per-date stats over train and test together *with targets*.
- Humans review diffs for anything that touches data loading or the submission. Agents never write the final CSV.
- Have one agent keep `LOG.md` up to date ("summarise all experiments"), following the NVIDIA workflow.

### Gaps
- I found no source with timed splits specifically for 4-hour data hackathons. The schedule above is synthesised from multi-day competition practice and general hackathon guidance.
- I found no published guidance from QRT on whether LLM coding agents are allowed at its hackathons.

---

## 5. Common failure modes

### Takeaway
In short events, most lost placings come from process errors rather than modelling errors: a malformed or misaligned submission, no early submission, leaky CV that misleads the team, chasing public-LB noise, compute or RAM crashes, and losing track of which file produced which score.

### Cited Findings
- **Wrong fold semantics:** sklearn `TimeSeriesSplit` silently ignores `groups`. String era labels sort lexicographically ("era1", "era10", "era100"), which gives wrong fold order. Fix by converting to integers. — [Numerai forum](https://forum.numer.ai/t/era-wise-time-series-cross-validation/791)
- **Leakage from pre-fitting a component outside CV** (for example, an autoencoder fitted on all data before CV). — [BulletTech 2022-01-30](https://bullettech.github.io/BulletTech/Main_Course/Machine_Learning/2022-01-30-Kaggle-exp-s2)
- **Non-time-aware CV is worse:** replacing purged K-fold with ordinary K-fold worsened the public score in Optiver TatC. — [fan2goa1 blog](https://fan2goa1.github.io/mkdocs-material/blog/2023/12/24/kaggle-optiver---trading-at-the-close)
- **Signal decay and non-stationarity:** fold scores in Jane Street 2024 fell from 0.0161 to 0.0011 when the gap grew to 200 dates (without online learning). — [Volkova solution.md](https://github.com/evgeniavolkova/kagglejanestreet/blob/main/solution.md)
- **Public-LB overfitting / shake-up:** see Section 2. Noisy financial leaderboards showed "many big jumps and precipitous drops". — [G-Research](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/)
- **Compute failure mid-event:** a Colab out-of-memory crash forced a platform switch. — [lycoriolis blog](https://lycoriolis.github.io/posts/HiParis2025)
- **Over-investing in complexity:** a two-stage model, kernel PCA and tuning added only +0.02 R² over the first CatBoost submission. — [lycoriolis blog](https://lycoriolis.github.io/posts/HiParis2025)
- **Submission format:** QRT requires exactly two columns (`ID`, `RET`) with IDs matching the test rows. — [ChallengeData #23](https://challengedata.ens.fr/challenges/23)
- **Losing track of results:** winners use experiment trackers (WandB) and save OOF/test predictions for every run. — [Volkova](https://github.com/evgeniavolkova/kagglejanestreet/blob/main/solution.md); [NVIDIA](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/)

### Inferences (pre-submission checklist; background knowledge)
1. Row count equals the test row count. The ID set equals the test ID set (`set(sub.ID) == set(test.ID)`). There are no duplicate IDs.
2. Join predictions to test **by ID, never by position**, especially after any groupby/sort/merge that reorders rows. Pandas `groupby().transform` keeps order, but `merge` and `sort_values` do not.
3. The prediction column name and dtype match the sample: for example int 0/1 vs bool vs probability. Sending probabilities where labels are expected (or the reverse) silently costs points. Check the sign convention for C-index and the label convention for accuracy (does 1 mean > 0?).
4. No NaN or inf. Values are in the expected range. Prediction distribution is sane (for sign tasks, not 100% one class unless intended).
5. The file name includes the run_id, and the run_id is logged with CV and LB in `LOG.md` *before* uploading.
6. Submit early and often enough to have a safety net, but keep ≥2 submissions in reserve for the final hour if there is a cap.
7. Do not spend more than about 20 minutes on hyperparameter tuning. In noisy financial data, features, validation and averaging matter more (consistent with the G-Research organisers' observation about feature engineering).
8. Plan the last 20 minutes: no new code, only verification and upload.

### Gaps
- I found no systematic study that quantifies how often short-hackathon teams lose to format or ID errors. The checklist above comes from general practice.
