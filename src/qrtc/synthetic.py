"""Synthetic datasets shaped like past QRT challenges (Challenge-Data file layout).

Each generator writes X_train.csv, y_train.csv, X_test.csv, sample_submission.csv and
y_test_hidden.csv (to score submissions locally) into ``<out_dir>``. Signals are weak
on purpose (financial data is noisy): a good pipeline beats the naive baseline by a little.

  stock_sign   panel (DATE x STOCK), RET_1..20 / VOLUME_1..20, target sign of next return,
               test dates interleaved with train dates, metric accuracy
  allocation   like QRT #167 "asset allocation performance" (2026): TS x ALLOCATION panel,
               real-valued TARGET scored on its sign (binarize_target), metric accuracy
  electricity  daily FR/DE rows, weather/commodity features, target price change,
               test days after train days, metric spearman
  football     match rows with HOME_*/AWAY_* team stats, one-hot target
               HOME_WINS/DRAW/AWAY_WINS, metric accuracy
  survival     patients with clinical + mutation features, OS_YEARS/OS_STATUS,
               test patients from unseen centres, metric ipcw_cindex (tau = 7)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

Frames = tuple[pd.DataFrame, pd.DataFrame, pd.Series]  # X, y, is_test mask


def _panel(rng: np.random.Generator, n_dates: int, n_assets: int, key: str, entity: str,
           n_lags: int = 20) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulated returns with sector factor, short-term reversal and momentum signals."""
    sectors = rng.integers(0, 8, n_assets)
    T = n_dates + n_lags + 1
    factor = rng.normal(0, 0.01, (T, 8))
    rets = factor[:, sectors] + rng.normal(0, 0.02, (T, n_assets))
    for t in range(n_lags + 1, T):  # weak reversal (lag 1) + sector momentum (5-day)
        rets[t] += -0.12 * rets[t - 1] + 0.15 * factor[t - 5:t, sectors].sum(axis=0)
    rows = []
    for d in range(n_dates):
        t = d + n_lags
        past = rets[t - n_lags:t][::-1]  # RET_1 = most recent
        block = pd.DataFrame(past.T, columns=[f"RET_{k}" for k in range(1, n_lags + 1)])
        block.insert(0, entity, np.arange(n_assets))
        block.insert(0, key, d)
        block["SECTOR"] = sectors
        block["_future"] = rets[t]
        rows.append(block)
    return pd.concat(rows, ignore_index=True), sectors


def stock_sign(rng: np.random.Generator) -> Frames:
    panel, _ = _panel(rng, n_dates=160, n_assets=250, key="DATE", entity="STOCK")
    for k in range(1, 21):
        panel[f"VOLUME_{k}"] = rng.lognormal(0, 0.5, len(panel)) * (1 + 20 * panel[f"RET_{k}"].abs())
    panel["INDUSTRY"] = panel["SECTOR"] * 3 + panel["STOCK"] % 3
    panel = panel.mask(rng.random(panel.shape) < 0.02).fillna({"DATE": 0, "STOCK": 0})
    future = panel.pop("_future").fillna(0)
    test_dates = rng.choice(160, 40, replace=False)
    X = panel.assign(ID=np.arange(len(panel)))[["ID", *panel.columns]]
    return X, pd.DataFrame({"ID": X["ID"], "RET": future > 0}), X["DATE"].isin(test_dates)


def allocation(rng: np.random.Generator) -> Frames:
    """Mimics QRT #167: ROW_ID, TS (DATE_xxxx, shuffled), ALLOCATION, RET_1..20,
    SIGNED_VOLUME_1..20, MEDIAN_DAILY_TURNOVER, GROUP; TARGET = real next-day return."""
    panel, sectors = _panel(rng, n_dates=300, n_assets=60, key="TS", entity="ALLOCATION")
    for k in range(1, 21):
        panel[f"SIGNED_VOLUME_{k}"] = rng.normal(0, 1, len(panel)) + 30 * panel[f"RET_{k}"]
    panel["MEDIAN_DAILY_TURNOVER"] = rng.gamma(2, 0.05, len(panel))
    panel["GROUP"] = "GROUP_" + pd.Series(sectors[panel["ALLOCATION"]] % 4).astype(str)
    panel["ALLOCATION"] = panel["ALLOCATION"].map(lambda a: f"ALLOCATION_{a + 1:02d}")
    order = rng.permutation(300)  # shuffled, anonymised dates
    test_ts = set(order[:75])
    is_test = panel["TS"].isin(test_ts)
    panel["TS"] = panel["TS"].map(lambda t: f"DATE_{order[t] + 1:04d}")
    future = panel.pop("_future")
    panel = panel.drop(columns=["SECTOR"])
    X = panel.assign(ROW_ID=np.arange(len(panel)))[["ROW_ID", *panel.columns]]
    return X, pd.DataFrame({"ROW_ID": X["ROW_ID"], "TARGET": future}), is_test


def electricity(rng: np.random.Generator) -> Frames:
    n_days = 900
    days = np.repeat(np.arange(n_days), 2)
    country = np.tile(["FR", "DE"], n_days)
    feats = {name: rng.normal(0, 1, 2 * n_days) for name in
             ["CONSUMPTION", "WIND", "SOLAR", "TEMP", "GAS_RET", "COAL_RET", "CARBON_RET",
              "NUCLEAR", "HYDRO", "EXCHANGE"]}
    target = (0.25 * feats["CONSUMPTION"] - 0.2 * feats["WIND"] + 0.15 * feats["GAS_RET"]
              + 0.1 * (country == "DE") * feats["COAL_RET"] + rng.standard_t(3, 2 * n_days))
    X = pd.DataFrame({"ID": np.arange(2 * n_days), "DAY_ID": days, "COUNTRY": country, **feats})
    X = X.mask(rng.random(X.shape) < 0.03).fillna({"ID": -1, "DAY_ID": 0, "COUNTRY": "FR"})
    X["ID"] = np.arange(len(X))
    return X, pd.DataFrame({"ID": X["ID"], "TARGET": target}), X["DAY_ID"] >= 700


def football(rng: np.random.Generator) -> Frames:
    n = 6000
    strength_h, strength_a = rng.normal(0, 1, n), rng.normal(0, 1, n)
    cols = {}
    for side, s in (("HOME", strength_h), ("AWAY", strength_a)):
        for stat in ["GOALS", "SHOTS", "SHOTS_ON_TARGET", "POSSESSION", "PASSES", "FOULS"]:
            cols[f"{side}_TEAM_{stat}_season_average"] = s * rng.uniform(0.2, 0.6) + rng.normal(0, 1, n)
            cols[f"{side}_TEAM_{stat}_5_last_match_average"] = s * 0.3 + rng.normal(0, 1, n)
    diff = strength_h - strength_a + 0.3 + rng.logistic(0, 1, n)
    outcome = np.select([diff > 0.6, diff < -0.6], [0, 2], 1)
    X = pd.DataFrame({"ID": np.arange(n), **cols})
    y = pd.DataFrame({"ID": X["ID"], "HOME_WINS": (outcome == 0).astype(int),
                      "DRAW": (outcome == 1).astype(int), "AWAY_WINS": (outcome == 2).astype(int)})
    return X, y, pd.Series(rng.random(n) < 0.25)


def survival(rng: np.random.Generator) -> Frames:
    n = 3000
    center = rng.choice([f"C{i}" for i in range(12)], n)
    clin = {"BM_BLAST": rng.gamma(2, 5, n), "WBC": rng.lognormal(1.5, 0.8, n),
            "ANC": rng.lognormal(0.5, 0.7, n), "MONOCYTES": rng.lognormal(-1, 0.8, n),
            "HB": rng.normal(10, 1.8, n), "PLT": rng.lognormal(4.5, 0.7, n),
            "N_MUTATIONS": rng.poisson(3, n), "TP53_MUT": (rng.random(n) < 0.1).astype(int),
            "CYTO_RISK": rng.choice(["favorable", "intermediate", "adverse"], n, p=[.3, .5, .2])}
    risk = (0.03 * clin["BM_BLAST"] - 0.15 * clin["HB"] - 0.3 * np.log(clin["PLT"])
            + 0.9 * clin["TP53_MUT"] + 0.1 * clin["N_MUTATIONS"]
            + np.select([clin["CYTO_RISK"] == "adverse", clin["CYTO_RISK"] == "favorable"],
                        [0.8, -0.6], 0))
    event_time = rng.exponential(np.exp(-risk) * 25)
    censor_time = rng.uniform(0.5, 10, n)
    X = pd.DataFrame({"ID": [f"P{i}" for i in range(n)], "CENTER": center, **clin})
    y = pd.DataFrame({"ID": X["ID"], "OS_YEARS": np.minimum(event_time, censor_time).round(3),
                      "OS_STATUS": (event_time <= censor_time).astype(int)})
    return X, y, X["CENTER"].isin(["C10", "C11"])


GENERATORS: dict[str, Callable[[np.random.Generator], Frames]] = {
    "stock_sign": stock_sign, "allocation": allocation, "electricity": electricity,
    "football": football, "survival": survival,
}


SAMPLE_OVERRIDES: dict[str, dict[str, object]] = {"allocation": {"TARGET": 1}}


def _sample_values(y_test: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Sample submission: ids + a constant placeholder of the right dtype."""
    sample = y_test.copy()
    for col in sample.columns[1:]:
        sample[col] = sample[col].iloc[0] if len(sample) else 0
    if kind == "survival":  # submission is ID + risk_score (QRT #162)
        return pd.DataFrame({"ID": y_test["ID"], "risk_score": 0.0})
    return sample.assign(**SAMPLE_OVERRIDES.get(kind, {}))


def generate(kind: str, out_dir: Path, seed: int = 0) -> Path:
    """Write the synthetic dataset ``kind`` to ``out_dir``; returns the directory."""
    if kind not in GENERATORS:
        raise ValueError(f"unknown kind {kind!r}; choose from {sorted(GENERATORS)}")
    X, y, is_test = GENERATORS[kind](np.random.default_rng(seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(is_test, dtype=bool)
    X[~mask].to_csv(out_dir / "X_train.csv", index=False)
    y[~mask].to_csv(out_dir / "y_train.csv", index=False)
    X[mask].to_csv(out_dir / "X_test.csv", index=False)
    y[mask].to_csv(out_dir / "y_test_hidden.csv", index=False)
    _sample_values(y[mask], kind).to_csv(out_dir / "sample_submission.csv", index=False)
    return out_dir
