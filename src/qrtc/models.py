"""Model zoo with one interface: ``make_model(name, task, params, seed)`` -> object with
``fit(X, y, X_val=None, y_val=None)`` and ``predict(X)`` returning task-shaped predictions
(see qrtc.metrics for the conventions).

``tabicl``, ``tabpfn`` and ``tabdpt`` are pretrained foundation models (qrtc.foundation; install
with ``make models``). Defaults are deliberately conservative for low signal-to-noise financial data (small learning
rate, large min_child_samples, row/column subsampling). Early stopping on the validation fold
makes the CV score optimistic: it is off unless ``early_stopping_rounds`` is passed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from qrtc import foundation

DEFAULTS: dict[str, dict[str, Any]] = {
    "lgbm": dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                 verbose=-1, n_jobs=-1),
    "xgb": dict(n_estimators=500, learning_rate=0.03, max_depth=6, min_child_weight=10,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                enable_categorical=True, n_jobs=-1),
    "catboost": dict(iterations=800, learning_rate=0.05, depth=6, verbose=0,
                     allow_writing_files=False),
    "ridge": dict(alpha=10.0),
    "logreg": dict(C=0.1, max_iter=2000),
    "extratrees": dict(n_estimators=400, min_samples_leaf=50, max_features=0.5, n_jobs=-1),
    "rf": dict(n_estimators=400, min_samples_leaf=50, max_features=0.5, n_jobs=-1),
    "hgb": dict(max_iter=400, learning_rate=0.05, min_samples_leaf=100,
                categorical_features="from_dtype"),
    "coxnet": dict(l1_ratio=0.5, alpha_min_ratio=0.01, fit_baseline_model=False),
    "coxph": dict(alpha=1.0),
    "rsf": dict(n_estimators=300, min_samples_leaf=20, max_features="sqrt", n_jobs=-1),
    "gbsa": dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8),
}

# Named presets: (base model, param overrides). Shallow GBMs did best on noisy QRT data.
PRESETS: dict[str, tuple[str, dict[str, Any]]] = {
    "lgbm_shallow": ("lgbm", dict(num_leaves=7, max_depth=3, learning_rate=0.02,
                                  n_estimators=800, min_child_samples=200)),
    "lgbm_deep": ("lgbm", dict(num_leaves=127, learning_rate=0.02, n_estimators=1500,
                               min_child_samples=50)),
    "xgb_shallow": ("xgb", dict(max_depth=3, learning_rate=0.02, n_estimators=800)),
    "catboost_shallow": ("catboost", dict(depth=4, iterations=1200, learning_rate=0.03)),
}

SUPPORTED: dict[str, frozenset[str]] = {
    "binary": frozenset({"lgbm", "xgb", "catboost", "logreg", "extratrees", "rf", "hgb"})
    | foundation.NAMES,
    "multiclass": frozenset({"lgbm", "xgb", "catboost", "logreg", "extratrees", "rf", "hgb"})
    | foundation.NAMES,
    "regression": frozenset({"lgbm", "xgb", "catboost", "ridge", "extratrees", "rf", "hgb"})
    | foundation.NAMES,
    "survival": frozenset({"xgb", "coxnet", "coxph", "rsf", "gbsa"}),
}


def _linear_prep(scale: bool = True):
    numeric = make_pipeline(SimpleImputer(strategy="median"),
                            *( [StandardScaler()] if scale else []))
    categorical = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                              sparse_output=False))
    return make_column_transformer(
        (numeric, make_column_selector(dtype_include=np.number)),
        (categorical, make_column_selector(dtype_include=["object", "category", "bool"])),
    )


def _tree_prep():
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    return make_column_transformer(
        (SimpleImputer(strategy="median", keep_empty_features=True),
         make_column_selector(dtype_include=np.number)),
        (categorical, make_column_selector(dtype_include=["object", "category", "bool"])),
    )


def _cat_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("category",
                                                                                 "bool")]


class Model:
    """Thin adapter giving every estimator the same fit/predict contract."""

    def __init__(self, name: str, task: str, estimator: Any):
        self.name, self.task, self.estimator = name, task, estimator
        self.feature_names: list[str] = []

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.name != "catboost":
            return X
        cats = _cat_cols(X)
        return X.assign(**{c: X[c].astype(str) for c in cats})

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame | None = None,
            y_val: np.ndarray | None = None, **fit_kwargs: Any) -> Model:
        self.feature_names = list(X.columns)
        Xp = self._prep(X)
        if self.name == "catboost":
            fit_kwargs.setdefault("cat_features", _cat_cols(X))
        if self.task == "survival":
            y = _survival_target(self.name, y)
        if X_val is not None and fit_kwargs.pop("use_eval_set", False):
            fit_kwargs["eval_set"] = [(self._prep(X_val), y_val)]
        if isinstance(self.estimator, Pipeline) and "sample_weight" in fit_kwargs:
            last_step = self.estimator.steps[-1][0]  # Pipeline routes weights per step
            fit_kwargs[f"{last_step}__sample_weight"] = fit_kwargs.pop("sample_weight")
        self.estimator.fit(Xp, y, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xp = self._prep(X[self.feature_names])
        if self.task == "binary":
            return self.estimator.predict_proba(Xp)[:, 1]
        if self.task == "multiclass":
            return self.estimator.predict_proba(Xp)
        return np.asarray(self.estimator.predict(Xp), dtype=float)

    def feature_importance(self) -> pd.Series:
        """Native importances when available (gain for GBMs), else empty."""
        est = self.estimator
        if hasattr(est, "feature_importances_") and len(est.feature_importances_) == len(
                self.feature_names):
            return pd.Series(est.feature_importances_, index=self.feature_names).sort_values(
                ascending=False)
        return pd.Series(dtype=float)


def _survival_target(name: str, y: np.ndarray):
    time, event = np.asarray(y)[:, 0].astype(float), np.asarray(y)[:, 1].astype(bool)
    if name == "xgb":  # survival:cox convention: negative time = right-censored
        return np.where(event, time, -time)
    from sksurv.util import Surv

    return Surv.from_arrays(event=event, time=time)


def _estimator(name: str, task: str, p: dict[str, Any], seed: int) -> Any:
    if name in foundation.NAMES:  # pretrained in-context learners (optional extra "fm")
        return foundation.ICLEstimator(name, task, p, seed)
    clf = task in ("binary", "multiclass")
    if name == "lgbm":
        import lightgbm as lgb

        return (lgb.LGBMClassifier if clf else lgb.LGBMRegressor)(random_state=seed, **p)
    if name == "xgb":
        import xgboost as xgb

        if task == "survival":
            return xgb.XGBRegressor(objective="survival:cox", random_state=seed, **p)
        return (xgb.XGBClassifier if clf else xgb.XGBRegressor)(random_state=seed, **p)
    if name == "catboost":
        import catboost as cb

        return (cb.CatBoostClassifier if clf else cb.CatBoostRegressor)(random_seed=seed, **p)
    return _sklearn_estimator(name, task, p, seed)


def _sklearn_estimator(name: str, task: str, p: dict[str, Any], seed: int) -> Any:
    from sklearn import ensemble, linear_model

    clf = task in ("binary", "multiclass")
    if name == "ridge":
        return make_pipeline(_linear_prep(), linear_model.Ridge(**p))
    if name == "logreg":
        return make_pipeline(_linear_prep(), linear_model.LogisticRegression(**p))
    if name in ("extratrees", "rf"):
        kind = "ExtraTrees" if name == "extratrees" else "RandomForest"
        cls = getattr(ensemble, f"{kind}{'Classifier' if clf else 'Regressor'}")
        return make_pipeline(_tree_prep(), cls(random_state=seed, **p))
    if name == "hgb":
        cls = (ensemble.HistGradientBoostingClassifier if clf
               else ensemble.HistGradientBoostingRegressor)
        return cls(random_state=seed, **p)
    return _survival_estimator(name, p, seed)


def _survival_estimator(name: str, p: dict[str, Any], seed: int) -> Any:
    from sksurv import ensemble as sens
    from sksurv import linear_model as slin

    if name == "coxnet":
        return make_pipeline(_linear_prep(), slin.CoxnetSurvivalAnalysis(**p))
    if name == "coxph":
        return make_pipeline(_linear_prep(), slin.CoxPHSurvivalAnalysis(**p))
    if name == "rsf":
        return make_pipeline(_tree_prep(), sens.RandomSurvivalForest(random_state=seed, **p))
    if name == "gbsa":
        return make_pipeline(_tree_prep(),
                             sens.GradientBoostingSurvivalAnalysis(random_state=seed, **p))
    raise ValueError(f"unknown model {name!r}")


def make_model(name: str, task: str, params: dict[str, Any] | None = None,
               seed: int = 42) -> Model:
    """Build a model with DEFAULTS[name] (or a PRESETS entry) updated by ``params``."""
    if name in PRESETS:
        base, preset = PRESETS[name]
        return make_model(base, task, {**preset, **(params or {})}, seed)
    if name not in SUPPORTED.get(task, frozenset()):
        raise ValueError(f"model {name!r} not supported for task {task!r}; "
                         f"use one of {sorted(SUPPORTED.get(task, []))}")
    merged = {**DEFAULTS.get(name, {}), **(params or {})}
    if name == "hgb" and task == "regression":
        merged.setdefault("loss", "squared_error")
    return Model(name, task, _estimator(name, task, merged, seed))
