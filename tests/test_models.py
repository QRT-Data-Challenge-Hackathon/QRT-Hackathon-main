"""Model adapter: sample weights reach every estimator, including sklearn Pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qrtc.models import make_model


@pytest.fixture()
def weighted_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = (X["a"] + 0.5 * rng.normal(size=400) > 0).astype(int).to_numpy()
    w = np.where(X["b"] > 0, 5.0, 0.1)
    return X, y, w


@pytest.mark.parametrize("name", ["logreg", "extratrees", "lgbm_shallow"])
def test_fit_accepts_sample_weight(name, weighted_data):
    X, y, w = weighted_data
    pred = make_model(name, "binary", seed=0).fit(X, y, sample_weight=w).predict(X)
    assert pred.shape == (len(X),) and np.isfinite(pred).all()


def test_ridge_regression_accepts_sample_weight(weighted_data):
    X, _, w = weighted_data
    target = X["a"].to_numpy() * 2.0
    pred = make_model("ridge", "regression", seed=0).fit(X, target, sample_weight=w).predict(X)
    assert np.corrcoef(pred, target)[0, 1] > 0.99


def test_pipeline_weights_change_the_fit(weighted_data):
    X, y, w = weighted_data
    flipped = np.where(X["b"] > 0, 1 - y, y)  # heavy rows disagree with light rows
    plain = make_model("logreg", "binary", seed=0).fit(X, flipped).predict(X)
    heavy = make_model("logreg", "binary", seed=0).fit(X, flipped, sample_weight=w).predict(X)
    assert not np.allclose(plain, heavy)
