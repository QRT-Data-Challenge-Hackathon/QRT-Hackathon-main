"""Foundation-model adapters (qrtc.foundation, qrtc.fm_worker) with fake backends.

The real backends need the optional ``fm`` extra and downloaded weights, so CI tests the adapter
and the worker process with fakes. ``QRTC_FM_TESTS=1 uv run pytest tests/test_foundation.py``
also runs the real models on a tiny table (after ``make models``).
"""

import io as stdio
import os
import pickle

import fm_fakes
import numpy as np
import pandas as pd
import pytest

from qrtc import fm_worker, foundation, io
from qrtc.cv import make_splits
from qrtc.features import auto_categoricals, to_category
from qrtc.models import make_model
from qrtc.train import run_cv


def _use(monkeypatch, backend, isolate):
    backend.built = []
    monkeypatch.setattr(foundation, "BUILDERS", {name: backend for name in foundation.NAMES})
    monkeypatch.setattr(foundation, "available", lambda name: True)
    monkeypatch.setattr(foundation, "ISOLATE", isolate)
    return backend


@pytest.fixture
def fake(monkeypatch):
    return _use(monkeypatch, fm_fakes.FakeBackend, isolate=False)


@pytest.fixture
def worker_with(monkeypatch):
    yield lambda backend: _use(monkeypatch, backend, isolate=True)
    foundation.shutdown_worker()


def _table(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "cat": rng.choice(["x", "y", "z"], n)})
    return X, (X["a"] > 0).astype(int).to_numpy()


def test_encoder_codes_categoricals_and_maps_unknown_to_nan():
    train = pd.DataFrame({"num": [1.0, 2.0, np.nan], "cat": ["b", "a", None]})
    test = pd.DataFrame({"num": [3.0, 4.0], "cat": ["a", "new"]})
    enc = foundation.Encoder.fit(train)
    assert enc.categorical_indices == [1]
    np.testing.assert_array_equal(enc.transform(train)[:, 1], [1.0, 0.0, np.nan])
    np.testing.assert_array_equal(enc.transform(test), [[3.0, 0.0], [4.0, np.nan]])
    assert train["cat"].tolist() == ["b", "a", None]  # input not mutated


def test_encoder_handles_pandas_category_dtype():
    X = pd.DataFrame({"c": pd.Categorical(["u", "v", "u"])})
    np.testing.assert_array_equal(foundation.Encoder.fit(X).transform(X)[:, 0], [0.0, 1.0, 0.0])


def test_context_rows_keeps_everything_when_small():
    np.testing.assert_array_equal(foundation.context_rows(np.zeros(5), "binary", 10, 0),
                                  np.arange(5))


def test_context_rows_stratifies_and_keeps_rare_classes():
    y = np.array([0] * 900 + [1] * 99 + [2])
    rows = foundation.context_rows(y, "multiclass", 100, seed=3)
    assert set(y[rows]) == {0, 1, 2}
    assert abs((y[rows] == 0).mean() - 0.9) < 0.02
    np.testing.assert_array_equal(rows, foundation.context_rows(y, "multiclass", 100, seed=3))
    assert np.all(np.diff(rows) > 0)


def test_context_rows_regression_has_exact_size():
    rows = foundation.context_rows(np.arange(1000.0), "regression", 250, seed=0)
    assert len(rows) == 250 and len(np.unique(rows)) == 250


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_make_model_returns_task_shaped_predictions(fake, task):
    X, y = _table()
    if task == "multiclass":
        y = np.digitize(X["b"].to_numpy(), [-0.5, 0.5])
    pred = make_model("tabicl", task).fit(X, y).predict(X)
    assert pred.shape == ((len(X), 3) if task == "multiclass" else (len(X),))
    assert fake.built[0].cat_idx == [2]
    assert fake.built[0].X.dtype == float


def test_classifier_predict_returns_labels(fake):
    X, y = _table()
    assert set(foundation.ICLEstimator("tabdpt", "binary").fit(X, y).predict(X)) <= {0, 1}
    with pytest.raises(AttributeError):
        foundation.ICLEstimator("tabdpt", "regression").fit(X, y).predict_proba(X)


def test_bags_average_models_fit_on_capped_subsamples(fake):
    X, y = _table(n=200)
    est = foundation.ICLEstimator("tabpfn", "binary", {"max_rows": 50, "n_bags": 3}).fit(X, y)
    assert len(est.context_) == 3
    assert est.predict_proba(X).shape == (200, 2)
    assert all(45 <= len(m.y) <= 55 for m in fake.built)
    assert len({m.seed for m in fake.built}) == 3


def test_bags_ignored_when_data_fits_in_context(fake):
    X, y = _table(n=40)
    est = foundation.ICLEstimator("tabicl", "binary", {"max_rows": 100, "n_bags": 4}).fit(X, y)
    est.predict(X)
    assert len(est.context_) == 1 and len(fake.built) == 1 and len(fake.built[0].y) == 40


def test_backend_params_exclude_wrapper_params(fake):
    X, y = _table()
    foundation.ICLEstimator("tabpfn", "binary", {"n_estimators": 4}).fit(X, y).predict(X)
    assert fake.built[0].params == {"version": "v2", "n_estimators": 4}
    foundation.ICLEstimator("tabicl", "binary", {"n_estimators": 2}).fit(X, y).predict(X)
    assert fake.built[1].params == {"n_estimators": 2}


def test_sample_weight_is_ignored_with_a_warning(fake):
    X, y = _table()
    with pytest.warns(UserWarning, match="ignores sample_weight"):
        foundation.ICLEstimator("tabicl", "binary").fit(X, y, sample_weight=np.ones(len(y)))


@pytest.mark.parametrize("params, match", [({"version": "v3"}, "tabpfn param"),
                                           ({"max_rows": 1}, "max_rows"),
                                           ({"n_bags": 0}, "n_bags")])
def test_bad_params_are_rejected(fake, params, match):
    with pytest.raises(ValueError, match=match):
        foundation.ICLEstimator("tabicl", "binary", params)


def test_unknown_name_and_task_are_rejected(fake):
    with pytest.raises(ValueError, match="unknown foundation model"):
        foundation.ICLEstimator("gpt", "binary")
    with pytest.raises(ValueError, match="supports"):
        foundation.ICLEstimator("tabicl", "survival")
    with pytest.raises(ValueError, match="not supported"):
        make_model("tabpfn", "survival")


def test_missing_package_points_to_make_models(monkeypatch):
    monkeypatch.setattr(foundation, "available", lambda name: False)
    with pytest.raises(foundation.MissingModelError, match="make models"):
        make_model("tabicl", "binary")


def test_run_cv_with_a_foundation_model(fake, spec_for):
    spec = spec_for("electricity")
    ds = io.load_dataset(spec)
    feats = io.feature_columns(ds.train, spec)
    X, X_test = to_category(ds.train[feats], ds.test[feats], auto_categoricals(ds.train[feats]))
    splits = make_splits(ds.train, ds.y_vector, spec, test=ds.test)
    res = run_cv(spec, X, ds.y_vector, X_test, splits, "tabdpt", verbose=False)
    assert len(res.test_pred) == len(ds.test) and np.isfinite(res.score)
    assert len(fake.built) == 2 * len(splits)  # one validation + one test call per fold
    assert res.importance.empty


def test_worker_runs_backends_in_another_process(worker_with):
    worker_with(fm_fakes.PidBackend)
    X, _ = _table()
    pred = foundation.ICLEstimator("tabicl", "regression").fit(X, X["a"]).predict(X)
    assert len(set(pred)) == 1 and pred[0] != os.getpid()
    again = foundation.ICLEstimator("tabdpt", "regression").fit(X, X["a"]).predict(X)
    assert again[0] == pred[0]  # one long-lived worker, not one per call


def test_worker_reraises_backend_errors(worker_with):
    worker_with(fm_fakes.FailingBackend)
    X, y = _table()
    with pytest.raises(ValueError, match="bad backend param"):
        foundation.ICLEstimator("tabicl", "binary").fit(X, y).predict(X)


def test_dead_worker_is_reported_and_replaced(worker_with, monkeypatch):
    worker_with(fm_fakes.DyingBackend)
    X, y = _table()
    with pytest.raises(RuntimeError, match=r"worker died \(exit code 3\)"):
        foundation.ICLEstimator("tabicl", "binary").fit(X, y).predict(X)
    worker_with(fm_fakes.FakeBackend)
    assert foundation.ICLEstimator("tabicl", "binary").fit(X, y).predict(X).shape == (60,)


def test_serve_answers_requests_and_returns_errors():
    X, y = np.arange(6.0).reshape(3, 2), np.array([0, 1, 0])
    base = dict(task="binary", params={}, seed=0, cat_idx=[], X=X, y=y, X_query=X)
    requests = stdio.BytesIO()
    pickle.dump({**base, "builder": "fm_fakes:FakeBackend", "method": "predict_proba"},
                requests)
    pickle.dump({**base, "builder": "fm_fakes:FailingBackend", "method": "predict"}, requests)
    requests.seek(0)
    replies = stdio.BytesIO()
    fm_worker.serve(requests, replies)
    replies.seek(0)
    status, proba = pickle.load(replies)
    assert status == "ok" and proba.shape == (3, 2)
    status, err = pickle.load(replies)
    assert status == "error" and isinstance(err, ValueError)


def test_unpicklable_errors_become_runtime_errors():
    class Local(Exception):
        pass

    err = fm_worker._picklable(Local("x"))
    assert isinstance(err, RuntimeError) and "Local: x" in str(err)


REAL = os.environ.get("QRTC_FM_TESTS") == "1"


@pytest.mark.skipif(not REAL, reason="set QRTC_FM_TESTS=1 after `make models`")
@pytest.mark.parametrize("name", sorted(foundation.NAMES))
@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_real_backend_smoke(name, task):
    score = foundation.smoke_test(name, task)
    assert score > (0.5 if task == "regression" else 0.6)


@pytest.mark.skipif(not REAL, reason="set QRTC_FM_TESTS=1 after `make models`")
def test_real_backend_after_lightgbm_does_not_crash():
    X, y = _table(n=300)
    make_model("lgbm", "binary", {"n_estimators": 20}).fit(X.drop(columns="cat"), y)
    assert foundation.smoke_test("tabdpt", "binary") > 0.6
    make_model("lgbm", "binary", {"n_estimators": 20}).fit(X.drop(columns="cat"), y)
