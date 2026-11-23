"""Integration: load -> features -> CV -> track -> blend -> submit, for every problem shape."""

import numpy as np
import pytest

from qrtc import io, submit, tracking
from qrtc.cv import make_splits
from qrtc.features import auto_categoricals, to_category
from qrtc.train import run_cv

FAST = {"lgbm": {"n_estimators": 30}, "logreg": {}, "coxph": {}, "xgb": {"n_estimators": 30},
        "ridge": {}}
CASES = [("stock_sign", "lgbm"), ("allocation", "lgbm"), ("electricity", "ridge"),
         ("football", "logreg"), ("survival", "coxph"), ("survival", "xgb")]


def _prepare(spec):
    ds = io.load_dataset(spec)
    feats = io.feature_columns(ds.train, spec)
    X, X_test = to_category(ds.train[feats], ds.test[feats], auto_categoricals(ds.train[feats]))
    return ds, X, X_test


@pytest.mark.parametrize("kind, model", CASES)
def test_end_to_end(spec_for, kind, model):
    spec = spec_for(kind)
    ds, X, X_test = _prepare(spec)
    splits = make_splits(ds.train, ds.y_vector, spec, test=ds.test)
    groups = ds.train[spec.date_col].to_numpy() if spec.date_col else None
    res = run_cv(spec, X, ds.y_vector, X_test, splits, model, FAST[model], groups=groups,
                 verbose=False)
    assert np.isfinite(res.score)
    assert len(res.test_pred) == len(ds.test)
    run_id = tracking.save_run(res, spec, f"{kind}-{model}", ds.train[spec.id_col].to_numpy(),
                               ds.test[spec.id_col].to_numpy())
    assert tracking.load_preds(run_id)["test"].shape[0] == len(ds.test)
    sub = submit.build_submission(spec, ds.test[spec.id_col].to_numpy(), res.test_pred,
                                  ds.classes)
    path = submit.write_submission(sub, spec, ds.test[spec.id_col].to_numpy(), run_id)
    assert path.exists()


def test_allocation_target_is_binarised(spec_for):
    ds = io.load_dataset(spec_for("allocation"))
    assert set(np.unique(ds.y_vector)) == {0, 1}
    assert ds.y_raw is not None and ds.y_raw.iloc[:, 0].dtype == float


def test_leaderboard_lists_runs_and_lb_scores(spec_for):
    spec = spec_for("electricity")
    ds, X, X_test = _prepare(spec)
    splits = make_splits(ds.train, ds.y_vector, spec, "kfold")
    res = run_cv(spec, X, ds.y_vector, X_test, splits, "ridge", verbose=False)
    run_id = tracking.save_run(res, spec, "lb-test", ds.train.ID.to_numpy(), ds.test.ID.to_numpy())
    tracking.record_lb(run_id, 0.123)
    table = tracking.leaderboard()
    assert table.loc[table.run_id == run_id, "public_lb"].item() == 0.123


def test_load_preds_rejects_other_data(spec_for):
    spec = spec_for("electricity")
    ds, X, X_test = _prepare(spec)
    splits = make_splits(ds.train, ds.y_vector, spec, "kfold")
    res = run_cv(spec, X, ds.y_vector, X_test, splits, "ridge", verbose=False)
    run_id = tracking.save_run(res, spec, "ids", ds.train.ID.to_numpy(), ds.test.ID.to_numpy())
    with pytest.raises(ValueError, match="do not match"):
        tracking.load_preds(run_id, ds.train.ID.to_numpy()[::-1])
