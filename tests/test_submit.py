import numpy as np
import pandas as pd
import pytest

from qrtc import submit


def test_binary_labels_use_original_classes(spec_for):
    spec = spec_for("stock_sign")
    sub = submit.build_submission(spec, np.array([1, 2]), np.array([0.7, 0.2]), (False, True))
    assert sub["RET"].tolist() == [True, False]


def test_multiclass_one_hot_output(spec_for):
    spec = spec_for("football")
    proba = np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
    sub = submit.build_submission(spec, np.array([0, 1]), proba, spec.target_cols)
    assert sub[list(spec.target_cols)].to_numpy().tolist() == [[0, 1, 0], [1, 0, 0]]


def test_survival_writes_risk_score(spec_for):
    spec = spec_for("survival")
    sub = submit.build_submission(spec, np.array(["P1"]), np.array([0.3]))
    assert list(sub.columns) == ["ID", "risk_score"]


@pytest.mark.parametrize("bad, problem", [
    (pd.DataFrame({"ID": [1, 1], "TARGET": [0, 1]}), "duplicate"),
    (pd.DataFrame({"ID": [1], "TARGET": [0]}), "rows"),
    (pd.DataFrame({"ID": [1, 2], "TARGET": [0, np.nan]}), "NaN"),
    (pd.DataFrame({"ID": [1, 3], "TARGET": [0, 1]}), "id set"),
])
def test_validation_catches_problems(spec_for, bad, problem):
    spec = spec_for("electricity")
    problems = submit.validate_submission(bad, spec, np.array([1, 2]))
    assert any(problem in p for p in problems)


def test_write_refuses_invalid(spec_for, tmp_path):
    spec = spec_for("electricity")
    with pytest.raises(submit.SubmissionError):
        submit.write_submission(pd.DataFrame({"ID": [1], "TARGET": [np.nan]}), spec,
                                np.array([1]), "x", out_dir=tmp_path)
