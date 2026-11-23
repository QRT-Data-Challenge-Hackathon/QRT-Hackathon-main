import numpy as np
import pytest

from qrtc.metrics import METRICS, get_metric


def test_accuracy_thresholds_probabilities():
    assert get_metric("accuracy")([1, 0, 1, 0], [0.9, 0.1, 0.4, 0.6]) == 0.5


def test_accuracy_multiclass_argmax():
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    assert get_metric("accuracy")([0, 2], proba) == 1.0


def test_weighted_accuracy_uses_abs_weight():
    score = get_metric("weighted_accuracy")([1, 0], [0.9, 0.9], weight=[-3.0, 1.0])
    assert score == pytest.approx(0.75)


def test_weighted_accuracy_requires_weight():
    with pytest.raises(ValueError):
        get_metric("weighted_accuracy")([1], [1.0])


def test_mean_spearman_averages_per_group():
    y = [1, 2, 3, 1, 2, 3]
    pred = [1, 2, 3, 3, 2, 1]
    groups = [0, 0, 0, 1, 1, 1]
    assert get_metric("mean_spearman")(y, pred, groups=groups) == pytest.approx(0.0)


def test_cindex_higher_risk_means_earlier_event():
    y = np.array([[1.0, 1], [2.0, 1], [3.0, 1], [4.0, 1]])
    assert get_metric("cindex")(y, [4, 3, 2, 1]) == pytest.approx(1.0)
    assert get_metric("cindex")(y, [1, 2, 3, 4]) == pytest.approx(0.0)


def test_ipcw_cindex_perfect_ranking():
    rng = np.random.default_rng(0)
    time = rng.uniform(0.1, 6, 200)
    y = np.c_[time, rng.random(200) < 0.7]
    assert get_metric("ipcw_cindex")(y, -time, tau=7.0) == pytest.approx(1.0)


def test_zero_mean_r2_perfect_and_zero():
    y = np.array([1.0, -2.0, 3.0])
    assert get_metric("zero_mean_r2")(y, y) == 1.0
    assert get_metric("zero_mean_r2")(y, np.zeros(3)) == 0.0


def test_direction_flags():
    assert METRICS["rmse"].better(0.1, 0.2)
    assert METRICS["accuracy"].better(0.6, 0.5)


def test_logloss_multiclass():
    proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    assert get_metric("logloss")([0, 1], proba) == pytest.approx(-np.log(0.8))
