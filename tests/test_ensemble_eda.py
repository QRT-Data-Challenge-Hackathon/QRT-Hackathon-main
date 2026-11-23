import numpy as np
import pandas as pd

from qrtc import eda, ensemble, postprocess
from qrtc.metrics import get_metric


def test_hill_climb_prefers_the_good_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    good = np.clip(y * 0.6 + rng.random(2000) * 0.5, 0, 1)
    noise = rng.random(2000)
    acc = get_metric("accuracy")
    w = ensemble.hill_climb([good, noise], lambda p: acc(y, p))
    assert w[0] > w[1]


def test_weighted_blend_normalises_and_ranks():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([30.0, 20.0, 10.0])
    assert np.allclose(ensemble.weighted_blend([a, a], [2, 2]), a)
    assert np.allclose(ensemble.weighted_blend([a, b], [1, 1], rank=True), [2 / 3] * 3)


def test_common_rows_skips_nan():
    assert ensemble.common_rows([np.array([1, np.nan, 3]), np.array([1, 2, np.nan])]).tolist() == [0]


def test_median_split_within_balances_each_group():
    labels = postprocess.median_split_within(np.arange(8.0), np.repeat([0, 1], 4))
    assert labels.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


def test_best_threshold_finds_shift():
    y = np.array([0, 0, 1, 1])
    t, score = postprocess.best_threshold(y, np.array([0.6, 0.65, 0.7, 0.8]),
                                          get_metric("accuracy"), grid=np.array([0.5, 0.67]))
    assert t == 0.67 and score == 1.0


def test_lag_chain_links_rebuilds_order():
    rets = np.random.default_rng(0).normal(size=6)
    df = pd.DataFrame({"S": [1, 1], "RET_1": [rets[1], rets[2]], "RET_2": [rets[0], rets[1]],
                       "RET_3": [rets[5], rets[0]]})
    links = eda.lag_chain_links(df, "S", ["RET_1", "RET_2", "RET_3"])
    assert links[["row", "next_row"]].values.tolist() == [[0, 1]]


def test_leakage_checks_flags_constant_column(spec_for):
    spec = spec_for("electricity")
    train = pd.DataFrame({"ID": np.arange(50), "DAY_ID": np.arange(50), "k": 1})
    flags = eda.leakage_checks(train, np.random.default_rng(0).normal(size=50), spec)
    assert any("constant" in f for f in flags)
    assert any("id order" in f for f in flags)


def test_signal_scan_per_date():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=600), "b": rng.normal(size=600)})
    y = X["a"] + rng.normal(size=600)
    out = eda.signal_scan(X, y.to_numpy(), np.repeat(np.arange(30), 20))
    assert out.index[0] == "a" and out.loc["a", "t_stat"] > 5
