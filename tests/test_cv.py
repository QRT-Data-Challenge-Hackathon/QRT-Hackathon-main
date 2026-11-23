import numpy as np
import pandas as pd
import pytest

from qrtc.cv import date_block_splits, detect_strategy, make_splits, time_splits


def _dates(n_dates=20, per=5):
    return np.repeat(np.arange(n_dates), per)


def test_date_blocks_never_share_a_date():
    dates = _dates()
    for tr, va in date_block_splits(dates, 5):
        assert not set(dates[tr]) & set(dates[va])


def test_date_blocks_cover_every_row_once():
    dates = _dates()
    val = np.concatenate([va for _, va in date_block_splits(dates, 4)])
    assert sorted(val) == list(range(len(dates)))


def test_embargo_removes_neighbouring_dates():
    dates = _dates()
    tr, va = date_block_splits(dates, 4, embargo=2)[1]
    gap = np.abs(dates[tr][:, None] - dates[va][None, :]).min()
    assert gap > 2


def test_time_splits_train_strictly_before_validation():
    dates = _dates()
    for tr, va in time_splits(dates, 4, embargo=1):
        assert dates[tr].max() < dates[va].min() - 1


def test_string_dates_supported():
    dates = np.array([f"DATE_{d:04d}" for d in _dates()])
    assert len(date_block_splits(dates, 5)) == 5


def test_detect_strategy(spec_for):
    spec = spec_for("stock_sign")
    tr = pd.DataFrame({"DATE": [1, 2, 3], "STOCK": [1, 1, 1]})
    assert detect_strategy(tr, pd.DataFrame({"DATE": [4, 5], "STOCK": [1, 1]}), spec) == "time"
    assert detect_strategy(tr, pd.DataFrame({"DATE": [2], "STOCK": [1]}), spec) == "date_blocks"


def test_group_split_keeps_groups_apart(spec_for):
    spec = spec_for("survival")
    train = pd.DataFrame({"CENTER": np.repeat(list("abcdefgh"), 10)})
    for tr, va in make_splits(train, np.zeros(80), spec, "group"):
        assert not set(train.CENTER.iloc[tr]) & set(train.CENTER.iloc[va])


def test_unknown_strategy_raises(spec_for):
    with pytest.raises(ValueError):
        make_splits(pd.DataFrame({"a": range(10)}), np.zeros(10), spec_for("football"), "x")
