import numpy as np
import pandas as pd

from qrtc import features as F


def _panel():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"DATE": np.repeat([0, 1, 2], 4), "SECTOR": [0, 0, 1, 1] * 3,
                         "RET_2": rng.normal(size=12), "RET_1": rng.normal(size=12),
                         "RET_10": rng.normal(size=12)})


def test_col_block_sorts_numerically():
    assert F.col_block(_panel(), "RET") == ["RET_1", "RET_2", "RET_10"]


def test_cs_rank_within_date_and_pure():
    df = _panel()
    before = df.copy()
    out = F.cs_rank(df, ["RET_1"], "DATE")
    pd.testing.assert_frame_equal(df, before)
    assert out.groupby("DATE")["RET_1_csrank"].max().eq(1.0).all()


def test_group_demean_zero_mean():
    out = F.group_demean(_panel(), ["RET_1"], ["DATE", "SECTOR"])
    assert out.groupby(["DATE", "SECTOR"])["RET_1_dm_DATE_SECTOR"].sum().abs().max() < 1e-12


def test_row_stats_and_window_sums():
    out = F.window_sums(F.row_stats(_panel(), ["RET_1", "RET_2"], "R"), ["RET_1", "RET_2"],
                        [1, 2], "R")
    assert np.allclose(out["R_sum2"], out["RET_1"] + out["RET_2"])
    assert "R_std" in out


def test_oof_target_encoding_does_not_use_own_fold():
    train = pd.DataFrame({"c": ["a"] * 4 + ["b"] * 4})
    test = pd.DataFrame({"c": ["a", "b"]})
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    splits = [(np.arange(4, 8), np.arange(0, 4)), (np.arange(0, 4), np.arange(4, 8))]
    tr, te = F.oof_target_encode(train, test, "c", y, splits, smoothing=0.0)
    # category "a" never appears in its own training fold -> falls back to the prior
    assert np.allclose(tr["c_te"], 0.5)
    assert te["c_te"].tolist() == [1.0, 0.0]


def test_to_category_shares_categories():
    tr, te = F.to_category(pd.DataFrame({"s": ["x", "y"]}), pd.DataFrame({"s": ["z"]}), ["s"])
    assert list(tr["s"].cat.categories) == list(te["s"].cat.categories)


def test_rank_target_is_gaussianised_per_group():
    z = F.rank_target(np.array([1.0, 5.0, 3.0, 10.0, 20.0, 30.0]), np.array([0, 0, 0, 1, 1, 1]))
    assert np.allclose(z[:3], z[3:][[0, 2, 1]])
    assert abs(z.mean()) < 1e-9
