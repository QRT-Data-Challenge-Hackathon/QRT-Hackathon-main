"""CLI smoke tests: every command on a synthetic dataset, isolated from the real repo dirs."""

import pytest

from qrtc import cli, config, tracking


@pytest.fixture
def stock_cfg(spec_for, tmp_path, monkeypatch):
    spec_for("stock_sign")  # writes tmp_path/stock_sign.toml pointing at temp data
    monkeypatch.setenv("QRTC_CONFIG", str(tmp_path / "stock_sign.toml"))
    return tmp_path


def test_baseline_lb_blend_and_scoring(stock_cfg, capsys):
    params = '{"n_estimators": 20}'
    assert cli.main(["baseline", "--model", "lgbm", "--params", params]) == 0
    assert cli.main(["baseline", "--model", "logreg"]) == 0
    runs = tracking.leaderboard()["run_id"].tolist()
    assert len(runs) == 2
    assert cli.main(["lb"]) == 0
    assert cli.main(["blend", *runs, "--method", "hill"]) == 0
    assert cli.main(["record-lb", runs[0], "0.51"]) == 0
    sub = sorted(config.SUBMISSIONS_DIR.glob("*.csv"))[0]
    assert cli.main(["check-sub", str(sub)]) == 0
    assert cli.main(["score-hidden", str(sub)]) == 0
    assert "hidden accuracy" in capsys.readouterr().out


def test_eda_runs(stock_cfg, capsys):
    assert cli.main(["eda"]) == 0
    assert "leakage flags" in capsys.readouterr().out


def test_bad_config_returns_error_code(tmp_path, monkeypatch):
    monkeypatch.setenv("QRTC_CONFIG", str(tmp_path / "missing.toml"))
    assert cli.main(["eda"]) == 2


@pytest.mark.parametrize("kind, model", [("survival", "coxph"), ("football", "logreg"),
                                         ("allocation", "ridge_unsupported")])
def test_baseline_other_tasks(spec_for, tmp_path, monkeypatch, kind, model):
    spec_for(kind)
    monkeypatch.setenv("QRTC_CONFIG", str(tmp_path / f"{kind}.toml"))
    if model.endswith("unsupported"):
        assert cli.main(["baseline", "--model", "ridge"]) == 2
        return
    assert cli.main(["baseline", "--model", model]) == 0
    sub = sorted(config.SUBMISSIONS_DIR.glob("*.csv"))[-1]
    assert cli.main(["score-hidden", str(sub)]) == 0
