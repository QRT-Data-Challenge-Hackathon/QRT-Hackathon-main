from pathlib import Path

import pytest

from qrtc import config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "tests" / "fixtures" / "specs").glob("*.toml")))
def test_fixture_specs_load(path):
    spec = config.load_spec(path)
    assert spec.task in config.TASKS
    assert spec.target_cols


def test_default_competition_toml_loads():
    assert config.load_spec(ROOT / "competition.toml").id_col


def _write(tmp_path, comp: str, files: str = 'dir = "d"\nx_train = "a.csv"\nx_test = "b.csv"'):
    path = tmp_path / "c.toml"
    path.write_text(f"[competition]\n{comp}\n[files]\n{files}\n")
    return path


def test_unknown_metric_rejected(tmp_path):
    path = _write(tmp_path, 'task="binary"\nmetric="nope"\nid_col="ID"\ntarget_cols=["y"]')
    with pytest.raises(config.SpecError, match="unknown metric"):
        config.load_spec(path)


def test_missing_fields_reported(tmp_path):
    path = _write(tmp_path, 'task="binary"')
    with pytest.raises(config.SpecError, match="missing"):
        config.load_spec(path)


def test_survival_needs_two_targets(tmp_path):
    path = _write(tmp_path, 'task="survival"\nmetric="cindex"\nid_col="ID"\ntarget_cols=["t"]')
    with pytest.raises(config.SpecError, match="survival"):
        config.load_spec(path)


def test_relative_data_dir_resolves_to_shared_root(tmp_path):
    path = _write(tmp_path, 'task="binary"\nmetric="accuracy"\nid_col="ID"\ntarget_cols=["y"]')
    assert config.load_spec(path).data_dir == config.SHARED_ROOT / "d"
