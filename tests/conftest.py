"""Shared fixtures: synthetic rehearsal data and specs in a temp dir, isolated run dirs."""

from __future__ import annotations

from pathlib import Path

import pytest

from qrtc import config, synthetic

REHEARSAL = Path(__file__).resolve().parents[1] / "configs" / "rehearsal"


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never write runs/registry/submissions into the real repo during tests."""
    for name in ("RUNS_DIR", "REGISTRY_DIR", "LB_DIR", "SUBMISSIONS_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    monkeypatch.setattr(config, "SHARED_LB_DIR", tmp_path / "runs_dir" / "_lb")
    return tmp_path


@pytest.fixture(scope="session")
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic")
    for kind in synthetic.GENERATORS:
        synthetic.generate(kind, root / kind, seed=1)
    return root


@pytest.fixture
def spec_for(data_root: Path, tmp_path: Path):
    """Load a rehearsal config with its data dir pointed at the temp synthetic data."""

    def _load(kind: str) -> config.Spec:
        text = (REHEARSAL / f"{kind}.toml").read_text()
        text = text.replace(f'dir = "data/synthetic/{kind}"', f'dir = "{data_root / kind}"')
        path = tmp_path / f"{kind}.toml"
        path.write_text(text)
        return config.load_spec(path)

    return _load
