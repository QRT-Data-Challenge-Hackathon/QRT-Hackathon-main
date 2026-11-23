"""Tests for .claude/hooks/guard_upload.py (browser uploads need a ledger reservation)."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import pytest
from conftest import REPO_ROOT

UPLOAD_GUARD = REPO_ROOT / ".claude" / "hooks" / "guard_upload.py"


@pytest.fixture(scope="module")
def upload_guard():
    spec = importlib.util.spec_from_file_location("guard_upload", UPLOAD_GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_ledger(root, entries):
    (root / "submissions").mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(e) + "\n" for e in entries)
    (root / "submissions" / "ledger.jsonl").write_text(lines)


@pytest.fixture
def sub(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("ID,RET\n1,1\n")
    return path


def test_reserved_file_allowed(upload_guard, tmp_path, sub):
    write_ledger(tmp_path, [{"upload_id": "U001", "sha256": sha(sub), "status": "pending"}])
    assert upload_guard.evaluate({"paths": [str(sub)]}, tmp_path) is None


@pytest.mark.parametrize("status", ["scored", "submitted", "failed"])
def test_finished_reservation_blocked(upload_guard, tmp_path, sub, status):
    write_ledger(tmp_path, [{"upload_id": "U001", "sha256": sha(sub), "status": "pending"},
                            {"upload_id": "U001", "status": status}])
    assert "qrtc upload" in upload_guard.evaluate({"paths": [str(sub)]}, tmp_path)


def test_unreserved_missing_or_unledgered_blocked(upload_guard, tmp_path, sub):
    assert "no submissions ledger" in upload_guard.evaluate({"paths": [str(sub)]}, tmp_path)
    write_ledger(tmp_path, [{"upload_id": "U001", "sha256": "other", "status": "pending"}])
    assert "no pending reservation" in upload_guard.evaluate({"paths": [str(sub)]}, tmp_path)
    missing = str(tmp_path / "nope.csv")
    assert "not a file" in upload_guard.evaluate({"paths": [missing]}, tmp_path)


def test_corrupt_ledger_blocked(upload_guard, tmp_path, sub):
    (tmp_path / "submissions").mkdir()
    (tmp_path / "submissions" / "ledger.jsonl").write_text("{not json\n")
    assert "cannot read" in upload_guard.evaluate({"paths": [str(sub)]}, tmp_path)


def run_hook(payload, shared):
    env = {**os.environ, "QRTC_SHARED_DIR": str(shared)}
    return subprocess.run([sys.executable, str(UPLOAD_GUARD)], input=json.dumps(payload),
                          capture_output=True, text=True, env=env, check=False)


def test_main_blocks_and_allows(tmp_path, sub):
    payload = {"tool_name": "mcp__claude-in-chrome__file_upload", "cwd": str(tmp_path),
               "tool_input": {"paths": [str(sub)], "ref": "ref_1", "tabId": 1}}
    blocked = run_hook(payload, tmp_path)
    assert blocked.returncode == 2 and "qrtc upload" in blocked.stderr
    write_ledger(tmp_path, [{"upload_id": "U001", "sha256": sha(sub), "status": "pending"}])
    assert run_hook(payload, tmp_path).returncode == 0
    assert run_hook({"tool_name": "Bash", "tool_input": {}}, tmp_path).returncode == 0


def test_main_fails_closed_on_bad_input(tmp_path):
    result = subprocess.run([sys.executable, str(UPLOAD_GUARD)], input="{", text=True,
                            capture_output=True, check=False)
    assert result.returncode == 2
