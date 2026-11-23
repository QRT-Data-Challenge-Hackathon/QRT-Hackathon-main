"""Agent uploads: designated uploader, validation, budget, ledger, backends, CLI."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrtc import cli, config, io, platforms, submit
from qrtc.config import SpecError
from qrtc.platforms import UploadConfig, UploadError, load_upload_config, public_mask
from qrtc.upload import budget, confirm, ledger_path, read_ledger, upload_file, upload_table


@pytest.fixture
def env(spec_for, tmp_path, monkeypatch):
    spec = spec_for("stock_sign")
    monkeypatch.setenv("QRTC_CONFIG", str(tmp_path / "stock_sign.toml"))
    monkeypatch.setenv("AGENT_NAME", "bot-1")
    return spec


def make_sub(spec: config.Spec, tag: str = "sub", seed: int = 0) -> Path:
    test_ids = io.read_features(spec, spec.x_test)[spec.id_col].to_numpy()
    pred = np.random.default_rng(seed).random(len(test_ids))
    return submit.write_submission(submit.build_submission(spec, test_ids, pred), spec,
                                   test_ids, tag)


def cfg(**kwargs) -> UploadConfig:
    return UploadConfig(**{"backend": "local", "uploaders": ("bot-1",), **kwargs})


# --- config ------------------------------------------------------------------------------


def test_rehearsal_config_has_upload_section(env):
    loaded = load_upload_config()
    assert loaded.backend == "local" and loaded.allows("bot-1")


@pytest.mark.parametrize("body", ['backend = "ftp"', 'bakend = "local"',
                                  "public_fraction = 1.5", "max_total = 2\nreserve_final = 2"])
def test_bad_upload_config_rejected(tmp_path, body):
    path = tmp_path / "c.toml"
    path.write_text(f"[upload]\n{body}\n")
    with pytest.raises(SpecError):
        load_upload_config(path)


def test_missing_upload_section_allows_nobody(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[competition]\n")
    assert not load_upload_config(path).allows("anyone")


# --- guards before anything is uploaded -------------------------------------------------


def test_non_uploader_refused(env):
    with pytest.raises(UploadError, match="not a designated uploader"):
        upload_file(make_sub(env), cfg=cfg(uploaders=("someone-else",)))
    assert read_ledger() == []


def test_invalid_file_refused(env, tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"ID": [1, 2], "RET": [0, 1]}).to_csv(bad, index=False)
    with pytest.raises(UploadError, match="invalid"):
        upload_file(bad, cfg=cfg())
    assert read_ledger() == []


def test_duplicate_refused_unless_allowed(env):
    path = make_sub(env)
    upload_file(path, cfg=cfg())
    with pytest.raises(UploadError, match="already uploaded"):
        upload_file(path, cfg=cfg())
    assert upload_file(path, cfg=cfg(), allow_duplicate=True)["upload_id"] == "U002"


def test_dry_run_writes_nothing(env):
    record = upload_file(make_sub(env), cfg=cfg(), dry_run=True)
    assert record["status"] == "dry-run" and not ledger_path().exists()


# --- budget ------------------------------------------------------------------------------


def test_reserve_final_slots(env):
    limited = cfg(max_total=3, reserve_final=1)
    upload_file(make_sub(env, "a", 1), cfg=limited)
    upload_file(make_sub(env, "b", 2), cfg=limited)
    with pytest.raises(UploadError, match="kept for finals"):
        upload_file(make_sub(env, "c", 3), cfg=limited)
    assert upload_file(make_sub(env, "c", 3), cfg=limited, final=True)["final"] is True
    with pytest.raises(UploadError, match="no submissions left"):
        upload_file(make_sub(env, "d", 4), cfg=limited, final=True)


def test_budget_counts_only_live_uploads_of_this_competition():
    records = [
        {"competition": "c", "status": "scored", "day": "2026-11-24"},
        {"competition": "c", "status": "pending", "day": "2026-11-23"},
        {"competition": "c", "status": "failed", "day": "2026-11-24"},
        {"competition": "other", "status": "scored", "day": "2026-11-24"},
    ]
    state = budget(records, cfg(max_total=5, max_per_day=2), "c", today="2026-11-24")
    assert (state.used, state.used_today, state.left) == (2, 1, 1)
    assert budget(records, cfg(), "c").left is None
    assert budget(records, cfg(), "c").refusal(final=False) is None


# --- backends ----------------------------------------------------------------------------


def test_local_upload_scores_public_and_hides_private(env):
    run_dir = config.RUNS_DIR / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text("{}")
    record = upload_file(make_sub(env, "r1"), message="first", cfg=cfg())
    assert record["status"] == "scored" and record["run_id"] == "r1"
    assert 0.3 < record["public"] < 0.7 and 0.3 < record["private"] < 0.7
    shared = json.loads((config.SHARED_LB_DIR / "r1.json").read_text())
    assert shared["public_lb"] == record["public"]
    assert not (config.LB_DIR / "r1.json").exists()  # rehearsal scores are never committed
    assert "private" not in upload_table(env.name).columns
    assert "private" in upload_table(env.name, reveal=True).columns


def test_public_mask_is_deterministic_partition():
    ids = pd.Series(range(2000))
    mask = public_mask(ids, 0.3, seed=7)
    assert np.array_equal(mask, public_mask(ids, 0.3, seed=7))
    assert 0.25 < mask.mean() < 0.35


def test_browser_upload_is_two_phase(env):
    browser = cfg(backend="browser", url="https://example.test/submit", max_total=2)
    record = upload_file(make_sub(env, "a", 1), cfg=browser)
    assert record["status"] == "pending" and "upload-confirm U001" in record["detail"]
    assert budget(read_ledger(), browser, env.name).used == 1
    assert confirm("U001", score=0.51, cfg=browser)["status"] == "scored"
    with pytest.raises(UploadError, match="already scored"):
        confirm("U001", score=0.5, cfg=browser)
    upload_file(make_sub(env, "b", 2), cfg=browser)
    assert confirm("U002", failed="form rejected", cfg=browser)["status"] == "failed"
    assert budget(read_ledger(), browser, env.name).used == 1


def test_confirm_checks_arguments_and_ids(env):
    with pytest.raises(UploadError, match="not both"):
        confirm("U001", score=0.5, failed="x", cfg=cfg())
    with pytest.raises(UploadError, match="unknown upload"):
        confirm("U404", cfg=cfg())
    with pytest.raises(UploadError, match="not a designated uploader"):
        confirm("U001", cfg=cfg(uploaders=()))


def _serve(status: int, body: dict, seen: dict) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (http.server API)
            seen["headers"] = dict(self.headers)
            seen["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_http_upload_sends_file_secret_header_and_reads_score(env, monkeypatch):
    seen: dict = {}
    server = _serve(200, {"result": {"score": 0.523}}, seen)
    monkeypatch.setenv("QRT_TOKEN", "secret-x")
    http = cfg(backend="http", url=f"http://127.0.0.1:{server.server_port}/submit",
               headers_env={"Authorization": "QRT_TOKEN"}, fields={"team": "t1"},
               message_field="comment", score_key="result.score")
    try:
        record = upload_file(make_sub(env), message="hello", cfg=http)
    finally:
        server.shutdown()
    assert record["status"] == "scored" and record["public"] == pytest.approx(0.523)
    assert seen["headers"]["Authorization"] == "secret-x"
    assert b'name="team"' in seen["body"] and b"hello" in seen["body"]
    assert b"ID,RET" in seen["body"]
    assert "secret-x" not in ledger_path().read_text()


def test_http_error_marks_upload_failed(env):
    server = _serve(500, {"error": "boom"}, {})
    http = cfg(backend="http", url=f"http://127.0.0.1:{server.server_port}/submit")
    try:
        with pytest.raises(UploadError, match="HTTP 500"):
            upload_file(make_sub(env), cfg=http)
    finally:
        server.shutdown()
    assert read_ledger()[-1]["status"] == "failed"
    assert budget(read_ledger(), http, env.name).used == 0


@pytest.mark.parametrize("kwargs,match", [
    ({"url": "http://example.com/submit"}, "https"),
    ({"url": "https://example.com/submit", "headers_env": {"X-Key": "MISSING_ENV_VAR"}},
     "MISSING_ENV_VAR"),
])
def test_http_refuses_plain_http_and_missing_secrets(env, monkeypatch, kwargs, match):
    monkeypatch.delenv("MISSING_ENV_VAR", raising=False)
    with pytest.raises(UploadError, match=match):
        upload_file(make_sub(env), cfg=cfg(backend="http", **kwargs))


def test_kaggle_upload_calls_cli(env, tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "kaggle"
    fake.write_text(f'#!/bin/sh\necho "$@" > "{tmp_path}/args"\necho "Successfully submitted"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    record = upload_file(make_sub(env), message="m", cfg=cfg(backend="kaggle", competition="qrt"))
    assert record["status"] == "submitted"
    assert "competitions submit -c qrt -f" in (tmp_path / "args").read_text()
    assert confirm(record["upload_id"], score=0.5, cfg=cfg())["public"] == 0.5


def test_kaggle_without_cli_fails_cleanly(env, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(UploadError, match="kaggle CLI not found"):
        upload_file(make_sub(env), cfg=cfg(backend="kaggle", competition="qrt"))


def test_unexpected_backend_crash_keeps_the_slot(env, monkeypatch):
    def boom(*_args):
        raise RuntimeError("parser exploded")

    monkeypatch.setitem(platforms.BACKENDS, "local", boom)
    with pytest.raises(UploadError, match="outcome unknown"):
        upload_file(make_sub(env), cfg=cfg())
    assert read_ledger()[-1]["status"] == "submitted"


# --- CLI ---------------------------------------------------------------------------------


def test_cli_upload_flow(env, tmp_path, capsys):
    path = make_sub(env)
    assert cli.main(["upload", str(path), "-m", "first"]) == 0
    assert "U001 scored" in capsys.readouterr().out
    assert cli.main(["upload", str(path)]) == 2
    assert cli.main(["uploads", "--reveal"]) == 0
    assert "private" in capsys.readouterr().out
    toml = tmp_path / "stock_sign.toml"
    toml.write_text(toml.read_text().replace('backend = "local"', 'backend = "browser"'))
    assert cli.main(["upload", str(make_sub(env, "b", 5)), "--final"]) == 0
    assert cli.main(["upload-confirm", "U002", "--score", "0.5"]) == 0
    assert cli.main(["uploads"]) == 0
    assert "U002" in capsys.readouterr().out
