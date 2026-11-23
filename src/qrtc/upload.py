"""Upload a submission to the competition platform: validated, budgeted, logged.

Any agent named in ``[upload].uploaders`` (competition.toml; ONE name at the event) may upload.
:func:`upload_file`
  1. refuses callers who are not designated uploaders,
  2. validates the file exactly like ``qrtc check-sub``,
  3. under a lock shared by every worktree on this machine, refuses a file that was already
     uploaded and checks the budget (``max_total``, ``max_per_day``, and ``reserve_final``
     slots that only ``final=True`` uploads may use),
  4. writes a ``pending`` ledger entry, calls the backend (:mod:`qrtc.platforms`), writes the
     outcome, and stores the public score against the run (``tracking.record_lb``).
Browser uploads stay ``pending`` until :func:`confirm` records their score or failure.

Ledger: ``<shared>/submissions/ledger.jsonl``, append-only; the last line per upload wins.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from qrtc import config, io, submit, tracking
from qrtc.config import Spec
from qrtc.platforms import BACKENDS, UploadConfig, UploadError, load_upload_config

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows: no cross-process lock
    fcntl = None

COUNTED = frozenset({"pending", "submitted", "scored"})   # these use up a platform slot
SHOWN = ["upload_id", "created", "agent", "status", "public", "final", "run_id", "backend",
         "message"]


def ledger_path() -> Path:
    return config.SUBMISSIONS_DIR / "ledger.jsonl"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _append(entry: dict[str, Any]) -> None:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_ledger() -> list[dict[str, Any]]:
    """Current state of every upload (ledger lines folded by upload_id), oldest first."""
    path = ledger_path()
    if not path.exists():
        return []
    state: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            state[entry["upload_id"]] = {**state.get(entry["upload_id"], {}), **entry}
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise UploadError(f"{path}:{number} is corrupt ({exc}); repair it by hand") from exc
    return list(state.values())


@contextmanager
def _locked() -> Iterator[None]:
    """Serialise uploads across all worktrees and agents on this machine."""
    path = config.SUBMISSIONS_DIR / ".upload.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        if fcntl:
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl:
                fcntl.flock(fh, fcntl.LOCK_UN)


@dataclass(frozen=True)
class Budget:
    used: int
    used_today: int
    max_total: int
    max_per_day: int
    reserve_final: int

    @property
    def left(self) -> int | None:
        """Uploads left on the tighter limit, None when no limit is configured."""
        limits = [cap - used for cap, used in ((self.max_total, self.used),
                                               (self.max_per_day, self.used_today)) if cap]
        return min(limits) if limits else None

    def refusal(self, final: bool) -> str | None:
        left = self.left
        if left is None:
            return None
        if left <= 0:
            return (f"no submissions left ({self.used} used, limits total={self.max_total} "
                    f"per_day={self.max_per_day})")
        if not final and left <= self.reserve_final:
            return (f"only {left} submission(s) left and {self.reserve_final} are kept for "
                    "finals: upload with --final only for a final pick the Captain chose")
        return None


def budget(records: list[dict], cfg: UploadConfig, competition: str,
           today: str | None = None) -> Budget:
    """Slots used by live uploads (pending, submitted, scored) of ``competition``."""
    live = [r for r in records if r.get("competition") == competition
            and r.get("status") in COUNTED]
    day = today or date.today().isoformat()
    return Budget(len(live), sum(r.get("day") == day for r in live), cfg.max_total,
                  cfg.max_per_day, cfg.reserve_final)


def _require_uploader(cfg: UploadConfig) -> str:
    agent = tracking.agent_name()
    if not cfg.allows(agent):
        raise UploadError(f"{agent!r} is not a designated uploader ([upload].uploaders = "
                          f"{list(cfg.uploaders)}); hand the file path to the uploader")
    return agent


def _validate(spec: Spec, path: Path) -> None:
    test_ids = io.read_features(spec, spec.x_test)[spec.id_col].to_numpy()
    problems = submit.validate_submission(pd.read_csv(path), spec, test_ids,
                                          submit.load_sample(spec))
    if problems:
        raise UploadError(f"{path.name} is invalid: " + "; ".join(problems))


def infer_run_id(path: Path) -> str | None:
    """Run id behind ``submissions/<run_id>[_HHMMSS].csv`` if that run exists locally."""
    for candidate in (path.stem, re.sub(r"_\d{6}$", "", path.stem)):
        if (config.RUNS_DIR / candidate / "meta.json").exists():
            return candidate
    return None


def _reserve(spec: Spec, cfg: UploadConfig, records: list[dict], path: Path, digest: str,
             agent: str, **info: Any) -> dict[str, Any]:
    if not info.pop("allow_duplicate") and any(
            r.get("sha256") == digest and r.get("competition") == spec.name
            and r.get("status") in COUNTED for r in records):
        raise UploadError(f"{path.name} was already uploaded (same content); "
                          "pass allow_duplicate / --allow-duplicate to resend it")
    refusal = budget(records, cfg, spec.name).refusal(info["final"])
    if refusal:
        raise UploadError(refusal)
    return {"upload_id": f"U{len(records) + 1:03d}", "created": _now(),
            "day": date.today().isoformat(), "agent": agent, "competition": spec.name,
            "backend": cfg.backend, "path": str(path), "sha256": digest,
            "run_id": info["run_id"] or infer_run_id(path), "message": info["message"],
            "final": info["final"], "status": "pending"}


def _run_backend(spec: Spec, cfg: UploadConfig, record: dict) -> dict[str, Any]:
    try:
        outcome = BACKENDS[cfg.backend](spec, cfg, record)
    except UploadError as exc:  # did not reach the platform: frees the slot
        _append({"upload_id": record["upload_id"], "updated": _now(), "status": "failed",
                 "detail": str(exc)})
        raise
    except Exception as exc:  # unknown whether the platform got it: keep the slot counted
        _append({"upload_id": record["upload_id"], "updated": _now(), "status": "submitted",
                 "detail": f"outcome unknown: {exc!r}"})
        raise UploadError(f"outcome unknown ({exc!r}); check the platform, then run "
                          f"`qrtc upload-confirm {record['upload_id']}` with "
                          "--score or --failed") from exc
    entry = {"upload_id": record["upload_id"], "updated": _now(), **outcome}
    _append(entry)
    return entry


def _store_score(record: dict[str, Any]) -> None:
    if record.get("public") is not None and record.get("run_id"):
        tracking.record_lb(record["run_id"], float(record["public"]),
                           note=f"{record['upload_id']} {record.get('message', '')}".strip())


def upload_file(path: str | Path, message: str = "", final: bool = False,
                run_id: str | None = None, dry_run: bool = False,
                allow_duplicate: bool = False, spec: Spec | None = None,
                cfg: UploadConfig | None = None) -> dict[str, Any]:
    """Validate, budget-check, log and upload ``path``; returns the upload's ledger state."""
    spec = spec or config.load_spec()
    cfg = cfg or load_upload_config()
    agent = _require_uploader(cfg)
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        raise UploadError(f"{file} not found")
    _validate(spec, file)
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    with _locked():
        record = _reserve(spec, cfg, read_ledger(), file, digest, agent, message=message,
                          final=final, run_id=run_id, allow_duplicate=allow_duplicate)
        if dry_run:
            return {**record, "status": "dry-run"}
        _append(record)
        record = {**record, **_run_backend(spec, cfg, record)}
    _store_score(record)
    return record


def confirm(upload_id: str, score: float | None = None, failed: str | None = None,
            cfg: UploadConfig | None = None) -> dict[str, Any]:
    """Record what happened to a pending/submitted upload: its public ``score``, that it went
    through without a score yet (neither argument), or that it ``failed`` (frees the slot)."""
    if score is not None and failed is not None:
        raise UploadError("give a score or a failure reason, not both")
    agent = _require_uploader(cfg or load_upload_config())
    with _locked():
        current = {r["upload_id"]: r for r in read_ledger()}.get(upload_id)
        if current is None:
            raise UploadError(f"unknown upload {upload_id}; see `qrtc uploads`")
        if current["status"] not in ("pending", "submitted"):
            raise UploadError(f"{upload_id} is already {current['status']}")
        entry: dict[str, Any] = {"upload_id": upload_id, "updated": _now(), "by": agent}
        if failed is not None:
            entry |= {"status": "failed", "detail": failed}
        elif score is not None:
            entry |= {"status": "scored", "public": float(score)}
        else:
            entry |= {"status": "submitted"}
        _append(entry)
    record = {**current, **entry}
    _store_score(record)
    return record


def upload_table(competition: str | None = None, reveal: bool = False) -> pd.DataFrame:
    """Uploads (optionally of one competition); ``reveal`` adds the local private score."""
    frame = pd.DataFrame(read_ledger())
    if frame.empty:
        return frame
    if competition:
        frame = frame[frame["competition"] == competition]
    cols = [c for c in SHOWN + (["private"] if reveal else []) if c in frame.columns]
    return frame[cols].reset_index(drop=True)
