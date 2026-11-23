#!/usr/bin/env python3
"""Claude Code PreToolUse guard for browser file uploads (`mcp__*__file_upload`).

A browser upload is allowed only for files that `uv run qrtc upload <file>` reserved: each
file's sha256 must match an upload whose current state in
`<shared>/submissions/ledger.jsonl` is `pending`. That keeps every upload inside the team's
submission budget and ledger (AGENTS.md §9.1). Fails closed: anything unreadable blocks.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HOW = "run `uv run qrtc upload <file>` first (designated uploader only, AGENTS.md §9.1)"


def shared_root(cwd: str) -> Path:
    """Main checkout shared by all worktrees ($QRTC_SHARED_DIR overrides), like qrtc.config."""
    env = os.environ.get("QRTC_SHARED_DIR")
    if env:
        return Path(env).expanduser().resolve()
    try:
        out = subprocess.run(["git", "-C", cwd, "rev-parse", "--path-format=absolute",
                              "--git-common-dir"], capture_output=True, text=True, timeout=5,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return Path(cwd)
    common = Path(out.stdout.strip())
    return common.parent if out.returncode == 0 and common.name == ".git" else Path(cwd)


def pending_hashes(ledger: Path) -> set[str]:
    state: dict[str, dict] = {}
    for line in ledger.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            state[entry["upload_id"]] = {**state.get(entry["upload_id"], {}), **entry}
    return {r.get("sha256", "") for r in state.values() if r.get("status") == "pending"}


def evaluate(tool_input: dict, root: Path) -> str | None:
    """Return a block reason for uploading ``tool_input['paths']``, or None to allow."""
    paths = tool_input.get("paths") or ([tool_input["path"]] if tool_input.get("path") else [])
    if not paths:
        return None
    ledger = root / "submissions" / "ledger.jsonl"
    if not ledger.exists():
        return f"no submissions ledger at {ledger}: {HOW}"
    try:
        pending = pending_hashes(ledger)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"cannot read {ledger} ({exc}); repair it before uploading"
    for raw in paths:
        path = Path(str(raw)).expanduser()
        if not path.is_file():
            return f"{raw} is not a file"
        if hashlib.sha256(path.read_bytes()).hexdigest() not in pending:
            return (f"{raw} has no pending reservation in {ledger}: {HOW}, then upload exactly "
                    "the file it prints")
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"guard_upload: unreadable hook input ({exc}); blocking", file=sys.stderr)
        return 2
    if not str(payload.get("tool_name", "")).endswith("file_upload"):
        return 0
    cwd = payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    reason = evaluate(payload.get("tool_input") or {}, shared_root(cwd))
    if reason:
        print(f"Blocked by upload guard: {reason}.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
