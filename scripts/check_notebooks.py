"""Verify (default) or strip (--fix) outputs and execution counts in notebooks.

Committed notebooks must have no outputs: outputs bloat diffs, leak data, and make parallel
agents' merges conflict. Usage: python scripts/check_notebooks.py [--fix] [paths...]
(default paths: every tracked-looking *.ipynb outside .venv/.worktrees/notebooks/scratch).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SKIP = (".venv", ".worktrees", ".ipynb_checkpoints", "notebooks/scratch")


def dirty(nb: dict) -> bool:
    return any(c.get("outputs") or c.get("execution_count") for c in nb.get("cells", [])
               if c.get("cell_type") == "code")


def strip(nb: dict) -> dict:
    cells = [{**c, "outputs": [], "execution_count": None} if c.get("cell_type") == "code" else c
             for c in nb.get("cells", [])]
    return {**nb, "cells": cells}


def main(argv: list[str]) -> int:
    fix = "--fix" in argv
    args = [a for a in argv if a != "--fix"]
    root = Path(__file__).resolve().parents[1]
    paths = [Path(a) for a in args] or [p for p in root.rglob("*.ipynb")
                                        if not any(s in p.as_posix() for s in SKIP)]
    bad = []
    for path in (p for p in paths if p.suffix == ".ipynb" and p.exists()):
        nb = json.loads(path.read_text())
        if dirty(nb):
            if fix:
                path.write_text(json.dumps(strip(nb), indent=1, ensure_ascii=False) + "\n")
                print(f"stripped {path}")
            else:
                bad.append(str(path))
    if bad:
        print("notebooks with outputs (run: uv run python scripts/check_notebooks.py --fix):",
              *bad, sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
