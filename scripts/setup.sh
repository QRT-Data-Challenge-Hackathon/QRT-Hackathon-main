#!/usr/bin/env bash
# One-time setup per clone: dependencies, git guard hooks, Jupyter kernel.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
command -v uv >/dev/null || { echo "install uv: https://docs.astral.sh/uv/ (brew install uv)"; exit 1; }
git config core.hooksPath .githooks
chmod +x .githooks/* scripts/*.sh
fm="--extra fm"; [ -z "${QRTC_NO_FM:-}" ] || fm=""   # QRTC_NO_FM=1: skip the torch models
uv sync --all-groups $fm
uv run python -m ipykernel install --user --name qrtc --display-name "Python (qrtc)" >/dev/null
uv run python -c "import lightgbm" 2>/dev/null || echo "WARNING: lightgbm import failed (macOS: brew install libomp)"
echo "setup done: hooks on (core.hooksPath=.githooks), kernel 'Python (qrtc)' installed"
echo "next: export AGENT_NAME=<unique-name>; make rehearse; make check"
