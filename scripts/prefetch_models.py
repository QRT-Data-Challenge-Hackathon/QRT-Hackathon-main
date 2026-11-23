"""Download the foundation-model weights and prove they work offline (``make models``).

    uv run python scripts/prefetch_models.py            # TabICLv2, TabPFN v2, TabDPT
    uv run python scripts/prefetch_models.py --tabpfn3  # also TabPFN-3.5 (licence, see below)

Each model is fit on a tiny table (this downloads its weights into ~/.cache), then the same
checks run again in a fresh process with HF_HUB_OFFLINE=1 through the worker process, which is
how the toolkit runs them at the event.

TabPFN-3.5 is non-commercial and needs a one-time licence acceptance by a human: run
``--tabpfn3`` in your own terminal (not through an agent), log in at ux.priorlabs.ai when the
browser opens, accept the licence. Check the event rules before using it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from qrtc import foundation

TASKS = ("binary", "multiclass", "regression")
OPEN = [(name, {}) for name in ("tabicl", "tabpfn", "tabdpt")]
TABPFN3 = [("tabpfn", {"version": "v3_5"}), ("tabpfn", {"version": "v3_5_fast"})]


def run_checks(models: list[tuple[str, dict]]) -> list[str]:
    failures = []
    for name, params in models:
        label = f"{name}{params.get('version', '') and ' ' + params['version']}"
        if not foundation.available(name):
            failures.append(f"{label}: not installed (uv sync --all-groups --extra fm)")
            continue
        for task in TASKS:
            start = time.time()
            try:
                score = foundation.smoke_test(name, task, params)
            except Exception as exc:  # noqa: BLE001 - report every model, then fail
                failures.append(f"{label} {task}: {type(exc).__name__}: {exc}")
                print(f"FAIL {label:18s} {task:10s} {type(exc).__name__}", flush=True)
                continue
            print(f"ok   {label:18s} {task:10s} score={score:.2f} {time.time() - start:5.1f}s",
                  flush=True)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tabpfn3", action="store_true", help="also TabPFN-3.5 (licence)")
    parser.add_argument("--offline", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    models = OPEN + (TABPFN3 if args.tabpfn3 else [])
    if not args.offline:
        # download in this process so that TabPFN's licence prompt can use the terminal
        os.environ["QRTC_FM_INPROCESS"] = "1"
        print("downloading weights (first run only) ...", flush=True)
    else:
        print("offline check through the worker process ...", flush=True)
    failures = run_checks(models)
    if failures:
        print("\n".join(["", "FAILED:", *failures]), file=sys.stderr)
        return 1
    if args.offline:
        print("all models work offline")
        return 0
    env = {k: v for k, v in os.environ.items() if k != "QRTC_FM_INPROCESS"}
    cmd = [sys.executable, __file__, "--offline", *(["--tabpfn3"] if args.tabpfn3 else [])]
    return subprocess.call(cmd, env={**env, "HF_HUB_OFFLINE": "1"})


if __name__ == "__main__":
    sys.exit(main())
