"""Child process that runs the torch-based foundation models (``python -m qrtc.fm_worker``).

LightGBM and torch ship different OpenMP runtimes, and on macOS loading both into one process
segfaults in either order (tested on an M4, torch 2.15, LightGBM 4.x). qrtc.foundation therefore
sends every TabICL/TabPFN/TabDPT call to this worker and keeps LightGBM in the parent.

Protocol: pickled request dicts on stdin, pickled ``("ok", array)`` / ``("error", exception)``
replies on a private copy of stdout. Library output is redirected to stderr so that it can
never corrupt the reply stream. The worker exits when stdin closes.
"""

from __future__ import annotations

import importlib
import os
import pickle
import sys
from typing import Any, BinaryIO

import numpy as np


def resolve(spec: str) -> Any:
    """Import ``"package.module:Qual.name"``."""
    module, _, attr = spec.partition(":")
    obj: Any = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def run(request: dict[str, Any]) -> np.ndarray:
    """Build the backend, fit it on the context rows, return its predictions for X_query."""
    builder = resolve(request["builder"])
    model = builder(request["task"], dict(request["params"]), request["seed"],
                    list(request["cat_idx"]))
    model.fit(request["X"], request["y"])
    out = getattr(model, request["method"])(request["X_query"], **request.get("kwargs", {}))
    return np.asarray(out, dtype=float)


def _picklable(exc: BaseException) -> BaseException:
    try:
        pickle.dumps(exc)
        return exc
    except Exception:  # noqa: BLE001 - fall back to a plain error carrying the message
        return RuntimeError(f"{type(exc).__name__}: {exc}")


def serve(inp: BinaryIO, out: BinaryIO) -> None:
    """Answer requests until ``inp`` is exhausted; every failure is sent back, not raised."""
    while True:
        try:
            request = pickle.load(inp)
        except EOFError:
            return
        try:
            reply: tuple[str, Any] = ("ok", run(request))
        except Exception as exc:  # noqa: BLE001 - the caller re-raises it
            reply = ("error", _picklable(exc))
        pickle.dump(reply, out, protocol=pickle.HIGHEST_PROTOCOL)
        out.flush()


def main() -> None:
    replies = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # prints and progress bars of the libraries go to stderr
    os.nice(int(os.environ.get("QRTC_FM_NICE", "5")))  # keep the laptop responsive
    serve(sys.stdin.buffer, replies)


if __name__ == "__main__":
    main()
