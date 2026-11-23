"""Where an upload goes: the ``[upload]`` config and one function per platform ("backend").

A backend receives the spec, the upload config and the ledger record of a reserved upload, and
returns the outcome ``{"status": "scored" | "submitted" | "pending", "public": float?,
"private": float?, "detail": str}``. It raises :class:`UploadError` when the upload did not
happen.

  local    rehearsal only: score against y_test_hidden, split into public/private parts
  browser  someone (an agent driving Chrome, or a human) uploads the printed file, then
           records the outcome with ``qrtc upload-confirm``
  kaggle   ``kaggle competitions submit`` (credentials in ~/.kaggle, never in the repo)
  http     multipart POST; secrets come from environment variables named in the config
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from qrtc import config, io
from qrtc.config import Spec, SpecError
from qrtc.metrics import get_metric

BROWSER_LIMIT_BYTES = 10_000_000          # Claude in Chrome file_upload limit per call
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class UploadError(ValueError):
    """An upload was refused, or did not reach the platform."""


@dataclass(frozen=True)
class UploadConfig:
    """``[upload]`` table of competition.toml. Missing table = nobody may upload."""

    backend: str = "browser"
    uploaders: tuple[str, ...] = ()        # AGENT_NAMEs allowed to upload; "*" = anyone
    max_total: int = 0                     # platform submission limit (0 = unknown/none)
    max_per_day: int = 0
    reserve_final: int = 0                 # last slots only ``--final`` uploads may use
    url: str = ""                          # browser: submission page; http: POST endpoint
    competition: str = ""                  # kaggle competition slug
    file_field: str = "file"               # http: multipart field holding the CSV
    fields: dict[str, str] = field(default_factory=dict)        # http: extra form fields
    headers_env: dict[str, str] = field(default_factory=dict)   # http: header -> env var
    message_field: str = ""                # http: form field that receives the message
    score_key: str = ""                    # http: dotted JSON path of the score in the reply
    public_fraction: float = 0.5           # local: share of test rows on the public board
    timeout_s: float = 120.0

    def allows(self, agent: str) -> bool:
        return "*" in self.uploaders or agent in self.uploaders


def _check(cfg: UploadConfig) -> UploadConfig:
    if cfg.backend not in BACKENDS:
        raise SpecError(f"[upload] backend must be one of {sorted(BACKENDS)}, got {cfg.backend!r}")
    if not 0 < cfg.public_fraction < 1:
        raise SpecError("[upload] public_fraction must be between 0 and 1")
    if min(cfg.max_total, cfg.max_per_day, cfg.reserve_final) < 0:
        raise SpecError("[upload] limits must be >= 0")
    if cfg.max_total and cfg.reserve_final >= cfg.max_total:
        raise SpecError("[upload] reserve_final must be smaller than max_total")
    return cfg


def load_upload_config(path: str | Path | None = None) -> UploadConfig:
    """Read ``[upload]`` from the spec file in use ($QRTC_CONFIG or competition.toml)."""
    raw = dict(tomllib.loads(config.config_path(path).read_text()).get("upload", {}))
    unknown = sorted(set(raw) - {f.name for f in fields(UploadConfig)})
    if unknown:
        raise SpecError(f"[upload] has unknown keys {unknown}")
    raw["uploaders"] = tuple(raw.get("uploaders", ()))
    return _check(UploadConfig(**raw))


# --- local (rehearsal) -------------------------------------------------------------------


def public_mask(ids: pd.Series, fraction: float, seed: int) -> np.ndarray:
    """Deterministic pseudo-random public/private split of test ids (True = public)."""
    key = f"{seed:016d}"[-16:]
    hashed = pd.util.hash_pandas_object(ids.astype(str), index=False, hash_key=key)
    return (hashed.to_numpy() % 10_000) < fraction * 10_000


def _hidden_path(spec: Spec) -> Path:
    hidden = spec.path(spec.hidden_y_test)
    if not hidden or not hidden.exists():
        raise UploadError("no hidden_y_test configured (only rehearsal datasets have one)")
    return hidden


def hidden_score(spec: Spec, sub: pd.DataFrame, ids: np.ndarray | None = None) -> float:
    """Score a submission against the rehearsal's hidden labels (optionally only ``ids``)."""
    truth = pd.read_csv(_hidden_path(spec))
    if ids is not None:
        truth = truth[truth[spec.id_col].isin(ids)]
    sub = sub.rename(columns={c: f"{c}_pred" for c in spec.out_cols})
    merged = truth.merge(sub, on=spec.id_col, validate="one_to_one")
    pred_cols = [f"{c}_pred" for c in spec.out_cols]
    truth_y = merged[list(spec.target_cols)]
    if spec.binarize_target:
        truth_y = (truth_y > 0).astype(int)
    classes = io._class_labels(truth_y, spec)
    y_true = io.encode_target(truth_y, classes)
    pred = merged[pred_cols].to_numpy()
    pred = pred[:, 0] if pred.shape[1] == 1 else pred
    if spec.task in ("binary", "multiclass") and pred.ndim == 1 and pred.dtype == object:
        pred = io.encode_target(merged[pred_cols], classes)
    groups = None
    if spec.date_col:
        test = io.read_features(spec, spec.x_test)[[spec.id_col, spec.date_col]]
        groups = merged[[spec.id_col]].merge(test, on=spec.id_col)[spec.date_col].to_numpy()
    metric = get_metric(spec.metric)
    return float(metric(y_true, pred, groups=groups, **dict(spec.metric_params)))


def local_backend(spec: Spec, cfg: UploadConfig, record: dict) -> dict[str, Any]:
    ids = pd.read_csv(_hidden_path(spec))[spec.id_col]
    public = public_mask(ids, cfg.public_fraction, spec.seed)
    sub = pd.read_csv(record["path"])
    pub = hidden_score(spec, sub, ids[public].to_numpy())
    priv = hidden_score(spec, sub, ids[~public].to_numpy())
    return {"status": "scored", "public": pub, "private": priv,
            "detail": f"public {spec.metric} {pub:.5f} (private part hidden until "
                      "`qrtc uploads --reveal`, like the real final ranking)"}


# --- browser (two-phase: reserve here, upload in the browser, then upload-confirm) --------


def browser_backend(spec: Spec, cfg: UploadConfig, record: dict) -> dict[str, Any]:
    size = Path(record["path"]).stat().st_size
    warn = (f" WARNING: {size / 1e6:.1f} MB is over the 10 MB browser-upload limit; zip it if "
            "the platform accepts zips.") if size > BROWSER_LIMIT_BYTES else ""
    uid = record["upload_id"]
    detail = (f"reserved {uid}. Upload exactly {record['path']} at {cfg.url or 'the platform'}"
              f", then run `uv run qrtc upload-confirm {uid} --score <public score>` (no score "
              f"shown yet: `upload-confirm {uid}`; rejected: `--failed '<reason>'`).{warn}")
    return {"status": "pending", "detail": detail}


# --- kaggle ------------------------------------------------------------------------------


def kaggle_backend(spec: Spec, cfg: UploadConfig, record: dict) -> dict[str, Any]:
    exe = shutil.which("kaggle")
    if not exe:
        raise UploadError("kaggle CLI not found: `uv tool install kaggle`; credentials go in "
                          "~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY, never the repo")
    if not cfg.competition:
        raise UploadError("[upload] competition = '<kaggle slug>' is missing")
    message = " ".join(filter(None, [record["upload_id"], record.get("run_id"),
                                     record.get("message")]))
    cmd = [exe, "competitions", "submit", "-c", cfg.competition, "-f", record["path"],
           "-m", message]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.timeout_s,
                             check=False)
    except subprocess.TimeoutExpired as exc:
        raise UploadError(f"kaggle submit timed out after {cfg.timeout_s}s") from exc
    if out.returncode != 0:
        raise UploadError(f"kaggle submit failed: {(out.stderr or out.stdout).strip()[-500:]}")
    return {"status": "submitted",
            "detail": f"{out.stdout.strip()[-300:]} | score: `kaggle competitions submissions "
                      f"-c {cfg.competition}`, then `qrtc upload-confirm "
                      f"{record['upload_id']} --score <public>`"}


# --- http --------------------------------------------------------------------------------


def _multipart(form: dict[str, str], file_field: str, path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
             .encode() for k, v in form.items()]
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
            f'filename="{path.name}"\r\nContent-Type: text/csv\r\n\r\n').encode()
    parts.append(head + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _secret_headers(cfg: UploadConfig) -> dict[str, str]:
    headers = {}
    for header, env in cfg.headers_env.items():
        value = os.environ.get(env)
        if not value:
            raise UploadError(f"environment variable {env} (for header {header}) is not set")
        headers[header] = value
    return headers


def _score_from(text: str, key: str) -> float | None:
    if not key:
        return None
    try:
        node: Any = json.loads(text)
        for part in key.split("."):
            node = node[part]
        return float(node)
    except (ValueError, KeyError, TypeError, IndexError):
        return None


def http_backend(spec: Spec, cfg: UploadConfig, record: dict) -> dict[str, Any]:
    url = urlparse(cfg.url)
    if url.scheme != "https" and not (url.scheme == "http" and url.hostname in LOCAL_HOSTS):
        raise UploadError("[upload] url must be https:// (plain http only for localhost)")
    form = dict(cfg.fields)
    if cfg.message_field:
        form[cfg.message_field] = record.get("message", "")
    body, content_type = _multipart(form, cfg.file_field, Path(record["path"]))
    request = urllib.request.Request(cfg.url, data=body, method="POST",
                                     headers={**_secret_headers(cfg),
                                              "Content-Type": content_type})
    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout_s) as response:
            text = response.read(20_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        reply = exc.read(300).decode("utf-8", "replace")
        raise UploadError(f"HTTP {exc.code} from the platform: {reply}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UploadError(f"upload request failed: {exc}") from exc
    public = _score_from(text, cfg.score_key)
    return {"status": "scored" if public is not None else "submitted", "public": public,
            "detail": text[:500]}


Backend = Callable[[Spec, UploadConfig, dict], dict[str, Any]]
BACKENDS: dict[str, Backend] = {"local": local_backend, "browser": browser_backend,
                                "kaggle": kaggle_backend, "http": http_backend}
