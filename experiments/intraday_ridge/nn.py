"""1D-CNN over the 5-minute return path + MLP on the v1 features (torch, Apple GPU).

Inputs per row: 2 channels x 95 bars (stock log returns, stock-minus-country returns) and the
spec's feature table (standardised inside each fold). Target: Close return clipped at the
2% tails (MSE, like the best ridge). Same folds / metric as the spec; saved with save_run.
Never imports lightgbm (torch + LightGBM in one process segfaults on macOS).

  QRTC_CONFIG=experiments/intraday_ridge/spec.toml \
      uv run python experiments/intraday_ridge/nn.py --epochs 4 [--no-conv]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import build as B  # noqa: E402
import intraday as I  # noqa: E402

from qrtc import config, io, submit, tracking  # noqa: E402
from qrtc.cv import make_splits  # noqa: E402
from qrtc.train import CVResult, score_predictions  # noqa: E402

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def return_paths() -> tuple[np.ndarray, np.ndarray]:
    """(ids, array n x 2 x 95) of stock and country-relative 5-min log returns, scaled."""
    train, test, _ = B.load(config.SHARED_ROOT / B.RAW_DIR)
    flat = pd.concat([B.flatten(train), B.flatten(test)], ignore_index=True)
    bars = [c for c in flat.columns if c.startswith("P")]
    rets = I.log_prices(flat[bars]).diff(axis=1).iloc[:, 1:]
    rel = rets - rets.groupby([flat["DATE_ID"], flat["COUNTRY"]]).transform("mean")
    arr = np.nan_to_num(np.stack([rets.to_numpy(), rel.to_numpy()], axis=1))
    return flat["ID"].to_numpy(), (arr / arr.std()).astype(np.float32)


class Net(nn.Module):
    """arch: "cnn" (1D conv over the path), "gru" (GRU over the path, as in the Jane Street
    2024 public-LB-6th ensemble), or "mlp" (features only)."""

    def __init__(self, n_tab: int, arch: str = "cnn"):
        super().__init__()
        self.arch = arch
        self.conv = nn.Sequential(nn.Conv1d(2, 32, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(32, 32, 5, padding=2), nn.ReLU()) \
            if arch == "cnn" else None
        self.gru = nn.GRU(2, 64, num_layers=1, batch_first=True) if arch == "gru" else None
        width = {"cnn": 3 * 32, "gru": 2 * 64, "mlp": 0}[arch] + n_tab
        self.head = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(64, 1))

    def forward(self, path: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        parts = [tab]
        if self.conv is not None:
            h = self.conv(path)
            parts += [h.mean(-1), h[..., -12:].mean(-1), h[..., -1]]
        if self.gru is not None:
            seq, last = self.gru(path.transpose(1, 2))  # (n, 95, 64), (1, n, 64)
            parts += [last[0], seq[:, -12:].mean(1)]
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


def predict(net: Net, path: np.ndarray, tab: np.ndarray, batch: int = 8192) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(tab), batch):
            p = torch.from_numpy(path[i:i + batch]).to(DEVICE)
            t = torch.from_numpy(tab[i:i + batch]).to(DEVICE)
            out.append(net(p, t).cpu().numpy())
    return np.concatenate(out)


def fit_fold(path, tab, y, seed: int, epochs: int, arch: str) -> Net:
    torch.manual_seed(seed)
    net = Net(tab.shape[1], arch).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        net.train()
        for idx in np.array_split(rng.permutation(len(y)), max(1, len(y) // 2048)):
            p, t = torch.from_numpy(path[idx]).to(DEVICE), torch.from_numpy(tab[idx]).to(DEVICE)
            loss = ((net(p, t) - torch.from_numpy(y[idx]).to(DEVICE)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def standardise(train_tab: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    mu, sd = np.nanmean(train_tab, 0), np.nanstd(train_tab, 0) + 1e-12
    return [np.nan_to_num(np.clip((a - mu) / sd, -6, 6)).astype(np.float32)
            for a in (train_tab, *others)]


def run(spec, ds, epochs: int, arch: str, seeds: int = 1) -> CVResult:
    ids_all, paths = return_paths()
    row = pd.Series(np.arange(len(ids_all)), index=ids_all)
    tr_ids, te_ids = ds.train[spec.id_col].to_numpy(), ds.test[spec.id_col].to_numpy()
    p_tr, p_te = paths[row.loc[tr_ids].to_numpy()], paths[row.loc[te_ids].to_numpy()]
    feats = [c for c in io.feature_columns(ds.train, spec)
             if pd.api.types.is_numeric_dtype(ds.train[c])]
    tab_tr, tab_te = ds.train[feats].to_numpy(float), ds.test[feats].to_numpy(float)
    raw = ds.y_raw.iloc[:, 0].to_numpy(float)
    lo, hi = np.quantile(raw, [0.02, 0.98])
    y = (np.clip(raw, lo, hi) / raw.std()).astype(np.float32)
    splits = make_splits(ds.train, ds.y_vector, spec)
    oof, test = np.full(len(y), np.nan), np.zeros(len(te_ids))
    start = time.time()
    for k, (tr, va) in enumerate(splits):
        a, b, c = standardise(tab_tr[tr], tab_tr[va], tab_te)
        oof[va] = 0.0
        for s in range(seeds):  # seed ensemble, equal weights
            net = fit_fold(p_tr[tr], a, y[tr], spec.seed + k + 1000 * s, epochs, arch)
            oof[va] += predict(net, p_tr[va], b) / seeds
            test += predict(net, p_te, c) / (len(splits) * seeds)
        print(f"  fold {k} done ({time.time() - start:.0f}s)", flush=True)
    scale = float(np.nanstd(oof)) or 1.0
    oof_p, test_p = (1.0 / (1.0 + np.exp(-v / scale)) for v in (oof, test))
    weight = io.train_weight(ds, spec)
    folds = [score_predictions(spec, ds.y_vector, oof_p, va, weight) for _, va in splits]
    return CVResult(model=arch, params={"epochs": epochs, "seeds": seeds},
                    features=tuple(feats), metric=spec.metric, score=float(np.mean(folds)),
                    score_std=float(np.std(folds)), fold_scores=tuple(folds), oof=oof_p,
                    test_pred=test_p, elapsed_s=time.time() - start)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--no-conv", action="store_true", help="same as --arch mlp")
    ap.add_argument("--arch", choices=["cnn", "gru", "mlp"], default="cnn")
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    arch = "mlp" if a.no_conv else a.arch
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    result = run(spec, ds, a.epochs, arch, a.seeds)
    print(result.summary(), f"device={DEVICE}")
    tr_ids, te_ids = ds.train[spec.id_col].to_numpy(), ds.test[spec.id_col].to_numpy()
    run_id = tracking.save_run(result, spec, f"{result.model}-e{a.epochs}-s{a.seeds}", tr_ids, te_ids,
                               notes=f"torch {arch} on return paths + v1 features")
    path = submit.write_submission(submit.build_submission(spec, te_ids, result.test_pred,
                                                           ds.classes), spec, te_ids, run_id)
    print(f"run {run_id}\nsubmission {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
