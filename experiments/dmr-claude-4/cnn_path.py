"""1D CNN on the relative intraday price path (issue #29): a new model family for the blend.

Per row, 3 channels x 95 five-minute log returns, right-aligned (last bar = cutoff):
  rel   stock return minus the equal-weight DATE_ID x COUNTRY path
  ctry  the country path
  cum   cumulative relative return
each divided by that day x country's dispersion (median per-stock std of ``rel``), so the
input is scale-free (the test period is calmer). Trained with |y|-weighted BCE on the sign
(the metric's own surrogate), early stopping on an inner day split. Official date_blocks
folds via make_splits; saved with save_run. Runs without LightGBM in the process.

Usage: uv run python experiments/dmr-claude-4/cnn_path.py [--seeds 1] [--device mps]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from qrtc import config, cv, io, metrics, submit, tracking
from qrtc.train import CVResult

RAW = config.SHARED_ROOT / "data" / "raw" / "qrt-x-eth-hackathon-24-11-2026"
WIDTH = 96
EPS = 1e-6


def build_paths(ids_order: np.ndarray) -> np.ndarray:
    """(n, 3, WIDTH-1) float32 inputs, rows in ``ids_order``."""
    raw = pd.concat([pd.read_parquet(RAW / f) for f in ("x_train.parquet", "x_test.parquet")])
    raw = raw.set_index("ID").loc[ids_order]
    lp = np.full((len(raw), WIDTH), np.nan)
    for i, arr in enumerate(raw["Price TS"].to_numpy()):
        a = np.log(np.asarray(arr, dtype=float)[-WIDTH:])
        lp[i, WIDTH - len(a):] = a
    lp = pd.DataFrame(lp).ffill(axis=1).bfill(axis=1).to_numpy()
    r = np.nan_to_num(np.diff(lp, axis=1))
    g = raw.groupby(["DATE_ID", "COUNTRY"], sort=False).ngroup().to_numpy()
    ng = g.max() + 1
    cnt = np.bincount(g, minlength=ng)[:, None]
    ctry = np.zeros((ng, r.shape[1]))
    np.add.at(ctry, g, r)
    ctry /= cnt
    rel = r - ctry[g]
    row_sd = rel.std(axis=1)
    scale = pd.Series(row_sd).groupby(g).transform("median").to_numpy()[:, None] + EPS
    x = np.stack([rel / scale, ctry[g] / scale, np.cumsum(rel, axis=1) / scale], axis=1)
    x /= x.reshape(len(x), 3, -1).std(axis=(0, 2), keepdims=True)
    return np.clip(x, -10, 10).astype(np.float32)


class Net(nn.Module):
    def __init__(self, c: int = 48, tail: int = 12) -> None:
        super().__init__()
        self.tail = tail
        self.conv = nn.Sequential(
            nn.Conv1d(3, c, 5, padding="same"), nn.GELU(),
            nn.Conv1d(c, c, 5, padding="same", dilation=2), nn.GELU(),
            nn.Conv1d(c, c, 5, padding="same", dilation=4), nn.GELU())
        self.head = nn.Sequential(nn.Linear(3 * c + 3 * tail, 64), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        z = torch.cat([h.mean(-1), h.amax(-1), h[:, :, -1], x[:, :, -self.tail:].flatten(1)], 1)
        return self.head(z).squeeze(-1)


def predict(net: Net, x: np.ndarray, dev: str, bs: int = 8192) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(net(torch.from_numpy(x[i:i + bs]).to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def wacc(yb: np.ndarray, logit: np.ndarray, w: np.ndarray) -> float:
    return float(((logit > 0) == (yb > 0.5)) @ w / w.sum())


def fit_one(x, yb, w, days, seed, dev, epochs=14, patience=3, bs=1024) -> Net:
    """Train on rows (x, yb, w) with an inner 15% day split for early stopping."""
    rng = np.random.default_rng(seed)
    ud = np.unique(days)
    inner = np.isin(days, rng.choice(ud, int(0.15 * len(ud)), replace=False))
    tr, va = np.flatnonzero(~inner), np.flatnonzero(inner)
    torch.manual_seed(seed)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    xt, yt = torch.from_numpy(x[tr]), torch.from_numpy(yb[tr].astype(np.float32))
    wt = torch.from_numpy((w[tr] / w[tr].mean()).astype(np.float32))
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.from_numpy(rng.permutation(len(tr)))
        for i in range(0, len(tr), bs):
            b = perm[i:i + bs]
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(xt[b].to(dev)), yt[b].to(dev), weight=wt[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = wacc(yb[va], predict(net, x[va], dev), w[va])
        print(f"    epoch {ep}: inner wacc {score:.5f}", flush=True)
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--slug", default="cnn-path")
    a = p.parse_args()
    t0 = time.time()
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    tr_ids, te_ids = ds.train[spec.id_col].to_numpy(), ds.test[spec.id_col].to_numpy()
    x_all = build_paths(np.concatenate([tr_ids, te_ids]))
    x_tr, x_te = x_all[:len(tr_ids)], x_all[len(tr_ids):]
    r = ds.y_raw.iloc[:, 0].to_numpy(dtype=float)
    yb, w = (r > 0).astype(np.float32), np.abs(r)
    days = ds.train[spec.date_col].to_numpy()
    print(f"paths built {x_all.shape} in {time.time() - t0:.0f}s on {a.device}", flush=True)
    splits = cv.make_splits(ds.train, ds.y_vector, spec)
    oof, test_logit, folds = np.zeros(len(r)), np.zeros(len(te_ids)), []
    wa = metrics.get_metric(spec.metric)
    for k, (tri, vai) in enumerate(splits):
        lv = np.zeros(len(vai))
        for s in range(a.seeds):
            net = fit_one(x_tr[tri], yb[tri], w[tri], days[tri], spec.seed + 100 * k + s, a.device)
            lv += predict(net, x_tr[vai], a.device) / a.seeds
            test_logit += predict(net, x_te, a.device) / (a.seeds * len(splits))
        oof[vai] = lv
        folds.append(wa(yb[vai].astype(int), (lv > 0).astype(int), weight=w[vai]))
        print(f"  fold {k}: {spec.metric} = {folds[-1]:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
    res = CVResult(model="cnn1d", params={"seeds": a.seeds, "c": 48, "loss": "bce|y|"},
                   features=("path_rel_ctry_cum_3x95",), metric=spec.metric,
                   score=float(np.mean(folds)), score_std=float(np.std(folds)),
                   fold_scores=tuple(folds), oof=sig(oof), test_pred=sig(test_logit),
                   elapsed_s=time.time() - t0)
    run_id = tracking.save_run(res, spec, a.slug, tr_ids, te_ids, notes="issue #29")
    sub = submit.build_submission(spec, te_ids, res.test_pred, ds.classes)
    path = submit.write_submission(sub, spec, te_ids, run_id)
    print(f"cnn1d: {spec.metric} = {res.score:.5f} +- {res.score_std:.5f} {np.round(folds, 5)}")
    print(f"run {run_id}\nsubmission {path}")


if __name__ == "__main__":
    main()
