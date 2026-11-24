"""Supervised autoencoder + MLP (Jane Street 1st place, Yirun Zhang) on the v1 features.

Architecture and dropout rates copied from the public notebook (gogo827jz/jane-street-
supervised-autoencoder-mlp); losses = decoder MSE + |y|-weighted BCE (aux head) + |y|-weighted
BCE (main head). Adaptations: one target (sign of Close return), torch on the Apple GPU, and
early stopping on held-out *training-fold* days (the notebook stopped on the validation fold,
which inflates CV). The autoencoder is trained inside each fold, never on all rows.

  QRTC_CONFIG=experiments/dmr-claude-2/spec_official_v1.toml \
      uv run python experiments/dmr-claude-2/sae.py [--seeds 1]
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
from nn import DEVICE, standardise  # noqa: E402

from qrtc import config, io, submit, tracking  # noqa: E402
from qrtc.cv import make_splits  # noqa: E402
from qrtc.train import CVResult, score_predictions  # noqa: E402

HIDDEN = (96, 96, 896, 448, 448, 256)
DROPOUT = (0.035, 0.038, 0.424, 0.104, 0.492, 0.320, 0.272, 0.438)


class SAEMLP(nn.Module):
    def __init__(self, n: int, h=HIDDEN, d=DROPOUT):
        super().__init__()
        self.bn0, self.noise = nn.BatchNorm1d(n), d[0]
        self.enc = nn.Sequential(nn.Linear(n, h[0]), nn.BatchNorm1d(h[0]), nn.SiLU())
        self.dec = nn.Sequential(nn.Dropout(d[1]), nn.Linear(h[0], n))
        self.aux = nn.Sequential(nn.Linear(n, h[1]), nn.BatchNorm1d(h[1]), nn.SiLU(),
                                 nn.Dropout(d[2]), nn.Linear(h[1], 1))
        layers, prev = [nn.BatchNorm1d(n + h[0]), nn.Dropout(d[3])], n + h[0]
        for i in range(2, len(h)):
            layers += [nn.Linear(prev, h[i]), nn.BatchNorm1d(h[i]), nn.SiLU(), nn.Dropout(d[i + 2])]
            prev = h[i]
        self.mlp = nn.Sequential(*layers, nn.Linear(prev, 1))

    def forward(self, x: torch.Tensor):
        x0 = self.bn0(x)
        e = self.enc(x0 + torch.randn_like(x0) * self.noise if self.training else x0)
        dec = self.dec(e)
        return dec, self.aux(dec).squeeze(1), self.mlp(torch.cat([x0, e], 1)).squeeze(1)


REGRESS = False  # --regress: heads fit the clipped real target with MSE (like the best ridge)


def losses(net, x, y, w):
    dec, aux, out = net(x)
    if REGRESS:
        return ((dec - x) ** 2).mean() + ((aux - y) ** 2).mean() + ((out - y) ** 2).mean(), out
    bce = nn.functional.binary_cross_entropy_with_logits
    return (((dec - x) ** 2).mean() + (bce(aux, y, reduction="none") * w).mean()
            + (bce(out, y, reduction="none") * w).mean()), out


def logits(net, x: np.ndarray, batch: int = 16384) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(torch.from_numpy(x[i:i + batch]).to(DEVICE))[2].cpu().numpy()
                               for i in range(0, len(x), batch)])


def fit(x, y, w, x_es, y_es, w_es, seed: int, max_epochs: int = 40, patience: int = 4):
    torch.manual_seed(seed)
    net = SAEMLP(x.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng, best, best_state, bad = np.random.default_rng(seed), np.inf, None, 0
    t = [torch.from_numpy(a).to(DEVICE) for a in (x, y, w, x_es, y_es, w_es)]
    for _ in range(max_epochs):
        net.train()
        for idx in np.array_split(rng.permutation(len(y)), max(1, len(y) // 4096)):
            loss, _ = losses(net, t[0][idx], t[1][idx], t[2][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(losses(net, t[3], t[4], t[5])[0])
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        elif (bad := bad + 1) >= patience:
            break
    net.load_state_dict(best_state)
    return net


def run(spec, ds, seeds: int) -> CVResult:
    feats = [c for c in io.feature_columns(ds.train, spec)
             if pd.api.types.is_numeric_dtype(ds.train[c])]
    X, X_te = ds.train[feats].to_numpy(float), ds.test[feats].to_numpy(float)
    y = ds.y_vector.astype(np.float32)
    if REGRESS:
        raw = ds.y_raw.iloc[:, 0].to_numpy(float)
        lo, hi = np.quantile(raw, [0.02, 0.98])
        y = (np.clip(raw, lo, hi) / raw.std()).astype(np.float32)
    w = io.train_weight(ds, spec)
    w = (w / w.mean()).astype(np.float32)
    days = ds.train[spec.date_col].to_numpy()
    splits = make_splits(ds.train, ds.y_vector, spec)
    oof, test, start = np.full(len(y), np.nan), np.zeros(len(X_te)), time.time()
    for k, (tr, va) in enumerate(splits):
        rng = np.random.default_rng(spec.seed + k)
        es_days = rng.choice(np.unique(days[tr]), size=len(np.unique(days[tr])) // 7,
                             replace=False)
        es = np.isin(days[tr], es_days)
        fit_rows, es_rows = tr[~es], tr[es]
        a, b, c, e = standardise(X[fit_rows], X[va], X_te, X[es_rows])
        fold_oof, fold_te = np.zeros(len(va)), np.zeros(len(X_te))
        for s in range(seeds):
            net = fit(a, y[fit_rows], w[fit_rows], e, y[es_rows], w[es_rows],
                      seed=spec.seed + 100 * k + s)
            fold_oof += logits(net, b) / seeds
            fold_te += logits(net, c) / seeds
        oof[va], test = fold_oof, test + fold_te / len(splits)
        print(f"  fold {k} done ({time.time() - start:.0f}s)", flush=True)
    scale = float(np.nanstd(oof)) if REGRESS else 1.0
    oof_p, test_p = (1.0 / (1.0 + np.exp(-v / scale)) for v in (oof, test))
    weight = io.train_weight(ds, spec)
    folds = [score_predictions(spec, ds.y_vector, oof_p, va, weight) for _, va in splits]
    return CVResult(model="sae_mlp", params={"seeds": seeds, "hidden": HIDDEN},
                    features=tuple(feats), metric=spec.metric, score=float(np.mean(folds)),
                    score_std=float(np.std(folds)), fold_scores=tuple(folds), oof=oof_p,
                    test_pred=test_p, elapsed_s=time.time() - start)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--regress", action="store_true")
    a = ap.parse_args()
    global REGRESS
    REGRESS = a.regress
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    result = run(spec, ds, a.seeds)
    print(result.summary(), f"device={DEVICE}")
    tr_ids, te_ids = ds.train[spec.id_col].to_numpy(), ds.test[spec.id_col].to_numpy()
    run_id = tracking.save_run(result, spec, f"sae-mlp{'-reg' if a.regress else ''}-s{a.seeds}", tr_ids, te_ids,
                               notes="Jane Street supervised AE + MLP, in-fold, |y|-weighted")
    path = submit.write_submission(submit.build_submission(spec, te_ids, result.test_pred,
                                                           ds.classes), spec, te_ids, run_id)
    print(f"run {run_id}\nsubmission {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
