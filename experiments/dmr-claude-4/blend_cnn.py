"""Equal-weight z-score blend of saved runs (issue #29); saves the blend with save_run.

Each run's OOF/test score s in (0,1) becomes z = (s - 0.5) / std(oof - 0.5); the blend is the
mean z, mapped back through a sigmoid so the sign is kept (threshold 0.5).
Usage: uv run python experiments/dmr-claude-4/blend_cnn.py <slug> RUN1 RUN2 [...]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from qrtc import config, cv, io, metrics, submit, tracking
from qrtc.train import CVResult


def load(run_id: str, tr_ids: np.ndarray, te_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(config.SHARED_ROOT / "runs" / run_id / "preds.npz", allow_pickle=True)
    o = dict(zip(z["train_ids"], z["oof"], strict=True))
    t = dict(zip(z["test_ids"], z["test"], strict=True))
    oof, test = np.array([o[i] for i in tr_ids]) - 0.5, np.array([t[i] for i in te_ids]) - 0.5
    sd = oof.std() or 1.0
    return oof / sd, test / sd


def main(slug: str, runs: list[str]) -> None:
    t0 = time.time()
    spec = config.load_spec()
    ds = io.load_dataset(spec)
    tr_ids, te_ids = ds.train[spec.id_col].to_numpy(), ds.test[spec.id_col].to_numpy()
    parts = [load(r, tr_ids, te_ids) for r in runs]
    oof = np.mean([p[0] for p in parts], axis=0)
    test = np.mean([p[1] for p in parts], axis=0)
    r = ds.y_raw.iloc[:, 0].to_numpy(dtype=float)
    yb, w = (r > 0).astype(int), np.abs(r)
    wa = metrics.get_metric(spec.metric)
    folds = [wa(yb[v], (oof[v] > 0).astype(int), weight=w[v])
             for _, v in cv.make_splits(ds.train, ds.y_vector, spec)]
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    res = CVResult(model="blend-z", params={"runs": runs}, features=tuple(runs),
                   metric=spec.metric, score=float(np.mean(folds)),
                   score_std=float(np.std(folds)), fold_scores=tuple(folds), oof=sig(oof),
                   test_pred=sig(test), elapsed_s=time.time() - t0)
    run_id = tracking.save_run(res, spec, slug, tr_ids, te_ids, notes="issue #29 blend")
    path = submit.write_submission(submit.build_submission(spec, te_ids, res.test_pred),
                                   spec, te_ids, run_id)
    corr = np.corrcoef([p[0] for p in parts])
    print(f"{slug}: {res.score:.5f} +- {res.score_std:.5f}  oof corr:\n{np.round(corr, 3)}")
    print(f"run {run_id}\nsubmission {path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
