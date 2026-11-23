"""Fake foundation-model backends, importable by the worker process (tests/ is on sys.path)."""

import os

import numpy as np


class FakeBackend:
    """Stands in for TabICL/TabPFN/TabDPT: proba from class shares, regression = column 0."""

    built: list["FakeBackend"] = []

    def __init__(self, task, params, seed, cat_idx):
        self.task, self.params, self.seed, self.cat_idx = task, params, seed, cat_idx
        FakeBackend.built.append(self)

    def fit(self, X, y):
        self.X, self.y = X, y
        self.classes_ = np.unique(y)

    def predict_proba(self, X, seed=None):
        shares = np.array([(self.y == c).mean() for c in self.classes_])
        return np.tile(shares, (len(X), 1))

    def predict(self, X):
        return X[:, 0]


class PidBackend(FakeBackend):
    """Predicts the id of the process it runs in."""

    def predict(self, X):
        return np.full(len(X), float(os.getpid()))


class FailingBackend(FakeBackend):
    def fit(self, X, y):
        raise ValueError("bad backend param")


class DyingBackend(FakeBackend):
    def fit(self, X, y):
        os._exit(3)
