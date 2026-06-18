"""Trivial baselines the neural surrogate must beat to be credible.

Each baseline exposes the SAME interface as pbg_torch.SurrogateNet —
``predict_next(X_raw) -> next absolute target values`` — so the evaluator can
roll them out autoregressively exactly like the surrogate and compare on equal
footing. Feature ordering is targets-then-drivers, so the first ``n_targets``
columns of a feature row are the current target values.

  - PersistencePredictor   next = current (delta = 0). The "does the surrogate
                           do anything beyond noting the signal is smooth?" test.
  - MeanDeltaPredictor     next = current + mean training per-step delta (a
                           constant drift). Beats persistence on monotone trends.
  - LinearPredictor        next = current + W·features (ridge regression on the
                           per-step delta). A linear dynamical model — the bar a
                           nonlinear net must clear to justify itself.
"""
from __future__ import annotations

import numpy as np


def _training_signal(X, Y, n_targets, parameterization):
    if parameterization == "delta":
        return Y - X[:, :n_targets]
    return Y


class PersistencePredictor:
    def __init__(self, spec):
        self.spec = spec

    def fit(self, X, Y):
        return self

    def predict_next(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X[:, : self.spec.n_targets].copy()


class MeanDeltaPredictor:
    def __init__(self, spec):
        self.spec = spec
        self._mean_delta = None

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        self._mean_delta = (Y - X[:, : self.spec.n_targets]).mean(axis=0)
        return self

    def predict_next(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X[:, : self.spec.n_targets] + self._mean_delta


class LinearPredictor:
    """Ridge regression from features to the per-step signal (delta or absolute).

    Closed-form normal equations with L2 regularization — no extra deps.
    """

    def __init__(self, spec, ridge: float = 1.0):
        self.spec = spec
        self.ridge = ridge
        self._W = None  # (n_features+1, n_targets), last row = bias

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        sig = _training_signal(X, Y, self.spec.n_targets, self.spec.parameterization)
        # standardize features for a well-conditioned solve
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0)
        self._sd = np.where(self._sd < 1e-8, 1.0, self._sd)
        Xs = (X - self._mu) / self._sd
        Xb = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        n_feat = Xb.shape[1]
        reg = self.ridge * np.eye(n_feat)
        reg[-1, -1] = 0.0  # don't regularize the bias
        self._W = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ sig)
        return self

    def predict_next(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xs = (X - self._mu) / self._sd
        Xb = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        sig = Xb @ self._W
        if self.spec.parameterization == "delta":
            return X[:, : self.spec.n_targets] + sig
        return sig


def build_baselines(spec):
    """Return {name: predictor} for the standard trivial baselines."""
    return {
        "persistence": PersistencePredictor(spec),
        "mean-delta": MeanDeltaPredictor(spec),
        "linear": LinearPredictor(spec),
    }
