"""Turn a rating difference into home/draw/away probabilities.

A multinomial logistic regression on the pre-match Elo difference. It is refitted
inside the walk-forward loop, so it only ever learns from past matches.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class BaseRateModel:
    """Ignores the teams: predicts the home/draw/away frequencies of the training window.

    The floor every real model has to beat.
    """

    feature_cols = ["elo_diff"]

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        counts = np.array([w[np.asarray(y) == c].sum() for c in range(3)])
        self.rates = counts / counts.sum()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.rates, (len(X), 1))


class EloOutcomeModel:
    """P(home win), P(draw), P(away win) from the Elo difference."""

    feature_cols = ["elo_diff"]

    def __init__(self, scale: float = 100.0) -> None:
        self.scale = scale
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        self.model.fit(np.asarray(X, dtype=float) / self.scale, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Columns ordered home, draw, away (classes 0, 1, 2)."""
        proba = self.model.predict_proba(np.asarray(X, dtype=float) / self.scale)
        out = np.zeros((len(proba), 3))
        for j, cls in enumerate(self.model.classes_):
            out[:, int(cls)] = proba[:, j]
        return out
