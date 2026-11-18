"""Scoring rules for home/draw/away forecasts.

All functions take `probs` of shape (n, 3) in the order home, draw, away and
`outcomes` of shape (n,) with 0 = home win, 1 = draw, 2 = away win. Lower is better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _one_hot(outcomes: np.ndarray) -> np.ndarray:
    outcomes = np.asarray(outcomes, dtype=int)
    onehot = np.zeros((len(outcomes), 3))
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    return onehot


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean ranked probability score (Epstein 1969; Constantinou & Fenton 2012)."""
    cum_p = np.cumsum(np.asarray(probs, dtype=float), axis=1)[:, :2]
    cum_o = np.cumsum(_one_hot(outcomes), axis=1)[:, :2]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / 2))


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean negative log probability given to what actually happened."""
    probs = np.asarray(probs, dtype=float)
    picked = probs[np.arange(len(probs)), np.asarray(outcomes, dtype=int)]
    return float(-np.mean(np.log(np.clip(picked, EPS, 1.0))))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean multi-class Brier score."""
    return float(np.mean(np.sum((np.asarray(probs) - _one_hot(outcomes)) ** 2, axis=1)))


def score_all(probs: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(outcomes)),
        "rps": rps(probs, outcomes),
        "log_loss": log_loss(probs, outcomes),
        "brier": brier(probs, outcomes),
    }


def calibration_table(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency per probability bin, for all three outcomes pooled."""
    p = np.asarray(probs, dtype=float).ravel()
    hit = _one_hot(outcomes).ravel()
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    df = pd.DataFrame({"bin": bins, "predicted": p, "observed": hit})
    table = df.groupby("bin").agg(
        predicted=("predicted", "mean"), observed=("observed", "mean"), n=("observed", "size")
    )
    return table.reset_index(drop=True)
