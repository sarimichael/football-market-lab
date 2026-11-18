"""Remove the bookmaker's margin to get the market's probabilities.

Raw implied probabilities (1 / decimal odds) add up to more than 1; the excess is the
bookmaker's margin (overround). Each method below spreads that margin differently:

- multiplicative: scales every outcome by the same factor.
- power: raises the implied probabilities to a common power k, which takes more off
  longshots than favourites (the favourite-longshot bias).
- shin: Shin's (1993) model, which assumes part of the money comes from insiders.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq

METHODS = ("multiplicative", "power", "shin")


def implied(odds: Sequence[float]) -> np.ndarray:
    """Raw implied probabilities 1 / odds."""
    return 1.0 / np.asarray(odds, dtype=float)


def multiplicative(odds: Sequence[float]) -> np.ndarray:
    q = implied(odds)
    return q / q.sum()


def power(odds: Sequence[float]) -> np.ndarray:
    q = implied(odds)
    if q.sum() <= 1.0:
        return q / q.sum()
    k = brentq(lambda k: np.sum(q**k) - 1.0, 1.0, 100.0)
    return q**k


def shin(odds: Sequence[float]) -> np.ndarray:
    q = implied(odds)
    total = q.sum()
    if total <= 1.0:
        return q / total

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z**2 + 4 * (1 - z) * q**2 / total) - z) / (2 * (1 - z))

    z = brentq(lambda z: probs(z).sum() - 1.0, 0.0, 0.999)
    p = probs(z)
    return p / p.sum()


def remove_margin(odds: Sequence[float], method: str = "power") -> np.ndarray:
    """Fair probabilities from one set of decimal odds."""
    funcs = {"multiplicative": multiplicative, "power": power, "shin": shin}
    if method not in funcs:
        raise ValueError(f"method must be one of {METHODS}")
    return funcs[method](odds)


def market_probabilities(
    matches: pd.DataFrame,
    cols: Sequence[str] = ("PSCH", "PSCD", "PSCA"),
    fallback: Sequence[str] | None = ("AvgCH", "AvgCD", "AvgCA"),
    method: str = "power",
) -> pd.DataFrame:
    """Market probabilities (home, draw, away) for every match.

    Uses `cols` where all three odds exist, otherwise `fallback`. Rows with neither
    get NaN. Returns columns p_h, p_d, p_a, margin and odds_source, indexed like `matches`.
    """
    out = pd.DataFrame(index=matches.index, columns=["p_h", "p_d", "p_a", "margin"], dtype=float)
    out["odds_source"] = None
    sources = [list(cols)] + ([list(fallback)] if fallback else [])
    for source in sources:
        if not all(c in matches for c in source):
            continue
        todo = (
            out["p_h"].isna()
            & matches[source].notna().all(axis=1)
            & (matches[source] > 1).all(axis=1)
        )
        for idx in matches.index[todo]:
            odds = matches.loc[idx, source].to_numpy(dtype=float)
            out.loc[idx, ["p_h", "p_d", "p_a"]] = remove_margin(odds, method)
            out.loc[idx, "margin"] = implied(odds).sum() - 1.0
            out.loc[idx, "odds_source"] = source[0]
    return out
