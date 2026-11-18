"""Walk-forward evaluation with a rolling training window.

At each refit date t the model is trained only on matches in [t - window, t) and then
predicts matches in [t, next refit date). The window moves forward with time, so the
model always looks at the most recent `window_years` of football and never at the future.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def refit_dates(start: str | pd.Timestamp, end: pd.Timestamp, step: str = "MS") -> pd.DatetimeIndex:
    """Refit dates from `start` up to `end` (e.g. the first day of every month)."""
    start = pd.Timestamp(start)
    dates = pd.date_range(start=start, end=end, freq=step)
    if len(dates) == 0 or dates[0] != start:
        dates = dates.insert(0, start)
    return dates


def recency_weights(ages_days: np.ndarray, half_life_days: float | None) -> np.ndarray | None:
    """Exponential time decay: a match `half_life_days` old counts half as much."""
    if half_life_days is None:
        return None
    return 0.5 ** (np.asarray(ages_days, dtype=float) / half_life_days)


def walk_forward(
    data: pd.DataFrame,
    model_factory: Callable[[], object],
    feature_cols: list[str],
    target_col: str = "result",
    test_start: str = "2023-07-01",
    window_years: float = 3,
    step: str = "MS",
    half_life_days: float | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Out-of-sample probabilities for every match on or after `test_start`.

    `model_factory()` must return an object with fit(X, y, sample_weight) and
    predict_proba(X) -> (n, 3). Returns one row per predicted match with p_h, p_d, p_a,
    plus the window used, so the absence of leakage can be checked.
    """
    data = data.dropna(subset=feature_cols + [target_col])
    if float(window_years).is_integer():
        window = pd.DateOffset(years=int(window_years))
    else:
        window = pd.Timedelta(days=365.25 * window_years)
    dates = refit_dates(test_start, data[date_col].max(), step)
    edges = list(dates) + [data[date_col].max() + pd.Timedelta(seconds=1)]

    results = []
    for t, t_next in zip(edges[:-1], edges[1:], strict=True):
        test = data[(data[date_col] >= t) & (data[date_col] < t_next)]
        if test.empty:
            continue
        train_start = t - window
        train = data[(data[date_col] >= train_start) & (data[date_col] < t)]
        if train[target_col].nunique() < 3:
            continue
        ages = (t - train[date_col]).dt.total_seconds().to_numpy() / 86400
        model = model_factory()
        model.fit(
            train[feature_cols].to_numpy(),
            train[target_col].to_numpy(),
            sample_weight=recency_weights(ages, half_life_days),
        )
        probs = model.predict_proba(test[feature_cols].to_numpy())
        results.append(
            pd.DataFrame(
                {
                    "match_id": test["match_id"].to_numpy(),
                    "p_h": probs[:, 0],
                    "p_d": probs[:, 1],
                    "p_a": probs[:, 2],
                    "train_start": train_start,
                    "train_end": train[date_col].max(),
                    "refit_date": t,
                    "n_train": len(train),
                }
            )
        )
    return pd.concat(results, ignore_index=True)
