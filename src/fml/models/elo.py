"""Elo ratings for football clubs, computed strictly in time order.

Each match row gets the two teams' ratings *before* kickoff, so the ratings can be used
as features without leaking the result. Ratings are kept separately per league.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

START_RATING = 1500.0


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home_advantage: float = 60.0
    season_regression: float = 0.25
    new_team_quantile: float = 0.2
    goal_diff_multiplier: bool = True


def expected_home_score(r_home: float, r_away: float, home_advantage: float) -> float:
    """Expected score of the home side (win = 1, draw = 0.5, loss = 0)."""
    return 1.0 / (1.0 + 10 ** (-(r_home + home_advantage - r_away) / 400.0))


def goal_diff_factor(goal_diff: float) -> float:
    """Bigger wins move ratings more (the World Football Elo convention)."""
    gd = abs(goal_diff)
    if np.isnan(gd) or gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def compute_elo(matches: pd.DataFrame, params: EloParams | None = None) -> pd.DataFrame:
    """Pre-match Elo ratings for every match.

    `matches` needs columns league, season, date, home, away, result and, if the goal
    multiplier is on, home_goals and away_goals. Returns a frame indexed like `matches`
    with elo_home, elo_away and elo_diff (home minus away, before home advantage).
    """
    params = params or EloParams()
    order = matches.sort_values("date", kind="stable").index
    elo_home = pd.Series(np.nan, index=matches.index)
    elo_away = pd.Series(np.nan, index=matches.index)
    ratings: dict[str, dict[str, float]] = {}
    current_season: dict[str, str] = {}
    # Ratings frozen at the start of each season. Newcomers are rated from this snapshot,
    # never from ratings already updated by other matches, which could leak results.
    season_start: dict[str, list[float]] = {}

    for idx in order:
        row = matches.loc[idx]
        league, season = row["league"], row["season"]
        table = ratings.setdefault(league, {})

        if current_season.get(league) != season:
            # New season: pull existing ratings part of the way back to the mean.
            if table:
                mean = np.mean(list(table.values()))
                for team in table:
                    table[team] = mean + (1 - params.season_regression) * (table[team] - mean)
            current_season[league] = season
            season_start[league] = list(table.values())

        for team in (row["home"], row["away"]):
            if team not in table:
                # A promoted (or first-seen) team starts near the bottom of the league.
                known = season_start[league]
                table[team] = (
                    float(np.quantile(known, params.new_team_quantile)) if known else START_RATING
                )

        r_h, r_a = table[row["home"]], table[row["away"]]
        elo_home[idx], elo_away[idx] = r_h, r_a

        score = {0: 1.0, 1: 0.5, 2: 0.0}[int(row["result"])]
        expected = expected_home_score(r_h, r_a, params.home_advantage)
        k = params.k
        if params.goal_diff_multiplier and "home_goals" in row and "away_goals" in row:
            k *= goal_diff_factor(row["home_goals"] - row["away_goals"])
        delta = k * (score - expected)
        table[row["home"]] = r_h + delta
        table[row["away"]] = r_a - delta

    return pd.DataFrame(
        {"elo_home": elo_home, "elo_away": elo_away, "elo_diff": elo_home - elo_away}
    )
