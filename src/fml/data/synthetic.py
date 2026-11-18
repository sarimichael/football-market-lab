"""Synthetic matches in the same format as the real data.

Used by the tests and by `python -m fml demo`, so the whole pipeline can be checked
without internet access. Team strengths drift over time, three teams are replaced each
season (like promotion and relegation), goals are Poisson, and bookmaker odds are the
true probabilities plus a margin and some pricing noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from fml.data.clean import finalise


def outcome_probs(lam_home: float, lam_away: float, max_goals: int = 10) -> np.ndarray:
    """P(home win), P(draw), P(away win) for independent Poisson goal counts."""
    g = np.arange(max_goals + 1)
    grid = np.outer(poisson.pmf(g, lam_home), poisson.pmf(g, lam_away))
    p = np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])
    return p / p.sum()


def _round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """Double round robin with the circle method: each team plays every other twice."""
    teams = list(teams)
    n = len(teams)
    rounds = []
    for r in range(n - 1):
        pairs = []
        for i in range(n // 2):
            a, b = teams[i], teams[n - 1 - i]
            pairs.append((a, b) if r % 2 == 0 else (b, a))
        rounds.append(pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds + [[(b, a) for a, b in rnd] for rnd in rounds]


def generate(
    leagues: tuple[str, ...] = ("E0", "D1", "F1", "N1"),
    first_season_year: int = 2020,
    n_seasons: int = 6,
    n_teams: int = 18,
    margin: float = 0.03,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a clean match table shaped like `build_matches` output."""
    rng = np.random.default_rng(seed)
    rows = []
    for league in leagues:
        strength = {f"{league}_team{i:02d}": rng.normal(0, 0.35) for i in range(n_teams)}
        next_id = n_teams
        for s in range(n_seasons):
            year = first_season_year + s
            season = f"{str(year)[2:]}{str(year + 1)[2:]}"
            start = pd.Timestamp(f"{year}-08-15")
            for r, pairs in enumerate(_round_robin(list(strength))):
                date = start + pd.Timedelta(days=7 * r + (r // 17) * 21)  # winter break
                for home, away in pairs:
                    lam_h = np.exp(0.25 + 0.15 + strength[home] - strength[away])
                    lam_a = np.exp(0.25 - strength[home] + strength[away])
                    p_true = outcome_probs(lam_h, lam_a)
                    hg, ag = rng.poisson(lam_h), rng.poisson(lam_a)
                    result = 0 if hg > ag else (1 if hg == ag else 2)
                    # Closing price: true probability + margin + small noise.
                    noisy = p_true * np.exp(rng.normal(0, 0.03, 3))
                    close = 1 / (noisy / noisy.sum() * (1 + margin))
                    # Earlier price: noisier, slightly bigger margin.
                    noisy_open = p_true * np.exp(rng.normal(0, 0.07, 3))
                    open_ = 1 / (noisy_open / noisy_open.sum() * (1 + margin + 0.01))
                    rows.append(
                        {
                            "league": league,
                            "season": season,
                            "date": date + pd.Timedelta(hours=15),
                            "home": home,
                            "away": away,
                            "result": result,
                            "home_goals": hg,
                            "away_goals": ag,
                            "PSH": open_[0],
                            "PSD": open_[1],
                            "PSA": open_[2],
                            "PSCH": close[0],
                            "PSCD": close[1],
                            "PSCA": close[2],
                            "AvgCH": close[0] * 0.98,
                            "AvgCD": close[1] * 0.98,
                            "AvgCA": close[2] * 0.98,
                        }
                    )
                # Strengths drift a little every round.
                for team in strength:
                    strength[team] += rng.normal(0, 0.02)
            # "Relegate" the three weakest teams and "promote" three new ones.
            for team in sorted(strength, key=strength.get)[:3]:
                del strength[team]
                strength[f"{league}_team{next_id:02d}"] = rng.normal(-0.3, 0.15)
                next_id += 1
    return finalise(pd.DataFrame(rows))
