"""Turn raw football-data.co.uk CSVs into one clean match table.

Output: one row per match, sorted by kickoff, with a stable `match_id`, the result,
match statistics and the bookmaker odds we use. Column meanings follow the site's
notes file (https://www.football-data.co.uk/notes.txt); check it if a column looks odd.
"""

from __future__ import annotations

from collections.abc import Iterable
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from fml.data.download import raw_path

RESULT_TO_INT = {"H": 0, "D": 1, "A": 2}  # ordered: home win < draw < away win

# Raw column -> clean column. Missing raw columns become NaN.
STAT_COLUMNS = {
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}

# Odds kept with their original names (H/D/A = home/draw/away; C = closing).
ODDS_COLUMNS = [
    "PSH",
    "PSD",
    "PSA",  # Pinnacle, pre-closing
    "PSCH",
    "PSCD",
    "PSCA",  # Pinnacle, closing
    "AvgH",
    "AvgD",
    "AvgA",  # market average, pre-closing
    "AvgCH",
    "AvgCD",
    "AvgCA",  # market average, closing
    "MaxH",
    "MaxD",
    "MaxA",  # best available, pre-closing
    "MaxCH",
    "MaxCD",
    "MaxCA",  # best available, closing
    "B365H",
    "B365D",
    "B365A",  # Bet365, pre-closing
    "B365CH",
    "B365CD",
    "B365CA",  # Bet365, closing
]


def _read_raw_csv(path: Path) -> pd.DataFrame:
    """Read one raw file; older files are Latin-1 and some have trailing empty columns."""
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(StringIO(text), on_bad_lines="skip")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    return df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])


def clean_league_season(raw: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    """Standardise one league-season of raw rows."""
    out = pd.DataFrame()
    out["league"] = [league] * len(raw)
    out["season"] = season
    dates = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed")
    if "Time" in raw.columns:
        times = pd.to_timedelta(raw["Time"].fillna("15:00").astype(str) + ":00", errors="coerce")
        dates = dates + times.fillna(pd.Timedelta(hours=15))
    out["date"] = dates.to_numpy()
    out["home"] = raw["HomeTeam"].astype(str).str.strip().to_numpy()
    out["away"] = raw["AwayTeam"].astype(str).str.strip().to_numpy()
    out["result"] = raw["FTR"].astype(str).str.strip().map(RESULT_TO_INT).to_numpy()
    for src, dst in STAT_COLUMNS.items():
        out[dst] = pd.to_numeric(raw[src], errors="coerce").to_numpy() if src in raw else np.nan
    for col in ODDS_COLUMNS:
        out[col] = pd.to_numeric(raw[col], errors="coerce").to_numpy() if col in raw else np.nan
    out = out.dropna(subset=["result"])
    out["result"] = out["result"].astype(int)
    return out


def build_matches(
    leagues: Iterable[str], seasons: Iterable[str], raw_dir: str | Path
) -> pd.DataFrame:
    """Combine all raw files into one clean, date-sorted match table."""
    frames = []
    for league in leagues:
        for season in seasons:
            path = raw_path(raw_dir, league, season)
            if not path.exists():
                print(f"missing {path}, skipped (run the download step first)")
                continue
            frames.append(clean_league_season(_read_raw_csv(path), league, season))
    if not frames:
        raise FileNotFoundError(f"No raw files found in {raw_dir}")
    matches = pd.concat(frames, ignore_index=True)
    return finalise(matches)


def finalise(matches: pd.DataFrame) -> pd.DataFrame:
    """Sort by kickoff, drop duplicates and add a stable match id."""
    matches = matches.sort_values(["date", "league", "home"]).reset_index(drop=True)
    matches = matches.drop_duplicates(subset=["league", "date", "home", "away"])
    matches["match_id"] = (
        matches["league"]
        + "_"
        + matches["date"].dt.strftime("%Y%m%d")
        + "_"
        + matches["home"].str.replace(" ", "")
        + "_"
        + matches["away"].str.replace(" ", "")
    )
    return matches.reset_index(drop=True)
