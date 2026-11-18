"""Download season CSVs from football-data.co.uk.

The raw files are never committed to the repo: the site's author holds the copyright,
so anyone reproducing the project downloads them with this script.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def raw_path(raw_dir: str | Path, league: str, season: str) -> Path:
    """Where the raw CSV for one league-season is stored, e.g. data/raw/E0_2324.csv."""
    return Path(raw_dir) / f"{league}_{season}.csv"


def download(
    leagues: Iterable[str],
    seasons: Iterable[str],
    raw_dir: str | Path,
    overwrite: bool = False,
    pause_seconds: float = 1.0,
) -> list[Path]:
    """Download every league-season CSV that is not already on disk.

    Args:
        leagues: Division codes such as "E0" (Premier League) or "N1" (Eredivisie).
        seasons: Season codes such as "2324" for 2023/24.
        raw_dir: Folder for the raw files.
        overwrite: Re-download files that already exist (use for the current season).
        pause_seconds: Wait between requests to be polite to the site.

    Returns:
        Paths of all files now on disk.
    """
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for league in leagues:
        for season in seasons:
            path = raw_path(raw_dir, league, season)
            if path.exists() and not overwrite:
                paths.append(path)
                continue
            url = BASE_URL.format(season=season, league=league)
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            path.write_bytes(response.content)
            print(f"downloaded {url} -> {path}")
            paths.append(path)
            time.sleep(pause_seconds)
    return paths
