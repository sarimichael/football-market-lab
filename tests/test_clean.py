from io import StringIO

import pandas as pd

from fml.data.clean import _read_raw_csv, clean_league_season, finalise

# Mimics the raw football-data.co.uk layout: two-digit year, no Pinnacle closing odds,
# a trailing empty column and a blank row at the end.
RAW = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,B365H,B365D,B365A,AvgCH,AvgCD,AvgCA,
N1,12/08/23,20:00,Ajax,PSV,1,1,D,12,9,2.1,3.6,3.3,2.15,3.5,3.25,
N1,13/08/23,14:30, Feyenoord ,Utrecht,3,0,H,15,4,1.4,4.8,7.5,1.42,4.7,7.2,
,,,,,,,,,,,,,,,,
"""


def _clean(tmp_path):
    path = tmp_path / "N1_2324.csv"
    path.write_text(RAW, encoding="latin-1")
    return finalise(clean_league_season(_read_raw_csv(path), "N1", "2324"))


def test_rows_dates_and_results(tmp_path):
    df = _clean(tmp_path)
    assert len(df) == 2
    assert df.loc[0, "date"] == pd.Timestamp("2023-08-12 20:00")
    assert list(df["result"]) == [1, 0]  # draw, home win
    assert df.loc[1, "home"] == "Feyenoord"  # whitespace stripped


def test_missing_odds_columns_become_nan(tmp_path):
    df = _clean(tmp_path)
    assert df["PSCH"].isna().all()
    assert df.loc[0, "AvgCH"] == 2.15


def test_match_id_is_stable_and_unique(tmp_path):
    df = _clean(tmp_path)
    assert df["match_id"].is_unique
    assert df.loc[0, "match_id"] == "N1_20230812_Ajax_PSV"


def test_read_raw_drops_unnamed_columns(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text(RAW, encoding="latin-1")
    raw = _read_raw_csv(path)
    assert not any(str(c).startswith("Unnamed") for c in raw.columns)
    assert len(pd.read_csv(StringIO(RAW))) == 3  # the blank row exists in the raw text
