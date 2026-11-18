import pandas as pd
import pytest

from fml.data.synthetic import generate
from fml.models.elo import EloParams, compute_elo, expected_home_score


@pytest.fixture(scope="module")
def matches():
    return generate(leagues=("E0",), n_seasons=2, n_teams=6, seed=1)


def test_expected_score_symmetric_without_home_advantage():
    assert expected_home_score(1500, 1500, 0) == pytest.approx(0.5)
    assert expected_home_score(1600, 1500, 0) + expected_home_score(1500, 1600, 0) == pytest.approx(
        1
    )


def test_winner_gains_loser_loses():
    df = pd.DataFrame(
        {
            "league": ["E0", "E0"],
            "season": ["2324", "2324"],
            "date": pd.to_datetime(["2023-08-12", "2023-08-19"]),
            "home": ["A", "A"],
            "away": ["B", "B"],
            "result": [0, 0],
            "home_goals": [1, 1],
            "away_goals": [0, 0],
        }
    )
    elo = compute_elo(df, EloParams(home_advantage=0))
    assert elo.loc[1, "elo_home"] > elo.loc[0, "elo_home"]
    assert elo.loc[1, "elo_away"] < elo.loc[0, "elo_away"]
    # Zero-sum: what A gains, B loses.
    gain = elo.loc[1, "elo_home"] - elo.loc[0, "elo_home"]
    loss = elo.loc[0, "elo_away"] - elo.loc[1, "elo_away"]
    assert gain == pytest.approx(loss)


def test_no_leakage_ratings_ignore_the_match_itself(matches):
    """Changing a match's result must not change any pre-match rating up to and including it."""
    before = compute_elo(matches)
    target = len(matches) // 2
    altered = matches.copy()
    altered.loc[target, "result"] = (altered.loc[target, "result"] + 1) % 3
    altered.loc[target, ["home_goals", "away_goals"]] = [7, 0]
    after = compute_elo(altered)
    earlier_or_same = matches["date"] <= matches.loc[target, "date"]
    pd.testing.assert_frame_equal(before[earlier_or_same], after[earlier_or_same])
    # ...but it does change later ratings, so the update really happens.
    assert not before[~earlier_or_same].equals(after[~earlier_or_same])


def test_promoted_team_starts_below_average(matches):
    elo = compute_elo(matches, EloParams(new_team_quantile=0.2))
    data = matches.join(elo)
    second = data[data["season"] == data["season"].max()]
    first_teams = set(data[data["season"] == data["season"].min()]["home"])
    newcomers = set(second["home"]) - first_teams
    assert newcomers
    first_game = second.sort_values("date").groupby("home").head(1)
    new_rating = first_game[first_game["home"].isin(newcomers)]["elo_home"].mean()
    old_rating = first_game[~first_game["home"].isin(newcomers)]["elo_home"].mean()
    assert new_rating < old_rating
