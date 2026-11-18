import pandas as pd
import pytest

from fml.data.synthetic import generate
from fml.evaluation.walkforward import recency_weights, walk_forward
from fml.models.elo import compute_elo
from fml.models.outcome import BaseRateModel, EloOutcomeModel


@pytest.fixture(scope="module")
def data():
    matches = generate(leagues=("E0", "N1"), n_seasons=5, n_teams=10, seed=3)
    return matches.join(compute_elo(matches))


@pytest.fixture(scope="module")
def preds(data):
    return walk_forward(
        data, EloOutcomeModel, ["elo_diff"], test_start="2023-07-01", window_years=3
    )


def test_never_trains_on_the_future(preds):
    assert (preds["train_end"] < preds["refit_date"]).all()


def test_window_is_three_years(preds):
    assert (preds["train_start"] == preds["refit_date"] - pd.DateOffset(years=3)).all()


def test_predictions_only_from_test_start_and_each_match_once(data, preds):
    predicted = data.set_index("match_id").loc[preds["match_id"]]
    assert (predicted["date"] >= pd.Timestamp("2023-07-01")).all()
    assert preds["match_id"].is_unique
    expected = data[data["date"] >= "2023-07-01"]["match_id"]
    assert set(preds["match_id"]) == set(expected)


def test_probabilities_are_valid(preds):
    p = preds[["p_h", "p_d", "p_a"]]
    assert ((p >= 0) & (p <= 1)).all().all()
    assert p.sum(axis=1).to_numpy() == pytest.approx(1.0)


def test_base_rate_model_returns_frequencies(data):
    out = walk_forward(data, BaseRateModel, ["elo_diff"], test_start="2023-07-01")
    assert out[["p_h", "p_d", "p_a"]].nunique().max() <= out["refit_date"].nunique()


def test_recency_weights_half_life():
    w = recency_weights([0, 365, 730], half_life_days=365)
    assert w == pytest.approx([1.0, 0.5, 0.25])
    assert recency_weights([1, 2], None) is None
