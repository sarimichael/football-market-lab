import numpy as np
import pandas as pd
import pytest

from fml.market.margin import implied, market_probabilities, remove_margin

ODDS = [1.50, 4.20, 7.00]  # favourite, draw, longshot; about 5.6% margin


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_probabilities_sum_to_one(method):
    p = remove_margin(ODDS, method)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert (p > 0).all()


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_fair_odds_are_unchanged(method):
    fair = np.array([0.5, 0.3, 0.2])
    assert remove_margin(1 / fair, method) == pytest.approx(fair, abs=1e-9)


def test_power_and_shin_take_more_from_the_longshot():
    mult = remove_margin(ODDS, "multiplicative")
    for method in ("power", "shin"):
        p = remove_margin(ODDS, method)
        assert p[2] < mult[2]  # longshot gets less probability
        assert p[0] > mult[0]  # favourite gets more


def test_multiplicative_by_hand():
    q = implied(ODDS)
    assert remove_margin(ODDS, "multiplicative") == pytest.approx(q / q.sum())


def test_market_probabilities_uses_fallback_and_reports_margin():
    df = pd.DataFrame(
        {
            "PSCH": [2.0, np.nan],
            "PSCD": [3.4, np.nan],
            "PSCA": [4.0, np.nan],
            "AvgCH": [1.9, 2.5],
            "AvgCD": [3.3, 3.2],
            "AvgCA": [3.9, 2.8],
        }
    )
    out = market_probabilities(df, method="multiplicative")
    assert out.loc[0, "odds_source"] == "PSCH"
    assert out.loc[1, "odds_source"] == "AvgCH"
    assert out.loc[0, "margin"] == pytest.approx(1 / 2.0 + 1 / 3.4 + 1 / 4.0 - 1)
    assert out[["p_h", "p_d", "p_a"]].sum(axis=1).to_numpy() == pytest.approx([1.0, 1.0])


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        remove_margin(ODDS, "magic")
