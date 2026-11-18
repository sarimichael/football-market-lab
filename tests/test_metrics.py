import numpy as np
import pytest

from fml.evaluation.metrics import brier, calibration_table, log_loss, rps


def test_perfect_forecast_scores_zero():
    probs = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    outcomes = np.array([0, 1, 2])
    assert rps(probs, outcomes) == pytest.approx(0.0)
    assert brier(probs, outcomes) == pytest.approx(0.0)
    assert log_loss(probs, outcomes) == pytest.approx(0.0, abs=1e-9)


def test_rps_by_hand():
    # Uniform forecast, home win: cumulative forecast (1/3, 2/3) vs (1, 1).
    # RPS = ((1/3 - 1)^2 + (2/3 - 1)^2) / 2 = 5/18
    assert rps(np.array([[1 / 3] * 3]), np.array([0])) == pytest.approx(5 / 18)


def test_rps_punishes_far_misses_more():
    confident_home = np.array([[0.8, 0.15, 0.05]])
    assert rps(confident_home, np.array([2])) > rps(confident_home, np.array([1]))


def test_log_loss_by_hand():
    assert log_loss(np.array([[0.5, 0.3, 0.2]]), np.array([1])) == pytest.approx(-np.log(0.3))


def test_calibration_table_shape():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([2, 1, 1], size=500)
    outcomes = np.array([rng.choice(3, p=p) for p in probs])
    table = calibration_table(probs, outcomes, n_bins=5)
    assert {"predicted", "observed", "n"} <= set(table.columns)
    assert table["n"].sum() == 1500
