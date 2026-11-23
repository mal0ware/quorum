"""Newey-West regression and the calendar-time portfolio."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import panel_from_returns
from quorum import portfolio
from quorum.stats import mean_nw, ols_nw


def test_ols_recovers_exact_coefficients() -> None:
    x = np.column_stack([np.ones(50), np.linspace(-1, 1, 50)])
    fit = ols_nw(x @ np.array([0.5, 2.0]), x)
    np.testing.assert_allclose(fit.coef, [0.5, 2.0], atol=1e-12)
    np.testing.assert_allclose(fit.se, 0.0, atol=1e-10)


def test_mean_with_no_lags_is_the_white_standard_error() -> None:
    x = np.array([1.0, 2.0, 4.0, 9.0])
    fit = mean_nw(x, 0)
    assert fit.coef[0] == pytest.approx(4.0)
    # sqrt(sum (x - mean)^2) / n = sqrt(9 + 4 + 0 + 25) / 4
    assert fit.se[0] == pytest.approx(np.sqrt(38.0) / 4.0)
    assert fit.t[0] == pytest.approx(4.0 / (np.sqrt(38.0) / 4.0))


def test_newey_west_weights_match_a_hand_calculation() -> None:
    x = np.array([1.0, -1.0, 2.0, -2.0, 0.0])
    e = x - x.mean()
    g0 = (e**2).sum()
    g1 = (e[1:] * e[:-1]).sum()
    g2 = (e[2:] * e[:-2]).sum()
    # Bartlett weights with L = 2: 2/3 and 1/3, each autocovariance counted twice.
    expected = np.sqrt(g0 + 2 * (2 / 3) * g1 + 2 * (1 / 3) * g2) / 5
    assert mean_nw(x, 2).se[0] == pytest.approx(expected)


def test_positive_autocorrelation_widens_the_standard_error() -> None:
    rng = np.random.default_rng(1)
    shocks = rng.normal(size=600)
    x = np.convolve(shocks, np.ones(6), mode="valid")  # overlapping sums
    assert mean_nw(x, 5).se[0] > 1.5 * mean_nw(x, 0).se[0]


def test_too_few_observations_give_nan() -> None:
    assert np.isnan(mean_nw(np.array([1.0]), 0).coef[0])
    assert mean_nw(np.array([np.nan, np.nan]), 0).nobs == 0


def make_panel():  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0005, 0.01, (130, 3))
    return panel_from_returns(returns, ["A", "B", "SPY"]), returns


def test_monthly_returns_compound_the_daily_ones() -> None:
    panel, returns = make_panel()
    monthly = portfolio.monthly_returns(panel)
    assert monthly.shape == (len(panel.months), 3)
    in_month = np.flatnonzero(panel.month_of_day == 2)
    expected = np.prod(1.0 + returns[in_month - 1], axis=0) - 1.0  # row k is day k + 1
    np.testing.assert_allclose(monthly[2], expected, rtol=1e-5)


def test_a_signal_is_held_in_the_following_months_only() -> None:
    panel, _ = make_panel()
    monthly = portfolio.monthly_returns(panel)
    signal = portfolio.signal_matrix(panel, np.array([2]), np.array([0]))
    ret, count = portfolio.calendar_time(signal, monthly, panel.liquid, hold=2)
    assert count.tolist()[:6] == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    assert ret[3] == pytest.approx(monthly[3, 0])
    assert np.isnan(ret[2])  # not held in the signal month itself
    with pytest.raises(ValueError):
        portfolio.calendar_time(signal, monthly, panel.liquid, hold=0)


def test_signals_outside_the_panel_are_ignored() -> None:
    panel, _ = make_panel()
    signal = portfolio.signal_matrix(panel, np.array([-1, 99, 1]), np.array([0, 0, -1]))
    assert not signal.any()


def test_universe_return_is_the_equal_weight_mean_of_liquid_stocks() -> None:
    panel, _ = make_panel()
    monthly = portfolio.monthly_returns(panel)
    universe = portfolio.universe_return(monthly, panel.liquid)
    assert np.isnan(universe[0])  # nothing is liquid in the first month
    assert universe[3] == pytest.approx(monthly[3].mean())


def test_alpha_and_beta_are_recovered() -> None:
    rng = np.random.default_rng(3)
    market = rng.normal(0.005, 0.04, 240)
    riskfree = np.full(240, 0.001)
    ret = riskfree + 0.004 + 1.3 * (market - riskfree)
    stats = portfolio.summarise(
        ret, np.full(240, 10.0), market, market, riskfree, np.ones(240, dtype=np.bool_)
    )
    assert stats.alpha == pytest.approx(0.004)
    assert stats.beta == pytest.approx(1.3)
    assert stats.months == 240
    assert stats.mean_holdings == 10.0
