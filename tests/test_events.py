"""Event-study maths: timing, closed forms and no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import panel_from_returns
from quorum import events

N_DAYS = 120


def make_panel(stock_returns: np.ndarray | None = None, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, (N_DAYS, 4))
    if stock_returns is not None:
        returns[:, 0] = stock_returns
    return panel_from_returns(returns, ["EVT", "OTHER", "SPY", "IWM"]), returns


def run(panel, event_date: str, horizon: int = 20):  # type: ignore[no-untyped-def]
    wealth = events.wealth_index(panel.ret)
    universe = events.universe_table(wealth, panel, horizon)
    table = pd.DataFrame({"symbol": ["EVT"], "event_date": [pd.Timestamp(event_date)]})
    return events.measure(table, panel, wealth, universe, {"market": "SPY"})


def test_entry_is_the_first_trading_day_strictly_after_the_public_date() -> None:
    dates = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    public = pd.Series(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-07"]))
    # Thursday -> Friday; Friday -> Monday; Saturday -> Monday; last day -> nothing.
    assert events.entry_index(dates, public).tolist() == [1, 2, 2, 4]


def test_constant_returns_give_the_closed_form() -> None:
    panel, returns = make_panel(np.full(N_DAYS, 0.01))
    er = run(panel, "2010-03-01", horizon=20)
    steps = np.arange(21)
    np.testing.assert_allclose(er.stock[0], 1.01**steps, rtol=1e-5)
    entry = panel.dates.get_loc(er.events["entry_date"].iloc[0])
    spy = np.concatenate([[1.0], np.cumprod(1.0 + returns[entry : entry + 20, 2])])
    np.testing.assert_allclose(er.benchmarks["market"][0], spy, rtol=1e-5)
    np.testing.assert_allclose(er.bhar("market")[0], 1.01**steps - spy, atol=1e-5)


def test_no_return_on_or_before_the_entry_day_is_measured() -> None:
    panel, returns = make_panel()
    base = run(panel, "2010-03-01")
    entry = panel.dates.get_loc(base.events["entry_date"].iloc[0])
    assert panel.dates[entry] > pd.Timestamp("2010-03-01")

    # Rewrite every return up to and including the entry day, for every stock.
    changed = returns.copy()
    changed[:entry, :] = 0.05  # row k is the return on trading day k + 1
    again = run(panel_from_returns(changed, ["EVT", "OTHER", "SPY", "IWM"]), "2010-03-01")
    np.testing.assert_allclose(again.stock, base.stock, rtol=1e-5)
    for name in base.benchmarks:
        np.testing.assert_allclose(again.benchmarks[name], base.benchmarks[name], rtol=1e-5)

    # Changing the first return after entry must change the result.
    changed = returns.copy()
    changed[entry, 0] += 0.1
    moved = run(panel_from_returns(changed, ["EVT", "OTHER", "SPY", "IWM"]), "2010-03-01")
    assert abs(moved.stock[0, 1] - base.stock[0, 1]) > 0.05


def test_universe_benchmark_is_the_mean_of_buy_and_hold_values() -> None:
    panel, returns = make_panel()
    er = run(panel, "2010-03-01", horizon=10)
    entry = panel.dates.get_loc(er.events["entry_date"].iloc[0])
    values = np.cumprod(1.0 + returns[entry : entry + 10], axis=0)
    np.testing.assert_allclose(er.benchmarks["universe"][0, 1:], values.mean(axis=1), rtol=1e-5)
    assert er.benchmarks["universe"][0, 0] == pytest.approx(1.0)


def test_paths_past_the_end_of_the_data_are_missing() -> None:
    panel, _ = make_panel()
    last = panel.dates[-6]
    er = run(panel, str(last.date()), horizon=20)
    assert np.isfinite(er.stock[0, :5]).all()
    assert np.isnan(er.stock[0, 5:]).all()


def test_events_without_a_symbol_or_before_the_stock_is_liquid_are_dropped() -> None:
    panel, _ = make_panel()
    wealth = events.wealth_index(panel.ret)
    universe = events.universe_table(wealth, panel, 5)
    table = pd.DataFrame(
        {
            "symbol": ["EVT", "MISSING", "EVT", None],
            # January has no month of history behind it, so nothing is liquid yet.
            "event_date": pd.to_datetime(["2010-03-01", "2010-03-01", "2010-01-11", "2010-03-01"]),
        }
    )
    er = events.measure(table, panel, wealth, universe, {})
    assert len(er.events) == 1


def test_a_stock_that_stops_trading_is_carried_at_its_last_price() -> None:
    ret = np.full((6, 1), np.nan, dtype=np.float32)
    ret[:3, 0] = 0.1
    wealth = events.wealth_index(ret)
    np.testing.assert_allclose(wealth[:, 0], [1.1, 1.21, 1.331, 1.331, 1.331, 1.331], rtol=1e-6)


def test_winsorized_mean_pulls_in_the_tails() -> None:
    x = np.concatenate([np.zeros(98), [100.0, -100.0]])
    assert events.winsorized_mean(x, tail=0.02) == pytest.approx(0.0, abs=1e-9)
    assert events.winsorized_mean(np.array([np.nan])) != events.winsorized_mean(np.array([np.nan]))
