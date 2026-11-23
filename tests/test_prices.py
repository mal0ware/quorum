"""Total returns, bad-print guards and the liquidity flag."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quorum import prices


def long_prices(
    closes: dict[str, list[float]], volume: float = 1e6
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    n = len(next(iter(closes.values())))
    calendar = pd.bdate_range("2020-01-01", periods=n, name="date")
    frames = [
        pd.DataFrame({"symbol": s, "date": calendar, "close": c, "volume": volume})
        for s, c in closes.items()
    ]
    return pd.concat(frames, ignore_index=True), calendar


NO_DIVIDENDS = pd.DataFrame({"symbol": [], "date": pd.to_datetime([]), "amount": []})


def test_return_includes_the_dividend_on_its_ex_date() -> None:
    px, calendar = long_prices({"A": [100.0, 99.0, 101.0]})
    dv = pd.DataFrame({"symbol": ["A"], "date": [calendar[1]], "amount": [2.0]})
    panel = prices.build_panel(px, dv, calendar, 0.0)
    assert np.isnan(panel.ret[0, 0])
    assert panel.ret[1, 0] == pytest.approx((99.0 + 2.0) / 100.0 - 1.0)
    assert panel.ret[2, 0] == pytest.approx(101.0 / 99.0 - 1.0)


def test_an_isolated_spike_is_dropped_and_the_return_spans_it() -> None:
    px, calendar = long_prices({"A": [10.0, 10.5, 1050.0, 10.6, 10.7]})
    panel = prices.build_panel(px, NO_DIVIDENDS, calendar, 0.0)
    assert panel.notes["spike_closes_dropped"] == 1
    assert np.isnan(panel.ret[2, 0])
    assert panel.ret[3, 0] == pytest.approx(10.6 / 10.5 - 1.0)


def test_a_real_crash_is_kept() -> None:
    px, calendar = long_prices({"A": [10.0, 10.0, 1.0, 1.1, 1.0]})
    panel = prices.build_panel(px, NO_DIVIDENDS, calendar, 0.0)
    assert panel.notes["spike_closes_dropped"] == 0
    assert panel.ret[2, 0] == pytest.approx(-0.9)


def test_an_impossible_jump_that_persists_is_left_missing() -> None:
    px, calendar = long_prices({"A": [1.0, 1.0, 50.0, 51.0, 50.0]})
    panel = prices.build_panel(px, NO_DIVIDENDS, calendar, 0.0)
    assert panel.notes["extreme_returns_blanked"] == 1
    assert np.isnan(panel.ret[2, 0])
    assert panel.ret[3, 0] == pytest.approx(0.02)


def test_non_positive_closes_and_off_calendar_rows_are_ignored() -> None:
    px, calendar = long_prices({"A": [10.0, 0.0, 11.0]})
    extra = pd.DataFrame(
        {"symbol": ["A"], "date": [pd.Timestamp("2020-01-04")], "close": [500.0], "volume": [1.0]}
    )  # a Saturday
    panel = prices.build_panel(pd.concat([px, extra]), NO_DIVIDENDS, calendar, 0.0)
    assert np.isnan(panel.ret[1, 0])
    assert panel.ret[2, 0] == pytest.approx(0.1)


def test_liquidity_uses_the_month_before() -> None:
    calendar = pd.bdate_range("2020-01-01", "2020-03-31", name="date")
    thin = np.where(calendar.month == 1, 10.0, 1e6)  # illiquid in January only
    px = pd.concat(
        [
            pd.DataFrame({"symbol": "THIN", "date": calendar, "close": 20.0, "volume": thin}),
            pd.DataFrame({"symbol": "DEEP", "date": calendar, "close": 20.0, "volume": 1e6}),
        ]
    )
    panel = prices.build_panel(px, NO_DIVIDENDS, calendar, 100_000.0)
    deep, thin_col = panel.column("DEEP"), panel.column("THIN")
    assert panel.liquid[:, deep].tolist() == [False, True, True]  # no history before January
    assert panel.liquid[:, thin_col].tolist() == [False, False, True]


def test_cik_map_takes_the_first_ticker_and_skips_blank_ciks(tmp_path: Path) -> None:
    path = tmp_path / "tickers.json"
    rows = {
        "0": {"cik_str": 5, "ticker": "AAA"},
        "1": {"cik_str": 5, "ticker": "AAA-B"},
        "2": {"cik_str": None, "ticker": "NOPE"},
        "3": {"cik_str": 9, "ticker": "BBB"},
    }
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert prices.load_cik_map(path).to_dict() == {5: "AAA", 9: "BBB"}
