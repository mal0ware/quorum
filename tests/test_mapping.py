"""The check of reported prices against the price file, with splits undone."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quorum import mapping, prices


def closes_frame(symbol: str, start: str, values: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(values))
    return pd.DataFrame({"symbol": symbol, "date": dates, "close": values})


def test_split_factor_parsing() -> None:
    assert prices.parse_split_factor("7:1") == pytest.approx(7.0)
    assert prices.parse_split_factor("1:15") == pytest.approx(1 / 15)
    assert prices.parse_split_factor("1398:1000") == pytest.approx(1.398)
    assert np.isnan(prices.parse_split_factor("1:0"))
    assert np.isnan(prices.parse_split_factor("nonsense"))


def test_as_traded_multiplies_by_every_later_split() -> None:
    dates = np.array(["2020-01-02", "2020-06-01", "2021-01-04"], dtype="datetime64[D]")
    adjusted = np.array([10.0, 10.0, 10.0])
    splits = np.array(["2020-03-02", "2020-09-01"], dtype="datetime64[D]")
    factors = np.array([2.0, 5.0])
    # Before both splits: 10 * 2 * 5. Between them: 10 * 5. After both: 10.
    np.testing.assert_allclose(
        mapping.as_traded(dates, adjusted, splits, factors), [100.0, 50.0, 10.0]
    )
    # A close on the split date is already in new shares.
    on_the_day = np.array(["2020-03-02"], dtype="datetime64[D]")
    np.testing.assert_allclose(
        mapping.as_traded(on_the_day, np.array([10.0]), splits, factors), [50.0]
    )
    np.testing.assert_allclose(
        mapping.as_traded(dates, adjusted, splits[:0], factors[:0]), adjusted
    )


def test_a_later_split_no_longer_breaks_the_match() -> None:
    # The stock traded near 70 and later split 7 for 1, so the file shows 10.
    closes = closes_frame("A", "2014-01-01", [10.0] * 40)
    splits = pd.DataFrame({"symbol": ["A"], "date": [pd.Timestamp("2014-06-09")], "factor": [7.0]})
    events = pd.DataFrame(
        {"symbol": ["A"], "event_date": [pd.Timestamp("2014-02-14")], "median_price": [69.0]}
    )
    out = mapping.reconcile(events, closes, splits)
    assert out["close_as_traded"].iloc[0] == pytest.approx(70.0)
    assert out["price_ratio"].iloc[0] == pytest.approx(69.0 / 70.0)
    assert out["price_ratio_adjusted"].iloc[0] == pytest.approx(6.9)
    assert bool(out["price_reconciled"].iloc[0])
    no_splits = mapping.reconcile(events, closes, splits.iloc[:0])
    assert not bool(no_splits["price_reconciled"].iloc[0])


def test_reconciled_means_inside_the_range_of_the_buying_window() -> None:
    # The stock fell from 20 to 10 over the 30 days in which the insiders bought.
    values = list(np.linspace(20.0, 10.0, 21))
    closes = closes_frame("A", "2020-03-02", values)
    day = closes["date"].iloc[-1]
    no_splits = pd.DataFrame({"symbol": [], "date": pd.to_datetime([]), "factor": []})
    events = pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "A", "GONE"],
            "event_date": [day] * 5,
            # inside the range; just above the top; far above; far below; no prices
            "median_price": [15.0, 21.5, 30.0, 5.0, 15.0],
        }
    )
    out = mapping.reconcile(events, closes, no_splits)
    assert out["price_reconciled"].tolist() == [True, True, False, False, False]
    assert np.isnan(out["close_as_traded"].iloc[4])
    # 15 against a last close of 10 is 50% off, yet it is a price that traded in the window.
    assert out["price_ratio"].iloc[0] == pytest.approx(1.5)


def test_prices_after_the_event_day_are_not_used() -> None:
    closes = closes_frame("A", "2020-03-02", [10.0] * 10 + [50.0] * 10)
    no_splits = pd.DataFrame({"symbol": [], "date": pd.to_datetime([]), "factor": []})
    events = pd.DataFrame(
        {
            "symbol": ["A"],
            "event_date": [closes["date"].iloc[9]],
            "median_price": [50.0],
        }
    )
    assert not bool(mapping.reconcile(events, closes, no_splits)["price_reconciled"].iloc[0])


def test_summary_counts_by_year_and_period() -> None:
    events = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2010-05-03", "2010-06-01", "2015-01-05"]),
            "price_ratio": [1.0, 2.0, 1.2],
            "price_ratio_adjusted": [7.0, 2.0, 1.2],
            "price_reconciled": [True, False, True],
        }
    )
    table = mapping.summary(events, pd.Timestamp("2012-07-01")).set_index("sample")
    assert table.loc["2010", "events"] == 2
    assert table.loc["2010", "within_25_percent_as_traded_close"] == 1
    assert table.loc["2010", "within_25_percent_split_adjusted_close"] == 0
    assert table.loc["before the split date", "share_reconciled"] == pytest.approx(0.5)
    assert table.loc["from the split date", "share_reconciled"] == pytest.approx(1.0)
    assert table.loc["all", "events"] == 3


def test_load_splits_reads_the_source_layout(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "C"],
            "report_date": ["2020-08-31", "2014-06-09", "2021-08-27", "2019-01-02"],
            "split_factor": ["4:1", "7:1", "1:15", "bad"],
        }
    )
    path = tmp_path / "splits.parquet"
    raw.to_parquet(path)
    out = prices.load_splits(path, {"A", "C"})
    assert out["symbol"].tolist() == ["A", "A"]
    assert out["date"].tolist() == [pd.Timestamp("2014-06-09"), pd.Timestamp("2020-08-31")]
    assert out["factor"].tolist() == [7.0, 4.0]
    assert json.loads(out.to_json()) is not None
