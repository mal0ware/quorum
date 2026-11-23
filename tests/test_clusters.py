"""Cluster detection on hand-built filings."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import buys_frame
from quorum.clusters import count_labels, detect_clusters


def test_three_insiders_make_one_event_on_the_third_public_date() -> None:
    buys = buys_frame(
        [
            (1, 10, "2015-03-02", "2015-03-03"),
            (1, 11, "2015-03-05", "2015-03-06"),
            (1, 12, "2015-03-10", "2015-03-12"),
        ]
    )
    events = detect_clusters(buys)
    assert len(events) == 1
    event = events.iloc[0]
    assert event["event_date"] == pd.Timestamp("2015-03-12")
    assert event["n_insiders"] == 3
    assert event["insiders"] == (10, 11, 12)
    assert event["value"] == 3000.0
    assert event["median_price"] == 10.0


def test_two_insiders_are_not_a_cluster() -> None:
    buys = buys_frame([(1, 10, "2015-03-02", "2015-03-03"), (1, 11, "2015-03-05", "2015-03-06")])
    assert detect_clusters(buys).empty


def test_one_insider_buying_three_times_is_not_a_cluster() -> None:
    buys = buys_frame([(1, 10, f"2015-03-0{d}", f"2015-03-0{d + 1}") for d in (2, 3, 4)])
    assert detect_clusters(buys).empty


def test_insiders_at_different_issuers_do_not_combine() -> None:
    buys = buys_frame(
        [
            (1, 10, "2015-03-02", "2015-03-03"),
            (1, 11, "2015-03-03", "2015-03-04"),
            (2, 12, "2015-03-04", "2015-03-05"),
        ]
    )
    assert detect_clusters(buys).empty


def test_trades_older_than_the_window_do_not_count() -> None:
    # The first trade is 31 days before the third filing: outside a 30-day window.
    rows = [
        (1, 10, "2015-03-01", "2015-03-02"),
        (1, 11, "2015-03-20", "2015-03-21"),
        (1, 12, "2015-03-30", "2015-04-01"),
    ]
    assert detect_clusters(buys_frame(rows), window_days=30).empty
    assert len(detect_clusters(buys_frame(rows), window_days=31)) == 1


def test_the_event_is_dated_by_the_filing_not_the_trade() -> None:
    # Three insiders traded on the same day, but the third filing arrives two weeks late.
    buys = buys_frame(
        [
            (1, 10, "2015-03-02", "2015-03-03"),
            (1, 11, "2015-03-02", "2015-03-04"),
            (1, 12, "2015-03-02", "2015-03-18"),
        ]
    )
    events = detect_clusters(buys)
    assert events["event_date"].tolist() == [pd.Timestamp("2015-03-18")]


def test_a_late_filing_of_a_stale_trade_does_not_complete_a_cluster() -> None:
    buys = buys_frame(
        [
            (1, 10, "2015-03-02", "2015-03-03"),
            (1, 11, "2015-03-03", "2015-03-04"),
            (1, 12, "2015-01-05", "2015-03-05"),  # traded two months before it was filed
        ]
    )
    assert detect_clusters(buys).empty


def test_cooldown_suppresses_a_second_event_until_it_expires() -> None:
    first = [(1, 10 + k, "2015-03-02", "2015-03-03") for k in range(3)]
    soon = [(1, 20 + k, "2015-06-01", "2015-06-02") for k in range(3)]
    later = [(1, 30 + k, "2015-10-01", "2015-10-02") for k in range(3)]
    events = detect_clusters(buys_frame(first + soon + later), cooldown_days=182)
    assert events["event_date"].tolist() == [pd.Timestamp("2015-03-03"), pd.Timestamp("2015-10-02")]
    events = detect_clusters(buys_frame(first + soon + later), cooldown_days=30)
    assert len(events) == 3


def test_threshold_is_configurable() -> None:
    buys = buys_frame([(1, 10 + k, "2015-03-02", "2015-03-03") for k in range(4)])
    assert len(detect_clusters(buys, min_insiders=4)) == 1
    assert detect_clusters(buys, min_insiders=5).empty


def test_bad_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        detect_clusters(buys_frame([]), min_insiders=0)


def test_count_labels_uses_the_event_year() -> None:
    buys = buys_frame([(1, 10 + k, "2015-03-02", "2015-03-03") for k in range(3)])
    labels = pd.DataFrame(
        {
            "issuer_cik": [1, 1, 1],
            "insider_cik": [10, 11, 12],
            "year": [2015, 2015, 2014],
            "label": ["opportunistic", "routine", "opportunistic"],
        }
    )
    out = count_labels(detect_clusters(buys), labels)
    assert out["n_opportunistic"].tolist() == [1]
    assert out["n_routine"].tolist() == [1]
