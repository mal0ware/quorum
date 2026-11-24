"""The whole pipeline on the synthetic data set, where the answer is known."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from quorum import pipeline, synthetic
from quorum.dataset import Inputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def results(synthetic_inputs: Inputs) -> pipeline.Results:
    i = synthetic_inputs
    return pipeline.run(i.trades, i.panel, i.cik_map, i.riskfree, closes=i.closes, splits=i.splits)


def headline(results: pipeline.Results, period: str, benchmark: str = "universe") -> pd.Series:
    table = results.tables["event_study"]
    row = table[
        (table["group"] == "all clusters")
        & (table["period"] == period)
        & (table["benchmark"] == benchmark)
        & (table["horizon_days"] == 63)
    ]
    assert len(row) == 1
    return row.iloc[0]


def test_planted_drift_is_found_before_the_split_and_not_after(results: pipeline.Results) -> None:
    pre, post = headline(results, "pre"), headline(results, "post")
    # Planted: 0.15% a day for 63 days from two days after the third filing, so a
    # pure cluster earns about 1.0015**62 - 1 = 9.7%. Chance clusters dilute the mean.
    planted = (1 + synthetic.DRIFT_PER_DAY) ** 62 - 1
    assert 0.6 * planted < pre["mean_bhar"] < 1.05 * planted
    assert pre["t_overlap"] > 4
    assert abs(post["mean_bhar"]) < 0.03
    assert abs(post["t_overlap"]) < 2.5
    assert pre["events"] > 100 and post["events"] > 100

    diff = results.tables["event_study_post_minus_pre"]
    row = diff[
        (diff["group"] == "all clusters")
        & (diff["benchmark"] == "universe")
        & (diff["horizon_days"] == 63)
    ].iloc[0]
    assert row["post_minus_pre"] == pytest.approx(post["mean_bhar"] - pre["mean_bhar"])
    assert row["t_overlap"] < -3


def test_the_effect_shows_against_every_benchmark(results: pipeline.Results) -> None:
    for benchmark in ("market", "small_caps"):
        assert headline(results, "pre", benchmark)["mean_bhar"] > 0.04


def test_cost_is_subtracted_once(results: pipeline.Results) -> None:
    row = headline(results, "pre")
    assert row["mean_net_of_cost"] == pytest.approx(
        row["mean_bhar"] - pipeline.Config().round_trip_cost
    )


def test_calendar_time_portfolio_agrees_in_sign(results: pipeline.Results) -> None:
    table = results.tables["calendar_time"]
    cluster = table[table["portfolio"] == "cluster of buyers"].set_index("period")
    assert cluster.loc["pre", "excess_universe"] > 0.01
    assert cluster.loc["pre", "excess_universe_t"] > 3
    assert abs(cluster.loc["post", "excess_universe_t"]) < 2.5
    benchmark = table[table["portfolio"] == pipeline.ALL_LIQUID].set_index("period")
    assert benchmark.loc["all", "excess_universe"] == 0.0


def test_routine_insiders_are_found(results: pipeline.Results) -> None:
    shares = results.tables["routine_shares"]
    buys = shares[(shares["side"] == "P") & (shares["period"] == "post")].iloc[0]
    # Two insiders per stock buy every March. A few random traders also reach three
    # straight years, so the share is high but not one.
    assert buys["routine_share_of_classified"] > 0.8
    calendar = results.tables["calendar_time"]
    routine = calendar[
        (calendar["portfolio"] == "routine buy") & (calendar["period"] == "all")
    ].iloc[0]
    assert routine["months"] > 5


def test_illiquid_stocks_are_never_events(
    results: pipeline.Results, synthetic_inputs: Inputs
) -> None:
    stocks = sorted(s for s in synthetic_inputs.panel.symbols if s.startswith("S") and s != "SPY")
    thin = set(stocks[-synthetic.N_ILLIQUID :])
    assert len(thin) == synthetic.N_ILLIQUID
    assert thin & set(results.cluster_events["symbol"]) == set()
    assert set(stocks[:5]) & set(results.cluster_events["symbol"])


def test_every_event_enters_after_it_is_public(results: pipeline.Results) -> None:
    events = results.cluster_events
    assert (events["entry_date"] > events["event_date"]).all()
    assert (events["n_insiders"] >= 3).all()
    assert (events["median_price"] >= pipeline.Config().min_price).all()


def test_sample_table_adds_up(results: pipeline.Results) -> None:
    sample = results.tables["sample"]
    assert sample["clusters_priced"].sum() == len(results.cluster_events)
    assert (sample["clusters_priced"] <= sample["clusters_found"]).all()


def test_variants_include_the_main_specification(results: pipeline.Results) -> None:
    variants = results.tables["variants"]
    main = variants[
        (variants["min_insiders"] == 3)
        & (variants["window_days"] == 30)
        & (variants["benchmark"] == "universe")
    ]
    assert main.set_index("period").loc["pre", "mean_bhar"] == pytest.approx(
        headline(results, "pre")["mean_bhar"]
    )
    assert len(variants) == 2 * 2 * (1 + len(pipeline.Config().variants))


def test_running_twice_gives_identical_tables(results: pipeline.Results) -> None:
    i = synthetic.make(seed=0)
    again = pipeline.run(i.trades, i.panel, i.cik_map, i.riskfree, closes=i.closes, splits=i.splits)
    assert set(again.tables) == set(results.tables)
    for name, table in results.tables.items():
        pd.testing.assert_frame_equal(table, again.tables[name], check_exact=True)
    pd.testing.assert_frame_equal(results.cluster_events, again.cluster_events)


def test_reproduce_fast_writes_the_same_files_twice(tmp_path: Path) -> None:
    outputs = []
    for name in ("a", "b"):
        out = tmp_path / name
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "reproduce.py"), "--fast", "--out", str(out)],
            check=True,
            capture_output=True,
        )
        outputs.append(out)
    names = sorted(p.name for p in outputs[0].iterdir())
    assert "fig1_estimates.png" in names and "event_study.csv" in names
    assert names == sorted(p.name for p in outputs[1].iterdir())
    for name in names:
        if name.endswith(".csv"):
            assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes(), name
