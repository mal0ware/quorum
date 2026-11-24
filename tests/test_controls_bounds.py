"""Matched controls, the survivorship bounds and the robustness table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quorum import events, pipeline
from quorum.dataset import Inputs
from quorum.prices import build_panel

N_DAYS = 90
VOLUMES = {"EVT": 1.0e6, "NEAR1": 1.1e6, "NEAR2": 0.9e6, "BUSY": 1.0e6, "FAR": 5.0e7, "TINY": 1.0}


def volume_panel(seed: int = 0):  # type: ignore[no-untyped-def]
    """Six stocks that differ in dollar volume. TINY is never liquid."""
    rng = np.random.default_rng(seed)
    calendar = pd.bdate_range("2010-01-04", periods=N_DAYS + 1, name="date")
    frames = []
    returns = {}
    for name, volume in VOLUMES.items():
        r = rng.normal(0.0, 0.01, N_DAYS)
        returns[name] = r
        close = 50.0 * np.concatenate([[1.0], np.cumprod(1.0 + r)])
        frames.append(
            pd.DataFrame({"symbol": name, "date": calendar, "close": close, "volume": volume / 50})
        )
    dv = pd.DataFrame({"symbol": [], "date": pd.to_datetime([]), "amount": []})
    return build_panel(pd.concat(frames, ignore_index=True), dv, calendar, 100_000.0), returns


def measure(panel, clusters: pd.DataFrame, horizon: int = 10):  # type: ignore[no-untyped-def]
    wealth = events.wealth_index(panel.ret)
    universe = events.universe_table(wealth, panel, horizon)
    table = pd.DataFrame({"symbol": ["EVT"], "event_date": [pd.Timestamp("2010-03-01")]})
    return events.measure(table, panel, wealth, universe, {}, clusters=clusters)


def test_dollar_volume_is_last_months_median_and_only_for_liquid_stocks() -> None:
    panel, _ = volume_panel()
    tiny = panel.column("TINY")
    evt = panel.column("EVT")
    assert not panel.liquid[:, tiny].any()
    assert np.isnan(panel.dollar_volume[:, tiny]).all()
    assert np.isnan(panel.dollar_volume[0, evt])  # no month before the first
    assert np.isfinite(panel.dollar_volume[1:, evt]).all()
    np.testing.assert_array_equal(np.isfinite(panel.dollar_volume), panel.liquid)
    assert panel.dollar_volume[2, panel.column("FAR")] > 10 * panel.dollar_volume[2, evt]


def test_controls_are_the_nearest_in_volume_without_a_cluster_of_their_own() -> None:
    panel, returns = volume_panel()
    clusters = pd.DataFrame(
        {
            "symbol": ["EVT", "BUSY", "UNLISTED"],
            "event_date": pd.to_datetime(["2010-03-01", "2010-02-10", "2010-03-01"]),
        }
    )
    er = measure(panel, clusters)
    assert er.events["n_controls"].iloc[0] == 3  # NEAR1, NEAR2, FAR; not itself, BUSY or TINY
    entry = panel.dates.get_loc(er.events["entry_date"].iloc[0])
    paths = [
        np.concatenate([[1.0], np.cumprod(1.0 + returns[name][entry : entry + 10])])
        for name in ("NEAR1", "NEAR2", "FAR")
    ]
    np.testing.assert_allclose(er.benchmarks[events.CONTROLS][0], np.mean(paths, axis=0), rtol=1e-5)
    np.testing.assert_allclose(
        er.bhar(events.CONTROLS)[0], er.stock[0] - np.mean(paths, axis=0), rtol=1e-5, atol=1e-7
    )


def test_a_cluster_outside_the_exclusion_span_does_not_bar_a_control() -> None:
    panel, _ = volume_panel()
    far_away = pd.DataFrame(
        {
            "symbol": ["EVT", "BUSY"],
            "event_date": pd.to_datetime(["2010-03-01", "2009-06-01"]),
        }
    )
    assert measure(panel, far_away).events["n_controls"].iloc[0] == 4


def test_nearest_controls_are_chosen_first() -> None:
    panel, _ = volume_panel()
    entry = np.array([panel.dates.get_loc(pd.Timestamp("2010-03-02"))], dtype=np.int64)
    col = np.array([panel.column("EVT")], dtype=np.int64)
    day = np.array(["2010-03-01"], dtype="datetime64[D]")
    picked = events.pick_controls(panel, col, entry, day, {})[0]
    names = [panel.symbols[c] for c in picked if c >= 0]
    assert names[0] == "BUSY"  # same volume as the event stock
    assert set(names[1:3]) == {"NEAR1", "NEAR2"}
    assert names[3] == "FAR"
    assert "EVT" not in names and "TINY" not in names


def test_without_a_cluster_list_there_is_no_control_benchmark() -> None:
    panel, _ = volume_panel()
    wealth = events.wealth_index(panel.ret)
    universe = events.universe_table(wealth, panel, 5)
    table = pd.DataFrame({"symbol": ["EVT"], "event_date": [pd.Timestamp("2010-03-01")]})
    er = events.measure(table, panel, wealth, universe, {})
    assert events.CONTROLS not in er.benchmarks


@pytest.fixture(scope="module")
def results(synthetic_inputs: Inputs) -> pipeline.Results:
    i = synthetic_inputs
    return pipeline.run(i.trades, i.panel, i.cik_map, i.riskfree, closes=i.closes, splits=i.splits)


def test_the_planted_effect_shows_against_matched_controls(results: pipeline.Results) -> None:
    table = results.tables["event_study"]
    row = table[
        (table["group"] == "all clusters")
        & (table["benchmark"] == events.CONTROLS)
        & (table["horizon_days"] == 63)
    ].set_index("period")
    assert row.loc["pre", "mean_bhar"] > 0.04
    assert row.loc["pre", "t_overlap"] > 4
    assert abs(row.loc["post", "t_overlap"]) < 2.5
    assert (results.cluster_events["n_controls"] == events.N_CONTROLS).all()


def test_three_standard_errors_are_reported_for_every_cell(results: pipeline.Results) -> None:
    table = results.tables["event_study"]
    for name in ("se_month_cluster", "se_quarter_cluster", "se_overlap"):
        assert (table.loc[table["events"] > 50, name] > 0).all()
    assert set(table["overlap_lags_months"]) == {2, 6, 12}


def test_robustness_rows(results: pipeline.Results) -> None:
    table = results.tables["robustness"]
    assert set(table["period"]) == {"pre", "post", "post minus pre"}
    assert {"all priced events", "excluding 2009", "2007 to 2014 only"} <= set(table["sample"])
    assert "reported price reconciled only" in set(table["sample"])
    assert {"mean", "median"} <= set(table["estimator"])
    base = table[(table["benchmark"] == "universe") & (table["period"] == "pre")]
    whole = base[(base["sample"] == "all priced events") & (base["estimator"] == "mean")].iloc[0]
    study = results.tables["event_study"]
    head = study[
        (study["group"] == "all clusters")
        & (study["benchmark"] == "universe")
        & (study["horizon_days"] == 63)
        & (study["period"] == "pre")
    ].iloc[0]
    assert whole["estimate"] == pytest.approx(head["mean_bhar"])
    assert whole["se_overlap"] == pytest.approx(head["se_overlap"])
    without = base[base["sample"] == "excluding 2009"].iloc[0]
    assert without["events"] < whole["events"]
    median = base[base["estimator"] == "median"].iloc[0]
    assert median["estimate"] == pytest.approx(head["median_bhar"])
    assert median["ci_low"] <= median["estimate"] <= median["ci_high"]
    assert (table["ci_low"] <= table["ci_high"]).all()


def test_synthetic_reported_prices_reconcile(results: pipeline.Results) -> None:
    check = results.tables["mapping_check"].set_index("sample")
    assert check.loc["all", "share_reconciled"] > 0.99
    assert results.cluster_events["price_reconciled"].all()


def bounds_fixture() -> tuple[pd.DataFrame, events.EventReturns]:
    """Before the split: 10 priced events at +10%, 30 clusters with no prices."""
    dates = pd.to_datetime(["2010-01-04"] * 10 + ["2015-01-05"] * 10)
    kept = pd.DataFrame({"event_date": dates})
    stock = np.ones((20, 64))
    stock[:10, 63] = 1.10
    stock[10:, 63] = 1.02
    er = events.EventReturns(
        events=kept, stock=stock, benchmarks={events.UNIVERSE: np.ones((20, 64))}
    )
    found = pd.DataFrame(
        {
            "event_date": pd.to_datetime(
                ["2010-01-04"] * 50 + ["2015-01-05"] * 20 + ["2010-01-04"] * 5
            ),
            "median_price": [10.0] * 70 + [1.0] * 5,  # the last five are under the price floor
            "symbol": ["A"] * 20 + [None] * 30 + ["A"] * 10 + [None] * 10 + [None] * 5,
        }
    )
    return found, er


def test_survivorship_inputs_and_bounds_follow_the_formula() -> None:
    found, er = bounds_fixture()
    inputs, grid, even = pipeline.survivorship_tables(found, er, pipeline.Config())
    i = inputs.set_index("item")
    assert i.loc["clusters_above_price_floor", "pre"] == 50
    assert i.loc["issuer_not_in_price_file", "pre"] == 30
    assert i.loc["share_of_in_file_clusters_priced", "pre"] == pytest.approx(0.5)
    assert i.loc["missing_assumed_to_pass_the_screen", "pre"] == pytest.approx(15.0)
    assert i.loc["mean_bhar_priced", "pre"] == pytest.approx(0.10)
    assert i.loc["mean_bhar_priced", "post"] == pytest.approx(0.02)

    def cell(other: str, loss: float, share: float) -> pd.Series:
        return grid[
            (grid["other_missing_events_earn"] == other)
            & (grid["loss_of_a_failed_event"] == loss)
            & (grid["share_of_missing_events_failed"] == share)
        ].iloc[0]

    # No failures and the others earn the priced mean: nothing changes.
    assert cell("priced mean", -1.0, 0.0)["pre_mean"] == pytest.approx(0.10)
    # 10 priced at +10%, 15 missing: 20% lose everything, the rest earn 10%.
    expected = (10 * 0.10 + 15 * (0.2 * -1.0 + 0.8 * 0.10)) / 25
    assert cell("priced mean", -1.0, 0.2)["pre_mean"] == pytest.approx(expected)
    # The others earn zero.
    assert cell("zero", -0.5, 0.1)["pre_mean"] == pytest.approx((1.0 + 15 * 0.1 * -0.5) / 25)

    row = even[
        (even["other_missing_events_earn"] == "zero") & (even["loss_of_a_failed_event"] == -1.0)
    ].iloc[0]
    # 10 * 0.10 = 15 * p * 1.0  ->  p = 1/15
    assert row["share_where_pre_mean_is_zero"] == pytest.approx(1 / 15)
    share = row["share_where_pre_equals_post"]
    pre = (10 * 0.10 - 15 * share) / 25
    post = (10 * 0.02 - 10 * share) / 20  # after: 10 priced, 10 missing, all pass the screen
    assert pre == pytest.approx(post)


def test_break_even_is_blank_when_no_share_is_enough() -> None:
    found, er = bounds_fixture()
    found = found[found["symbol"].notna()]  # nothing is missing
    _, _, even = pipeline.survivorship_tables(found, er, pipeline.Config())
    assert even["share_where_pre_mean_is_zero"].isna().all()


def test_extra_calendar_rows_cover_the_other_month_sets(results: pipeline.Results) -> None:
    tests = results.tables["calendar_time_tests"]
    names = set(tests["test"])
    assert "cluster of buyers: excess over universe, excluding 2009" in names
    assert "cluster of buyers: excess over universe, 2007 to 2014 only, post minus pre" in names
    every = tests[tests["test"] == "cluster of buyers: excess over universe, all months"]
    calendar = results.tables["calendar_time"]
    cluster = calendar[calendar["portfolio"] == "cluster of buyers"].set_index("period")
    for _, row in every.iterrows():
        assert row["estimate"] == pytest.approx(cluster.loc[row["period"], "excess_universe"])
        assert row["t"] == pytest.approx(cluster.loc[row["period"], "excess_universe_t"])
