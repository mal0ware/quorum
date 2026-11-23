"""The whole study, from cleaned trades and a price panel to result tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quorum import clusters, events, portfolio, routine
from quorum.prices import Panel
from quorum.stats import mean_nw, ols_nw

Float = NDArray[np.float64]
Bool = NDArray[np.bool_]

PERIODS = ("pre", "post", "all")
ALL_LIQUID = "all liquid stocks (the matched benchmark)"


@dataclass(frozen=True)
class Config:
    """Every choice that shapes the results. All were fixed before the first run."""

    # Cohen, Malloy and Pomorski appeared in the Journal of Finance in June 2012.
    split_date: str = "2012-07-01"
    first_event: str = "2003-07-01"
    last_event: str = "2025-09-30"
    min_insiders: int = 3
    window_days: int = 30
    cooldown_days: int = 182
    horizons: tuple[int, ...] = (21, 63, 126)
    headline_horizon: int = 63
    round_trip_cost: float = 0.01
    min_price: float = 2.0
    cluster_hold_months: int = 3
    market: str = "SPY"
    small_caps: str = "IWM"
    # (min_insiders, window_days) pairs run as robustness checks, besides the main one.
    variants: tuple[tuple[int, int], ...] = ((2, 30), (4, 30), (3, 90), (3, 10))


@dataclass(frozen=True)
class Results:
    """Result tables by name, plus the priced cluster events."""

    tables: dict[str, pd.DataFrame]
    cluster_events: pd.DataFrame


def _period_masks(dates: pd.Series, split: pd.Timestamp) -> dict[str, Bool]:
    before = (dates < split).to_numpy(dtype=np.bool_)
    return {"pre": before, "post": ~before, "all": np.ones(len(dates), dtype=np.bool_)}


def find_events(
    buys: pd.DataFrame,
    labels: pd.DataFrame,
    cik_map: pd.Series,
    config: Config,
    min_insiders: int,
    window_days: int,
) -> pd.DataFrame:
    """Detect clusters and keep those in the sample window, above the price floor, with a ticker."""
    found = clusters.detect_clusters(buys, min_insiders, window_days, config.cooldown_days)
    found = found[
        (found["event_date"] >= pd.Timestamp(config.first_event))
        & (found["event_date"] <= pd.Timestamp(config.last_event))
    ]
    found = clusters.count_labels(found, labels)
    found = found.assign(symbol=found["issuer_cik"].map(cik_map))
    return found.reset_index(drop=True)


def event_table(er: events.EventReturns, groups: dict[str, Bool], config: Config) -> pd.DataFrame:
    """Mean buy-and-hold abnormal returns by group, period, benchmark and horizon."""
    entry = er.events["entry_date"]
    month_id = (entry.dt.year * 12 + entry.dt.month).to_numpy(dtype=np.int64)
    periods = _period_masks(er.events["event_date"], pd.Timestamp(config.split_date))
    rows: list[dict[str, object]] = []
    for group, gmask in groups.items():
        for period in PERIODS:
            mask = gmask & periods[period]
            for bench in er.benchmarks:
                bhar = er.bhar(bench)
                for h in config.horizons:
                    x = bhar[mask, h]
                    mean, se, n = events.clustered_mean(x, month_id[mask])
                    good = x[np.isfinite(x)]
                    rows.append(
                        {
                            "group": group,
                            "period": period,
                            "benchmark": bench,
                            "horizon_days": h,
                            "events": n,
                            "mean_bhar": mean,
                            "se": se,
                            "t": mean / se if n > 1 and se > 0 else np.nan,
                            "median_bhar": float(np.median(good)) if n else np.nan,
                            "winsorized_mean": events.winsorized_mean(x),
                            "mean_net_of_cost": mean - config.round_trip_cost,
                            "share_positive": float((good > 0).mean()) if n else np.nan,
                            "mean_raw_return": float(np.nanmean(er.stock[mask, h] - 1.0))
                            if n
                            else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def difference_table(table: pd.DataFrame) -> pd.DataFrame:
    """Post minus pre for every group, benchmark and horizon."""
    keys = ["group", "benchmark", "horizon_days"]
    pre = table[table["period"] == "pre"].set_index(keys)
    post = table[table["period"] == "post"].set_index(keys)
    out = pd.DataFrame(
        {
            "pre_mean": pre["mean_bhar"],
            "post_mean": post["mean_bhar"],
            "post_minus_pre": post["mean_bhar"] - pre["mean_bhar"],
            "se": np.sqrt(pre["se"] ** 2 + post["se"] ** 2),
        }
    )
    out["t"] = out["post_minus_pre"] / out["se"]
    return out.reset_index()


def path_table(er: events.EventReturns, config: Config) -> pd.DataFrame:
    """Mean abnormal return at every day after entry, for the chart."""
    periods = _period_masks(er.events["event_date"], pd.Timestamp(config.split_date))
    frames = []
    for bench in er.benchmarks:
        bhar = er.bhar(bench)
        for period in ("pre", "post"):
            mean = np.nanmean(bhar[periods[period]], axis=0)
            frames.append(
                pd.DataFrame(
                    {
                        "benchmark": bench,
                        "period": period,
                        "day": np.arange(len(mean)),
                        "mean_bhar": mean,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def year_table(er: events.EventReturns, config: Config) -> pd.DataFrame:
    """Headline-horizon abnormal return by the year of the event."""
    h = config.headline_horizon
    frame = pd.DataFrame(
        {
            "year": er.events["event_date"].dt.year.to_numpy(),
            "bhar_universe": er.bhar(events.UNIVERSE)[:, h],
            "bhar_market": er.bhar("market")[:, h],
        }
    )
    grouped = frame.groupby("year")
    return pd.DataFrame(
        {
            "events": grouped.size(),
            "mean_bhar_universe": grouped["bhar_universe"].mean(),
            "median_bhar_universe": grouped["bhar_universe"].median(),
            "mean_bhar_market": grouped["bhar_market"].mean(),
        }
    ).reset_index()


def sample_table(
    buys: pd.DataFrame, found: pd.DataFrame, priced: pd.DataFrame, panel: Panel, config: Config
) -> pd.DataFrame:
    """Coverage by year: how much of the insider data the price file can follow."""
    in_panel = set(panel.symbols)
    window = (buys["public_date"] >= pd.Timestamp(config.first_event)) & (
        buys["public_date"] <= pd.Timestamp(config.last_event)
    )
    issuers = (
        buys[window]
        .assign(year=buys["public_date"].dt.year)
        .drop_duplicates(["issuer_cik", "year"])
    )
    issuers = issuers.assign(has_prices=issuers["symbol"].isin(in_panel))
    found = found.assign(year=found["event_date"].dt.year)
    priced = priced.assign(year=priced["event_date"].dt.year)
    out = pd.DataFrame(
        {
            "issuers_with_a_purchase": issuers.groupby("year").size(),
            "issuers_with_prices": issuers.groupby("year")["has_prices"].sum(),
            "clusters_found": found.groupby("year").size(),
            "clusters_above_price_floor": found[found["median_price"] >= config.min_price]
            .groupby("year")
            .size(),
            "clusters_priced": priced.groupby("year").size(),
        }
    ).fillna(0)
    out = out.astype("int64")
    out["share_of_issuers_with_prices"] = (
        out["issuers_with_prices"] / out["issuers_with_a_purchase"]
    )
    out["share_of_clusters_priced"] = out["clusters_priced"] / out["clusters_found"]
    out.index.name = "year"
    return out.reset_index()


def routine_table(trades: pd.DataFrame, config: Config) -> pd.DataFrame:
    """How many trades can be classified, and how they split."""
    split = pd.Timestamp(config.split_date)
    frame = trades[
        (trades["public_date"] >= pd.Timestamp(config.first_event))
        & (trades["public_date"] <= pd.Timestamp(config.last_event))
    ]
    period = np.where(frame["public_date"] < split, "pre", "post")
    counts = (
        frame.assign(period=period)
        .groupby(["period", "side", "label"])
        .size()
        .unstack("label", fill_value=0)
    )
    for name in (routine.ROUTINE, routine.OPPORTUNISTIC, routine.UNCLASSIFIED):
        if name not in counts:
            counts[name] = 0
    classified = counts[routine.ROUTINE] + counts[routine.OPPORTUNISTIC]
    total = classified + counts[routine.UNCLASSIFIED]
    out = pd.DataFrame(
        {
            "trades": total,
            "classified": classified,
            "share_classified": classified / total,
            "routine_share_of_classified": counts[routine.ROUTINE] / classified,
        }
    )
    return out.reset_index()


def calendar_tables(
    trades: pd.DataFrame,
    priced_events: pd.DataFrame,
    panel: Panel,
    riskfree: Float,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calendar-time portfolio statistics and the tests built on them."""
    monthly = portfolio.monthly_returns(panel)
    universe = portfolio.universe_return(monthly, panel.liquid)
    market = monthly[:, panel.column(config.market)]
    lookup = {s: i for i, s in enumerate(panel.symbols)}

    def matrix(frame: pd.DataFrame, date_col: str) -> Bool:
        months = panel.months.get_indexer(frame[date_col].dt.to_period("M")).astype(np.int64)
        cols = np.array([lookup.get(s, -1) for s in frame["symbol"]], dtype=np.int64)
        return portfolio.signal_matrix(panel, months, cols)

    window = (trades["public_date"] >= pd.Timestamp(config.first_event)) & (
        trades["public_date"] <= pd.Timestamp(config.last_event)
    )
    tr = trades[window]
    buys, sells = tr[tr["side"] == "P"], tr[tr["side"] == "S"]
    signals: dict[str, tuple[Bool, int]] = {
        "cluster of buyers": (matrix(priced_events, "event_date"), config.cluster_hold_months),
        "any buy": (matrix(buys, "public_date"), 1),
        "opportunistic buy": (
            matrix(buys[buys["label"] == routine.OPPORTUNISTIC], "public_date"),
            1,
        ),
        "routine buy": (matrix(buys[buys["label"] == routine.ROUTINE], "public_date"), 1),
        "opportunistic sell": (
            matrix(sells[sells["label"] == routine.OPPORTUNISTIC], "public_date"),
            1,
        ),
        "routine sell": (matrix(sells[sells["label"] == routine.ROUTINE], "public_date"), 1),
    }

    month_start = panel.months.to_timestamp()
    first_hold = pd.Timestamp(config.first_event) + pd.offsets.MonthBegin(1)
    live = np.asarray(month_start >= first_hold, dtype=np.bool_)
    before = np.asarray(month_start < pd.Timestamp(config.split_date), dtype=np.bool_)
    periods = {"pre": live & before, "post": live & ~before, "all": live}

    series: dict[str, Float] = {}
    rows: list[dict[str, object]] = []
    for name, (signal, hold) in signals.items():
        ret, count = portfolio.calendar_time(signal, monthly, panel.liquid, hold)
        series[name] = ret
        for period, mask in periods.items():
            st = portfolio.summarise(ret, count, universe, market, riskfree, mask)
            rows.append({"portfolio": name, "hold_months": hold, "period": period, **st.__dict__})
    members = (panel.liquid & np.isfinite(monthly)).sum(axis=1).astype(np.float64)
    for period, mask in periods.items():
        st = portfolio.summarise(universe, members, universe, market, riskfree, mask)
        rows.append({"portfolio": ALL_LIQUID, "hold_months": 1, "period": period, **st.__dict__})
    stats = pd.DataFrame(rows)

    tests: list[dict[str, object]] = []

    def spread(label: str, long: str, short: str) -> None:
        for period, mask in periods.items():
            diff = (series[long] - series[short])[mask]
            fit = mean_nw(diff, portfolio.NW_LAGS)
            tests.append(
                {
                    "test": label,
                    "period": period,
                    "months": fit.nobs,
                    "estimate": float(fit.coef[0]),
                    "t": float(fit.t[0]),
                }
            )

    spread("opportunistic buy minus routine buy", "opportunistic buy", "routine buy")
    spread("opportunistic buy minus opportunistic sell", "opportunistic buy", "opportunistic sell")
    spread("routine buy minus routine sell", "routine buy", "routine sell")
    for name in ("cluster of buyers", "any buy", "opportunistic buy"):
        excess = series[name] - universe
        use = live & np.isfinite(excess)
        post = (~before[use]).astype(np.float64)
        fit = ols_nw(excess[use], np.column_stack([np.ones(use.sum()), post]), portfolio.NW_LAGS)
        tests.append(
            {
                "test": f"{name}: excess over universe, post minus pre",
                "period": "all",
                "months": fit.nobs,
                "estimate": float(fit.coef[1]),
                "t": float(fit.t[1]),
            }
        )
    return stats, pd.DataFrame(tests)


def run(
    trades: pd.DataFrame,
    panel: Panel,
    cik_map: pd.Series,
    riskfree: Float,
    config: Config | None = None,
) -> Results:
    """Run the study.

    ``trades`` is the output of :func:`quorum.clean.clean_trades`; ``cik_map`` maps
    issuer CIK to a symbol in ``panel``; ``riskfree`` is a monthly return aligned
    with ``panel.months``.
    """
    config = config or Config()
    labels = routine.classify_insiders(trades)
    trades = trades.assign(
        label=routine.label_trades(trades, labels), symbol=trades["issuer_cik"].map(cik_map)
    )
    buys = trades[trades["side"] == "P"]

    horizon = max(config.horizons)
    wealth = events.wealth_index(panel.ret)
    universe = events.universe_table(wealth, panel, horizon)
    benchmarks = {"market": config.market, "small_caps": config.small_caps}

    def priced(found: pd.DataFrame) -> events.EventReturns:
        usable = found[(found["median_price"] >= config.min_price) & found["symbol"].notna()]
        return events.measure(usable, panel, wealth, universe, benchmarks)

    found = find_events(buys, labels, cik_map, config, config.min_insiders, config.window_days)
    er = priced(found)
    ev = er.events
    groups: dict[str, Bool] = {
        "all clusters": np.ones(len(ev), dtype=np.bool_),
        "with an opportunistic buyer": (ev["n_opportunistic"] > 0).to_numpy(dtype=np.bool_),
        "routine buyers only": ((ev["n_opportunistic"] == 0) & (ev["n_routine"] > 0)).to_numpy(
            dtype=np.bool_
        ),
        "no classified buyer": ((ev["n_opportunistic"] == 0) & (ev["n_routine"] == 0)).to_numpy(
            dtype=np.bool_
        ),
    }
    study = event_table(er, groups, config)

    variant_rows = []
    for k, window in ((config.min_insiders, config.window_days), *config.variants):
        v = (
            er
            if (k, window) == (config.min_insiders, config.window_days)
            else priced(find_events(buys, labels, cik_map, config, k, window))
        )
        table = event_table(v, {"all clusters": np.ones(len(v.events), dtype=np.bool_)}, config)
        table = table[
            (table["benchmark"] == events.UNIVERSE)
            & (table["horizon_days"] == config.headline_horizon)
            & (table["period"] != "all")
        ]
        variant_rows.append(
            table.assign(min_insiders=k, window_days=window)[
                ["min_insiders", "window_days", "period", "events", "mean_bhar", "t", "median_bhar"]
            ]
        )

    calendar, tests = calendar_tables(trades, ev, panel, riskfree, config)
    tables = {
        "sample": sample_table(buys, found, ev, panel, config),
        "event_study": study,
        "event_study_post_minus_pre": difference_table(study),
        "bhar_paths": path_table(er, config),
        "by_year": year_table(er, config),
        "variants": pd.concat(variant_rows, ignore_index=True),
        "routine_shares": routine_table(trades, config),
        "calendar_time": calendar,
        "calendar_time_tests": tests,
    }
    listing = ev.drop(columns=["insiders"])
    return Results(tables=tables, cluster_events=listing)
