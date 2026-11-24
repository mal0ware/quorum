"""The whole study, from cleaned trades and a price panel to result tables."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quorum import clusters, events, mapping, portfolio, routine
from quorum.prices import Panel
from quorum.stats import block_bootstrap_medians, mean_nw, ols_grouped, ols_nw

Float = NDArray[np.float64]
Bool = NDArray[np.bool_]
Int = NDArray[np.int64]

PERIODS = ("pre", "post", "all")
ALL_LIQUID = "all liquid stocks (the matched benchmark)"
DAYS_PER_MONTH = 21


@dataclass(frozen=True)
class Config:
    """Every choice that shapes the results.

    The first block is the specification of the first run (commit 5a346bf). The
    second block was added afterwards, with those results already known, and
    everything it produces is labelled post hoc in the write-up.
    """

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

    # Added after the first results.
    # Years with roughly level price coverage on both sides of the split date,
    # picked from the coverage table alone (19% to 26% of clusters priced).
    stable_window: tuple[str, str] = ("2007-01-01", "2014-12-31")
    excluded_year: int = 2009
    winsor_tails: tuple[float, ...] = (0.01, 0.05)
    bootstrap_block_months: int = 6
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 0
    # Survivorship bounds: abnormal return given to a missing event that failed,
    # and the share of missing events assumed to have failed inside the window.
    bound_losses: tuple[float, ...] = (-0.30, -0.50, -0.75, -1.00)
    bound_shares: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)


def overlap_lags(horizon_days: int) -> int:
    """Bartlett lags, in months, for returns measured over ``horizon_days``.

    Two events can share return days when their entries are up to
    ``ceil(horizon / 21)`` months apart. Twice that many lags keeps a weight of
    about one half or more on every lag where windows overlap.
    """
    return 2 * -(-horizon_days // DAYS_PER_MONTH)


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


def _time_ids(er: events.EventReturns) -> Int:
    entry = er.events["entry_date"]
    month: Int = (entry.dt.year * 12 + entry.dt.month - 1).to_numpy(dtype=np.int64)
    return month


def mean_tests(x: Float, month: Int, lags: int) -> dict[str, float]:
    """Mean of ``x`` with three standard errors.

    ``se_month_cluster`` clusters on the entry month and is too small when the
    return window is longer than a month. ``se_quarter_cluster`` clusters on the
    entry quarter. ``se_overlap`` sums scores by entry month and lets months up to
    ``lags`` apart be correlated (Bartlett weights); it is the one to read.
    """
    ones = np.ones((len(x), 1))
    by_month = ols_grouped(x, ones, month, 0)
    by_quarter = ols_grouped(x, ones, month // 3, 0)
    overlap = ols_grouped(x, ones, month, lags)
    return {
        "mean_bhar": float(by_month.coef[0]),
        "se_month_cluster": float(by_month.se[0]),
        "t_month_cluster": float(by_month.t[0]),
        "se_quarter_cluster": float(by_quarter.se[0]),
        "t_quarter_cluster": float(by_quarter.t[0]),
        "se_overlap": float(overlap.se[0]),
        "t_overlap": float(overlap.t[0]),
    }


def difference_tests(x: Float, post: Bool, month: Int, lags: int) -> dict[str, float]:
    """Post minus pre from a regression of ``x`` on a constant and a post dummy."""
    design = np.column_stack([np.ones(len(x)), post.astype(np.float64)])
    by_month = ols_grouped(x, design, month, 0)
    by_quarter = ols_grouped(x, design, month // 3, 0)
    overlap = ols_grouped(x, design, month, lags)
    return {
        "pre_mean": float(by_month.coef[0]),
        "post_mean": float(by_month.coef[0] + by_month.coef[1]),
        "post_minus_pre": float(by_month.coef[1]),
        "se_month_cluster": float(by_month.se[1]),
        "t_month_cluster": float(by_month.t[1]),
        "se_quarter_cluster": float(by_quarter.se[1]),
        "t_quarter_cluster": float(by_quarter.t[1]),
        "se_overlap": float(overlap.se[1]),
        "t_overlap": float(overlap.t[1]),
    }


def event_tables(
    er: events.EventReturns, groups: dict[str, Bool], config: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean buy-and-hold abnormal returns by group, period, benchmark and horizon.

    The second table is post minus pre for the same cells.
    """
    month = _time_ids(er)
    periods = _period_masks(er.events["event_date"], pd.Timestamp(config.split_date))
    rows: list[dict[str, object]] = []
    diffs: list[dict[str, object]] = []
    for group, gmask in groups.items():
        for bench in er.benchmarks:
            bhar = er.bhar(bench)
            for h in config.horizons:
                lags = overlap_lags(h)
                for period in PERIODS:
                    mask = gmask & periods[period] & np.isfinite(bhar[:, h])
                    x = bhar[mask, h]
                    n = len(x)
                    rows.append(
                        {
                            "group": group,
                            "period": period,
                            "benchmark": bench,
                            "horizon_days": h,
                            "events": n,
                            **mean_tests(x, month[mask], lags),
                            "overlap_lags_months": lags,
                            "median_bhar": float(np.median(x)) if n else np.nan,
                            "winsorized_mean": events.winsorized_mean(x),
                            "mean_net_of_cost": float(np.mean(x)) - config.round_trip_cost
                            if n
                            else np.nan,
                            "share_positive": float((x > 0).mean()) if n else np.nan,
                            "mean_raw_return": float(np.mean(er.stock[mask, h] - 1.0))
                            if n
                            else np.nan,
                        }
                    )
                mask = gmask & np.isfinite(bhar[:, h])
                diffs.append(
                    {
                        "group": group,
                        "benchmark": bench,
                        "horizon_days": h,
                        **difference_tests(bhar[mask, h], periods["post"][mask], month[mask], lags),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def _winsorize(x: Float, tail: float) -> Float:
    if len(x) == 0:
        return x
    lo, hi = np.quantile(x, [tail, 1.0 - tail])
    out: Float = np.clip(x, lo, hi)
    return out


def robustness_table(er: events.EventReturns, config: Config) -> pd.DataFrame:
    """The headline number under other samples and estimators.

    Every row uses the headline horizon. All of these were added after the first
    results were known. Intervals are 95%: estimate plus or minus 1.96 overlap
    standard errors for means, and block-bootstrap percentiles for medians.
    """
    h = config.headline_horizon
    lags = overlap_lags(h)
    month = _time_ids(er)
    date = er.events["event_date"]
    post = (date >= pd.Timestamp(config.split_date)).to_numpy(dtype=np.bool_)
    lo, hi = (pd.Timestamp(d) for d in config.stable_window)
    samples: dict[str, Bool] = {
        "all priced events": np.ones(len(date), dtype=np.bool_),
        f"excluding {config.excluded_year}": (date.dt.year != config.excluded_year).to_numpy(
            dtype=np.bool_
        ),
        f"{lo.year} to {hi.year} only": ((date >= lo) & (date <= hi)).to_numpy(dtype=np.bool_),
    }
    if "price_reconciled" in er.events:
        samples["reported price reconciled only"] = er.events["price_reconciled"].to_numpy(
            dtype=np.bool_
        )
    rows: list[dict[str, object]] = []

    def row(
        sample: str, estimator: str, bench: str, period: str, n: int, est: float, se: float
    ) -> dict[str, object]:
        return {
            "sample": sample,
            "estimator": estimator,
            "benchmark": bench,
            "period": period,
            "events": n,
            "estimate": est,
            "se_overlap": se,
            "t_overlap": est / se if se > 0 else np.nan,
            "ci_low": est - 1.96 * se,
            "ci_high": est + 1.96 * se,
        }

    def add(sample: str, estimator: str, bench: str, x: Float, p: Bool, m: Int) -> None:
        for period, mask in (("pre", ~p), ("post", p)):
            t = mean_tests(x[mask], m[mask], lags)
            rows.append(
                row(
                    sample,
                    estimator,
                    bench,
                    period,
                    int(mask.sum()),
                    t["mean_bhar"],
                    t["se_overlap"],
                )
            )
        d = difference_tests(x, p, m, lags)
        rows.append(
            row(
                sample,
                estimator,
                bench,
                "post minus pre",
                len(x),
                d["post_minus_pre"],
                d["se_overlap"],
            )
        )

    for bench in (events.UNIVERSE, events.CONTROLS):
        if bench not in er.benchmarks:
            continue
        bhar = er.bhar(bench)[:, h]
        finite = np.isfinite(bhar)
        for sample, smask in samples.items():
            use = smask & finite
            add(sample, "mean", bench, bhar[use], post[use], month[use])
        x, p, m = bhar[finite], post[finite], month[finite]
        for tail in config.winsor_tails:
            clipped = x.copy()
            clipped[~p] = _winsorize(x[~p], tail)
            clipped[p] = _winsorize(x[p], tail)
            name = f"mean, winsorised at {tail:.0%} and {1 - tail:.0%}"
            add("all priced events", name, bench, clipped, p, m)
        boot: dict[str, Float] = {}
        point: dict[str, float] = {}
        for k, (period, mask) in enumerate((("pre", ~p), ("post", p))):
            boot[period] = block_bootstrap_medians(
                x[mask],
                m[mask],
                config.bootstrap_block_months,
                config.bootstrap_draws,
                config.bootstrap_seed + k,
            )
            point[period] = float(np.median(x[mask])) if mask.any() else np.nan
        gap = boot["post"] - boot["pre"]
        for period, est, draws, n in (
            ("pre", point["pre"], boot["pre"], int((~p).sum())),
            ("post", point["post"], boot["post"], int(p.sum())),
            ("post minus pre", point["post"] - point["pre"], gap, len(x)),
        ):
            entry = row("all priced events", "median", bench, period, n, est, np.nan)
            entry["ci_low"] = float(np.nanquantile(draws, 0.025))
            entry["ci_high"] = float(np.nanquantile(draws, 0.975))
            rows.append(entry)
    return pd.DataFrame(rows)


def survivorship_tables(
    found: pd.DataFrame, er: events.EventReturns, config: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """What the mean would be if the clusters with no prices could be added.

    Inputs per period: the priced events and their mean abnormal return, and the
    clusters above the price floor whose issuer is not in the price file. The
    missing clusters are scaled by the share of in-file clusters that pass the
    liquidity screen, on the assumption that the same share of them would have.

    A scenario says that a share ``p`` of the missing events lost ``L`` against
    the benchmark over the window, and that the other missing events earned
    either the priced mean or zero. The all-event mean is then

        M = (n_priced * m_priced + n_missing * (p * L + (1 - p) * m_other))
            / (n_priced + n_missing)

    Returns the inputs, the scenario grid and the break-even shares.
    """
    h = config.headline_horizon
    split = pd.Timestamp(config.split_date)
    bhar = er.bhar(events.UNIVERSE)[:, h]
    above = found[found["median_price"] >= config.min_price]
    inputs: dict[str, dict[str, float]] = {}
    for period in ("pre", "post"):
        f = above[(above["event_date"] < split) == (period == "pre")]
        e = (er.events["event_date"] < split).to_numpy(dtype=np.bool_) == (period == "pre")
        x = bhar[e & np.isfinite(bhar)]
        in_file = int(f["symbol"].notna().sum())
        missing = int(f["symbol"].isna().sum())
        priced = int(e.sum())
        pass_rate = priced / in_file if in_file else np.nan
        inputs[period] = {
            "clusters_above_price_floor": len(f),
            "issuer_in_price_file": in_file,
            "issuer_not_in_price_file": missing,
            "priced": priced,
            "share_of_in_file_clusters_priced": pass_rate,
            "missing_assumed_to_pass_the_screen": missing * pass_rate,
            "priced_with_a_full_window": len(x),
            "mean_bhar_priced": float(x.mean()) if len(x) else np.nan,
        }

    def all_event_mean(period: str, share: float, loss: float, other: str) -> float:
        i = inputs[period]
        n_p, m_p = i["priced_with_a_full_window"], i["mean_bhar_priced"]
        n_u = i["missing_assumed_to_pass_the_screen"]
        m_other = m_p if other == "priced mean" else 0.0
        return float((n_p * m_p + n_u * (share * loss + (1 - share) * m_other)) / (n_p + n_u))

    def crossing(at_zero: float, at_one: float) -> float:
        """Share in [0, 1] where a quantity linear in the share reaches zero."""
        if at_zero <= 0:
            return 0.0
        if at_one >= 0:
            return float("nan")
        return at_zero / (at_zero - at_one)

    grid: list[dict[str, object]] = []
    even: list[dict[str, object]] = []
    for other in ("priced mean", "zero"):
        for loss in config.bound_losses:
            for share in config.bound_shares:
                pre = all_event_mean("pre", share, loss, other)
                post = all_event_mean("post", share, loss, other)
                grid.append(
                    {
                        "other_missing_events_earn": other,
                        "loss_of_a_failed_event": loss,
                        "share_of_missing_events_failed": share,
                        "pre_mean": pre,
                        "post_mean": post,
                        "pre_minus_post": pre - post,
                    }
                )
            ends = {
                period: (
                    all_event_mean(period, 0.0, loss, other),
                    all_event_mean(period, 1.0, loss, other),
                )
                for period in ("pre", "post")
            }
            even.append(
                {
                    "other_missing_events_earn": other,
                    "loss_of_a_failed_event": loss,
                    "share_where_pre_mean_is_zero": crossing(*ends["pre"]),
                    "share_where_post_mean_is_zero": crossing(*ends["post"]),
                    "share_where_pre_equals_post": crossing(
                        ends["pre"][0] - ends["post"][0], ends["pre"][1] - ends["post"][1]
                    ),
                }
            )
    table = pd.DataFrame(inputs).rename_axis("item").reset_index()
    return table, pd.DataFrame(grid), pd.DataFrame(even)


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
            "bhar_controls": er.bhar(events.CONTROLS)[:, h],
        }
    )
    grouped = frame.groupby("year")
    return pd.DataFrame(
        {
            "events": grouped.size(),
            "mean_bhar_universe": grouped["bhar_universe"].mean(),
            "median_bhar_universe": grouped["bhar_universe"].median(),
            "mean_bhar_controls": grouped["bhar_controls"].mean(),
            "median_bhar_controls": grouped["bhar_controls"].median(),
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
            "clusters_issuer_not_in_price_file": found[
                (found["median_price"] >= config.min_price) & found["symbol"].isna()
            ]
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
    # Added after the first results: the cluster portfolio on other sets of months.
    excess = series["cluster of buyers"] - universe
    lo, hi = (pd.Timestamp(d) for d in config.stable_window)
    subsets = {
        "all months": live,
        f"excluding {config.excluded_year}": live
        & np.asarray(month_start.year != config.excluded_year, dtype=np.bool_),
        f"{lo.year} to {hi.year} only": live
        & np.asarray((month_start >= lo) & (month_start <= hi), dtype=np.bool_),
    }
    for label, subset in subsets.items():
        use = subset & np.isfinite(excess)
        for period, mask in (("pre", use & before), ("post", use & ~before)):
            fit = mean_nw(excess[mask], portfolio.NW_LAGS)
            tests.append(
                {
                    "test": f"cluster of buyers: excess over universe, {label}",
                    "period": period,
                    "months": fit.nobs,
                    "estimate": float(fit.coef[0]),
                    "t": float(fit.t[0]),
                }
            )
        if label != "all months":
            dummy = (~before[use]).astype(np.float64)
            fit = ols_nw(
                excess[use], np.column_stack([np.ones(use.sum()), dummy]), portfolio.NW_LAGS
            )
            tests.append(
                {
                    "test": f"cluster of buyers: excess over universe, {label}, post minus pre",
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
    closes: pd.DataFrame | None = None,
    splits: pd.DataFrame | None = None,
) -> Results:
    """Run the study.

    ``trades`` is the output of :func:`quorum.clean.clean_trades`; ``cik_map`` maps
    issuer CIK to a symbol in ``panel``; ``riskfree`` is a monthly return aligned
    with ``panel.months``. With ``closes`` and ``splits`` (see
    :mod:`quorum.mapping`) each event's reported price is checked against the
    price file.
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
        return events.measure(usable, panel, wealth, universe, benchmarks, clusters=found)

    found = find_events(buys, labels, cik_map, config, config.min_insiders, config.window_days)
    er = priced(found)
    if closes is not None and splits is not None:
        er = replace(er, events=er.events.join(mapping.reconcile(er.events, closes, splits)))
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
    study, difference = event_tables(er, groups, config)

    variant_rows = []
    for k, window in ((config.min_insiders, config.window_days), *config.variants):
        v = (
            er
            if (k, window) == (config.min_insiders, config.window_days)
            else priced(find_events(buys, labels, cik_map, config, k, window))
        )
        table, _ = event_tables(v, {"all clusters": np.ones(len(v.events), dtype=np.bool_)}, config)
        table = table[
            table["benchmark"].isin([events.UNIVERSE, events.CONTROLS])
            & (table["horizon_days"] == config.headline_horizon)
            & (table["period"] != "all")
        ]
        variant_rows.append(
            table.assign(min_insiders=k, window_days=window)[
                [
                    "min_insiders",
                    "window_days",
                    "benchmark",
                    "period",
                    "events",
                    "mean_bhar",
                    "t_month_cluster",
                    "t_overlap",
                    "median_bhar",
                ]
            ]
        )

    calendar, tests = calendar_tables(trades, ev, panel, riskfree, config)
    bound_inputs, bound_grid, bound_even = survivorship_tables(found, er, config)
    tables = {
        "sample": sample_table(buys, found, ev, panel, config),
        "event_study": study,
        "event_study_post_minus_pre": difference,
        "robustness": robustness_table(er, config),
        "bhar_paths": path_table(er, config),
        "by_year": year_table(er, config),
        "variants": pd.concat(variant_rows, ignore_index=True),
        "routine_shares": routine_table(trades, config),
        "calendar_time": calendar,
        "calendar_time_tests": tests,
        "survivorship_inputs": bound_inputs,
        "survivorship_bounds": bound_grid,
        "survivorship_breakeven": bound_even,
    }
    if "price_reconciled" in ev:
        tables["mapping_check"] = mapping.summary(ev, pd.Timestamp(config.split_date))
    listing = ev.drop(columns=["insiders"])
    return Results(tables=tables, cluster_events=listing)
