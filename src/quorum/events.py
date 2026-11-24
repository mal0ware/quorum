"""Event-study returns after a filing becomes public.

Timing. A filing accepted on calendar day ``d`` is treated as public on ``d``.
The position is opened at the close of the first trading day strictly after
``d`` (the entry day). Nothing on or before ``d`` enters any return.

Returns. With ``V_i(h)`` the value after ``h`` trading days of one unit put into
stock ``i`` at the entry close,

    V_i(h)    = prod_{s=1..h} (1 + r_{i, entry+s})
    BHAR_i(h) = V_i(h) - V_b(h)

where ``V_b`` is the same quantity for a benchmark. This is the buy-and-hold
abnormal return of Barber and Lyon (1997, Journal of Financial Economics 43,
equation 3). Days on which a stock has no return count as zero, so a stock that
leaves the price file is carried at its last price.

The matched benchmark is an equal-weight buy-and-hold portfolio of every liquid
stock in the panel, formed at the same entry close:

    V_u(h) = mean_j V_j(h)

It is drawn from the same price file as the event stocks, so it carries the same
survivorship as they do.

The matched controls are a tighter version of the same idea. For each event,
take the stocks that are liquid in the entry month, have a return on the entry
day, and had no cluster of their own within :data:`CONTROL_EXCLUSION_DAYS`
calendar days either side of the event. Of those, the :data:`N_CONTROLS` nearest
in log dollar volume (the median daily dollar volume of the month before, the
only size measure in the data) form an equal-weight buy-and-hold portfolio.
Every control comes from the same price file, so it met the same condition as
the event stock: it was still listed when the file was built.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quorum.prices import Panel

Float = NDArray[np.float64]

UNIVERSE = "universe"
CONTROLS = "matched_controls"
N_CONTROLS = 5
CONTROL_EXCLUSION_DAYS = 182


def entry_index(dates: pd.DatetimeIndex, public_dates: pd.Series) -> NDArray[np.int64]:
    """Index of the first trading day strictly after each public date.

    Equals ``len(dates)`` when there is no such day.
    """
    days = dates.to_numpy(dtype="datetime64[D]")
    public = public_dates.to_numpy(dtype="datetime64[D]")
    out: NDArray[np.int64] = np.searchsorted(days, public, side="right").astype(np.int64)
    return out


def wealth_index(ret: NDArray[np.float32]) -> Float:
    """Running product of (1 + r) down each column, with missing returns as zero."""
    filled: Float = np.nan_to_num(ret.astype(np.float64), nan=0.0)
    out: Float = np.cumprod(1.0 + filled, axis=0, dtype=np.float64)
    return out


def value_paths(wealth: Float, entry: NDArray[np.int64], horizon: int) -> Float:
    """``V(h)`` for h = 0..horizon from one wealth-index column.

    Entries past the end of the data are NaN.
    """
    n_days = len(wealth)
    idx = entry[:, None] + np.arange(horizon + 1)[None, :]
    inside = idx < n_days
    base = wealth[np.minimum(entry, n_days - 1)]
    out: Float = wealth[np.where(inside, idx, 0)] / base[:, None]
    out[~inside] = np.nan
    return out


def universe_table(wealth: Float, panel: Panel, horizon: int) -> Float:
    """``V_u(h)`` from every trading day: a [days, horizon + 1] table.

    Row ``t`` is the equal-weight buy-and-hold value of the stocks that are
    liquid in the month of day ``t``, bought at the close of ``t``.
    """
    n_days = wealth.shape[0]
    out = np.full((n_days, horizon + 1), np.nan)
    members_of_month = [np.flatnonzero(row) for row in panel.liquid]
    for day in range(n_days):
        members = members_of_month[panel.month_of_day[day]]
        if len(members) == 0:
            continue
        last = min(day + horizon, n_days - 1)
        block = wealth[day : last + 1][:, members]
        out[day, : last - day + 1] = (block / block[0]).mean(axis=1)
    return out


@dataclass(frozen=True)
class EventReturns:
    """Value paths for a set of events. Rows are events, columns are h = 0..horizon."""

    events: pd.DataFrame
    stock: Float
    benchmarks: dict[str, Float]

    def bhar(self, benchmark: str) -> Float:
        """Buy-and-hold abnormal return paths against one benchmark."""
        return self.stock - self.benchmarks[benchmark]


def pick_controls(
    panel: Panel,
    col: NDArray[np.int64],
    entry: NDArray[np.int64],
    event_days: NDArray[np.datetime64],
    cluster_days: dict[int, NDArray[np.datetime64]],
) -> NDArray[np.int64]:
    """Columns of the matched controls for each event: an [events, N_CONTROLS] table.

    ``cluster_days`` maps a panel column to the sorted days on which that stock
    had a cluster. Missing controls are -1. Ties in distance go to the lower column.
    """
    out = np.full((len(col), N_CONTROLS), -1, dtype=np.int64)
    span = np.timedelta64(CONTROL_EXCLUSION_DAYS, "D")
    with np.errstate(invalid="ignore", divide="ignore"):
        log_dv = np.log(panel.dollar_volume)
    for i in range(len(col)):
        month = panel.month_of_day[entry[i]]
        target = log_dv[month, col[i]]
        ok = panel.liquid[month] & np.isfinite(panel.ret[entry[i]]) & np.isfinite(log_dv[month])
        ok[col[i]] = False
        candidates = np.flatnonzero(ok)
        if not np.isfinite(target) or len(candidates) == 0:
            continue
        order = candidates[np.argsort(np.abs(log_dv[month, candidates] - target), kind="stable")]
        found = 0
        for c in order:
            days = cluster_days.get(int(c))
            if days is not None:
                k = int(np.searchsorted(days, event_days[i] - span, side="left"))
                if k < len(days) and days[k] <= event_days[i] + span:
                    continue
            out[i, found] = c
            found += 1
            if found == N_CONTROLS:
                break
    return out


def control_paths(
    wealth: Float, controls: NDArray[np.int64], entry: NDArray[np.int64], horizon: int
) -> Float:
    """Equal-weight buy-and-hold value of each event's controls, h = 0..horizon.

    An event with no controls gets NaN. Days past the end of the data are NaN.
    """
    n_days = len(wealth)
    idx = entry[:, None] + np.arange(horizon + 1)[None, :]
    inside = idx < n_days
    safe = np.where(inside, idx, 0)
    total = np.zeros(idx.shape)
    count = np.zeros(len(entry))
    for j in range(controls.shape[1]):
        c = controls[:, j]
        have = c >= 0
        cs = np.maximum(c, 0)
        path = wealth[safe, cs[:, None]] / wealth[np.minimum(entry, n_days - 1), cs][:, None]
        total += np.where(have[:, None], path, 0.0)
        count += have
    with np.errstate(invalid="ignore", divide="ignore"):
        out: Float = total / count[:, None]
    out[~inside] = np.nan
    return out


def measure(
    events: pd.DataFrame,
    panel: Panel,
    wealth: Float,
    universe: Float,
    benchmark_symbols: dict[str, str],
    clusters: pd.DataFrame | None = None,
) -> EventReturns:
    """Measure value paths for the events that can be priced.

    ``events`` needs ``symbol`` and ``event_date``. An event is kept if its stock
    is in the panel, is liquid in the entry month, and has a return on the entry
    day (so the entry close exists). The returned ``events`` holds the kept rows
    with ``entry_date`` added. ``universe`` comes from :func:`universe_table` and
    sets the horizon.

    ``clusters`` lists every detected cluster (``symbol`` and ``event_date``),
    priced or not. When given, the matched-control benchmark is added and no
    control is a stock with a cluster of its own near the event.
    """
    horizon = universe.shape[1] - 1
    lookup = {s: i for i, s in enumerate(panel.symbols)}
    col = np.array([lookup.get(s, -1) for s in events["symbol"]], dtype=np.int64)
    entry = entry_index(panel.dates, events["event_date"])
    n_days = len(panel.dates)
    e_safe = np.minimum(entry, n_days - 1)
    c_safe = np.maximum(col, 0)
    ok = (col >= 0) & (entry < n_days)
    ok &= panel.liquid[panel.month_of_day[e_safe], c_safe]
    ok &= np.isfinite(panel.ret[e_safe, c_safe])

    kept = events[ok].reset_index(drop=True)
    col, entry = col[ok], entry[ok]
    kept["entry_date"] = panel.dates[entry]

    stock = np.full((len(kept), horizon + 1), np.nan)
    for c in np.unique(col):
        rows = np.flatnonzero(col == c)
        stock[rows] = value_paths(wealth[:, c], entry[rows], horizon)

    benchmarks = {UNIVERSE: universe[entry]}
    if clusters is not None:
        cluster_days: dict[int, NDArray[np.datetime64]] = {}
        known = clusters[clusters["symbol"].isin(lookup)]
        for symbol, days in known.groupby("symbol", sort=False)["event_date"]:
            cluster_days[lookup[str(symbol)]] = np.sort(days.to_numpy(dtype="datetime64[D]"))
        event_days = kept["event_date"].to_numpy(dtype="datetime64[D]")
        controls = pick_controls(panel, col, entry, event_days, cluster_days)
        benchmarks[CONTROLS] = control_paths(wealth, controls, entry, horizon)
        kept["n_controls"] = (controls >= 0).sum(axis=1)
    for name, symbol in benchmark_symbols.items():
        benchmarks[name] = value_paths(wealth[:, lookup[symbol]], entry, horizon)
    return EventReturns(events=kept, stock=stock, benchmarks=benchmarks)


def winsorized_mean(x: Float, tail: float = 0.01) -> float:
    """Mean after pulling each tail in to its ``tail`` quantile."""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    lo, hi = np.quantile(x, [tail, 1.0 - tail])
    return float(np.clip(x, lo, hi).mean())
