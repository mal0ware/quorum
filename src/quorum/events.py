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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quorum.prices import Panel

Float = NDArray[np.float64]

UNIVERSE = "universe"


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


def measure(
    events: pd.DataFrame,
    panel: Panel,
    wealth: Float,
    universe: Float,
    benchmark_symbols: dict[str, str],
) -> EventReturns:
    """Measure value paths for the events that can be priced.

    ``events`` needs ``symbol`` and ``event_date``. An event is kept if its stock
    is in the panel, is liquid in the entry month, and has a return on the entry
    day (so the entry close exists). The returned ``events`` holds the kept rows
    with ``entry_date`` added. ``universe`` comes from :func:`universe_table` and
    sets the horizon.
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
    for name, symbol in benchmark_symbols.items():
        benchmarks[name] = value_paths(wealth[:, lookup[symbol]], entry, horizon)
    return EventReturns(events=kept, stock=stock, benchmarks=benchmarks)


def clustered_mean(x: Float, groups: NDArray[np.int64]) -> tuple[float, float, int]:
    """Mean of ``x`` with a standard error clustered on ``groups``.

    ``se = sqrt(sum_g (sum_{i in g} (x_i - mean))^2) / n``: the cluster-robust
    sandwich for a regression on a constant, with no small-sample correction.
    """
    good = np.isfinite(x)
    x, groups = x[good], groups[good]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(x.mean())
    sums = pd.Series(x - mean).groupby(groups).sum().to_numpy(dtype=np.float64)
    se = float(np.sqrt((sums**2).sum()) / n)
    return mean, se, n


def winsorized_mean(x: Float, tail: float = 0.01) -> float:
    """Mean after pulling each tail in to its ``tail`` quantile."""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    lo, hi = np.quantile(x, [tail, 1.0 - tail])
    return float(np.clip(x, lo, hi).mean())
