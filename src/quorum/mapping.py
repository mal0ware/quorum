"""Check each event's join to the price file against the price the insiders reported.

Filings are joined to prices by SEC CIK. Two numbers from different sources
should then agree: the median price the insiders reported paying, and the close
in the price file around the same days.

The price file is split-adjusted, so a close from before a later split is not
the price that traded. The split table undoes that. A split dated ``s`` with
factor ``f`` (``f`` new shares per old share) means every adjusted close before
``s`` was divided by ``f``, so

    close_as_traded(t) = close_adjusted(t) * prod_{splits s > t} f_s

A cluster's purchases were traded in the :data:`WINDOW_DAYS` calendar days up to
the event date. An event is **reconciled** when the median reported price lies
between the lowest and highest as-traded close of that window, widened by
:data:`TOLERANCE` on each side for intraday prices. Anything else is a missing
split, a private purchase away from the market price, a bad filing, or a wrong
join, and the data cannot tell these apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_DAYS = 30
TOLERANCE = 0.10
NEAR = 0.25

COLUMNS = ["close_as_traded", "price_ratio", "price_ratio_adjusted", "price_reconciled"]


def as_traded(dates: np.ndarray, closes: np.ndarray, split_dates: np.ndarray, factors: np.ndarray):  # type: ignore[no-untyped-def]
    """Undo the split adjustment of one stock's closes. Inputs are sorted by date."""
    if len(split_dates) == 0:
        return closes.astype(np.float64)
    # tail[k] is the product of the factors of splits k, k+1, ...; tail[len] is 1.
    tail = np.concatenate([np.cumprod(factors[::-1])[::-1], [1.0]])
    later = np.searchsorted(split_dates, dates, side="right")
    out: np.ndarray = closes.astype(np.float64) * tail[later]
    return out


def reconcile(events: pd.DataFrame, closes: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """One row per event, in the order of ``events``, with the columns in :data:`COLUMNS`.

    ``events`` needs ``symbol``, ``event_date`` and ``median_price``; ``closes`` has
    ``symbol``, ``date``, ``close``; ``splits`` has ``symbol``, ``date``, ``factor``.
    ``price_ratio`` is the reported price over the last as-traded close on or
    before the event date, and ``price_ratio_adjusted`` is the same against the
    split-adjusted close.
    """
    n = len(events)
    last_traded = np.full(n, np.nan)
    last_adjusted = np.full(n, np.nan)
    ok = np.zeros(n, dtype=np.bool_)
    wanted = set(events["symbol"].dropna())
    px = closes[closes["symbol"].isin(wanted) & (closes["close"] > 0)]
    px_by = {str(s): g.sort_values("date") for s, g in px.groupby("symbol", sort=False)}
    sp_by = {
        str(s): g.sort_values("date")
        for s, g in splits[splits["symbol"].isin(wanted)].groupby("symbol", sort=False)
    }
    symbol = events["symbol"].to_numpy()
    day = events["event_date"].to_numpy(dtype="datetime64[D]")
    price = events["median_price"].to_numpy(dtype=np.float64)
    empty_days = np.array([], dtype="datetime64[D]")
    for i in range(n):
        frame = px_by.get(str(symbol[i]))
        if frame is None:
            continue
        dates = frame["date"].to_numpy(dtype="datetime64[D]")
        adjusted = frame["close"].to_numpy(dtype=np.float64)
        sp = sp_by.get(str(symbol[i]))
        traded = as_traded(
            dates,
            adjusted,
            empty_days if sp is None else sp["date"].to_numpy(dtype="datetime64[D]"),
            np.array([]) if sp is None else sp["factor"].to_numpy(dtype=np.float64),
        )
        hi = int(np.searchsorted(dates, day[i], side="right"))
        lo = int(np.searchsorted(dates, day[i] - np.timedelta64(WINDOW_DAYS, "D"), side="left"))
        if hi <= lo:
            continue
        last_traded[i] = traded[hi - 1]
        last_adjusted[i] = adjusted[hi - 1]
        window = traded[lo:hi]
        ok[i] = window.min() / (1.0 + TOLERANCE) <= price[i] <= window.max() * (1.0 + TOLERANCE)
    return pd.DataFrame(
        {
            "close_as_traded": last_traded,
            "price_ratio": price / last_traded,
            "price_ratio_adjusted": price / last_adjusted,
            "price_reconciled": ok,
        },
        index=events.index,
    )


def summary(events: pd.DataFrame, split: pd.Timestamp) -> pd.DataFrame:
    """Join quality by event year and by period. ``events`` carries :data:`COLUMNS`."""
    with np.errstate(invalid="ignore", divide="ignore"):
        off = np.abs(np.log(events["price_ratio"].to_numpy(dtype=np.float64)))
        off_adj = np.abs(np.log(events["price_ratio_adjusted"].to_numpy(dtype=np.float64)))
    frame = pd.DataFrame(
        {
            "events": 1,
            "with_a_close": np.isfinite(off),
            "within_25_percent_split_adjusted_close": off_adj < np.log(1.0 + NEAR),
            "within_25_percent_as_traded_close": off < np.log(1.0 + NEAR),
            "within_10_percent_as_traded_close": off < np.log(1.0 + TOLERANCE),
            "reconciled": events["price_reconciled"].to_numpy(dtype=np.bool_),
        }
    )
    year = events["event_date"].dt.year.astype(str).to_numpy()
    period = np.where(events["event_date"] < split, "before the split date", "from the split date")
    parts = [
        frame.groupby(year).sum(),
        frame.groupby(period).sum(),
        frame.sum().to_frame("all").T,
    ]
    out = pd.concat(parts).astype("int64")
    for name in ("within_25_percent_split_adjusted_close", "within_25_percent_as_traded_close"):
        out[f"share_{name}"] = out[name] / out["events"]
    out["share_reconciled"] = out["reconciled"] / out["events"]
    out.index.name = "sample"
    return out.reset_index()
