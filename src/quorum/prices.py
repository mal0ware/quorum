"""Daily total returns from split-adjusted closes and cash dividends.

The price file gives split-adjusted closes that are not adjusted for dividends,
and a separate table of dividends. The total return on day t is

    r_t = (close_t + dividend_t) / close_{t-1} - 1

where ``close_{t-1}` is the previous close of the same stock on the trading
calendar. Two guards deal with bad prints, and both are counted:

* A close that differs from both neighbours by more than :data:`SPIKE_FACTOR`
  while the neighbours agree to within :data:`NEIGHBOUR_FACTOR` is an isolated
  spike and is dropped.
* A remaining one-day move above :data:`MAX_DAILY_RATIO` is treated as unknown
  and the return for that day is left missing.

A stock is **liquid** in a month if, in the month before, it had at least
:data:`MIN_PRICED_DAYS` closes and a median daily dollar volume of at least the
chosen floor. Dollar volume is close times volume, which a split leaves unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SPIKE_FACTOR = 5.0
NEIGHBOUR_FACTOR = 2.0
MAX_DAILY_RATIO = 11.0
MIN_PRICED_DAYS = 15

PRICE_FILE = "stock_prices.parquet"
DIVIDEND_FILE = "stock_dividend_events.parquet"
TICKER_FILE = "company_tickers.json"
YIELD_FILE = "daily_treasury_yield.parquet"


@dataclass(frozen=True)
class Panel:
    """Daily returns on a common trading calendar, plus monthly liquidity flags."""

    dates: pd.DatetimeIndex
    symbols: list[str]
    ret: NDArray[np.float32]  # [days, stocks]; NaN where there is no return
    months: pd.PeriodIndex
    month_of_day: NDArray[np.int64]  # [days] index into months
    liquid: NDArray[np.bool_]  # [months, stocks]
    notes: dict[str, int] = field(default_factory=dict)

    def column(self, symbol: str) -> int:
        """Column of a symbol."""
        return self.symbols.index(symbol)


def build_panel(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    min_dollar_volume: float,
) -> Panel:
    """Build a :class:`Panel` from long tables.

    ``prices`` has ``symbol``, ``date``, ``close``, ``volume``; ``dividends`` has
    ``symbol``, ``date``, ``amount``. Rows off the ``calendar`` are ignored.
    """
    px = prices[["symbol", "date", "close", "volume"]].copy()
    close = px["close"].to_numpy(dtype=np.float64)
    px = px[np.isfinite(close) & (close > 0) & px["date"].isin(calendar).to_numpy()]
    px = px.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])

    by = px.groupby("symbol", sort=False)["close"]
    c = px["close"].to_numpy(dtype=np.float64)
    prev = by.shift(1).to_numpy(dtype=np.float64)
    nxt = by.shift(-1).to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        off_prev = np.maximum(c / prev, prev / c)
        off_next = np.maximum(c / nxt, nxt / c)
        neighbours = np.maximum(prev / nxt, nxt / prev)
    spike = (off_prev > SPIKE_FACTOR) & (off_next > SPIKE_FACTOR) & (neighbours < NEIGHBOUR_FACTOR)
    px = px[~spike]

    div = dividends.groupby(["symbol", "date"], as_index=False)["amount"].sum()
    px = px.merge(div, on=["symbol", "date"], how="left")
    c = px["close"].to_numpy(dtype=np.float64)
    prev = px.groupby("symbol", sort=False)["close"].shift(1).to_numpy(dtype=np.float64)
    amount = np.nan_to_num(px["amount"].to_numpy(dtype=np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = c / prev
        ret = (c + amount) / prev - 1.0
    extreme = np.isfinite(ratio) & (ratio > MAX_DAILY_RATIO)
    ret[extreme] = np.nan

    symbols = sorted(px["symbol"].unique().tolist())
    col = pd.Index(symbols).get_indexer(px["symbol"])
    row = calendar.get_indexer(px["date"])
    wide = np.full((len(calendar), len(symbols)), np.nan, dtype=np.float32)
    wide[row, col] = ret.astype(np.float32)

    months = calendar.to_period("M").unique()
    month_of_day = months.get_indexer(calendar.to_period("M")).astype(np.int64)
    px_month = months.get_indexer(px["date"].dt.to_period("M"))
    stats = (
        pd.DataFrame(
            {
                "m": px_month,
                "c": col,
                "dv": c * px["volume"].to_numpy(dtype=np.float64),
            }
        )
        .groupby(["m", "c"])["dv"]
        .agg(["median", "size"])
        .reset_index()
    )
    good = stats[(stats["median"] >= min_dollar_volume) & (stats["size"] >= MIN_PRICED_DAYS)]
    liquid = np.zeros((len(months), len(symbols)), dtype=np.bool_)
    nxt_month = good["m"].to_numpy(dtype=np.int64) + 1
    keep = nxt_month < len(months)
    liquid[nxt_month[keep], good["c"].to_numpy(dtype=np.int64)[keep]] = True

    notes = {
        "spike_closes_dropped": int(spike.sum()),
        "extreme_returns_blanked": int(extreme.sum()),
    }
    return Panel(
        dates=calendar,
        symbols=symbols,
        ret=wide,
        months=months,
        month_of_day=month_of_day,
        liquid=liquid,
        notes=notes,
    )


def load_cik_map(path: Path) -> pd.Series:
    """Map SEC CIK to one ticker: the first listed for that CIK in the catalogue."""
    with path.open(encoding="utf-8") as fh:
        rows = list(json.load(fh).values())
    table = pd.DataFrame(rows)[["cik_str", "ticker"]]
    table = table.assign(cik_str=pd.to_numeric(table["cik_str"], errors="coerce"))
    table = table.dropna(subset=["cik_str", "ticker"]).drop_duplicates("cik_str")
    out = table.set_index(table["cik_str"].astype("int64"))["ticker"].astype(str)
    out.index.name = "cik"
    return out


def load_long(prices_dir: Path, symbols: set[str], start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the price and dividend files for the given symbols from ``start`` on."""
    wanted = sorted(symbols)
    px = pd.read_parquet(
        prices_dir / PRICE_FILE,
        columns=["symbol", "report_date", "close", "volume"],
        filters=[("symbol", "in", wanted), ("report_date", ">=", start)],
    )
    px = pd.DataFrame(
        {
            "symbol": px["symbol"].astype(str),
            "date": pd.to_datetime(px["report_date"]),
            "close": px["close"].astype("float64"),
            "volume": px["volume"].astype("float64"),
        }
    )
    dv = pd.read_parquet(prices_dir / DIVIDEND_FILE)
    dv = pd.DataFrame(
        {
            "symbol": dv["symbol"].astype(str),
            "date": pd.to_datetime(dv["report_date"]),
            "amount": dv["amount"].astype("float64"),
        }
    )
    return px, dv[dv["symbol"].isin(symbols)]


def load_monthly_riskfree(path: Path, months: pd.PeriodIndex) -> NDArray[np.float64]:
    """Monthly risk-free return: the month's average 3-month Treasury yield over 12."""
    raw = pd.read_parquet(path, columns=["report_date", "bc_3month"])
    annual = pd.to_numeric(raw["bc_3month"], errors="coerce")
    period = pd.to_datetime(raw["report_date"]).dt.to_period("M")
    monthly = annual.groupby(period).mean() / 12.0
    out: NDArray[np.float64] = monthly.reindex(months).ffill().fillna(0.0).to_numpy(np.float64)
    return out
