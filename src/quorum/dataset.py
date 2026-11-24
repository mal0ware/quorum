"""Load the downloaded files into the inputs the pipeline needs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quorum import clean, layline, prices

LAYLINE_DIR = "layline"
PRICES_DIR = "prices"
CACHE_DIR = "cache"

# The trading calendar is cut here so that a later download gives the same sample.
PRICE_START = "2003-01-01"
PRICE_END = "2026-06-30"
MIN_DOLLAR_VOLUME = 100_000.0
CALENDAR_SYMBOL = "SPY"
BENCHMARK_SYMBOLS = ("SPY", "IWM")


@dataclass(frozen=True)
class Inputs:
    """Everything :func:`quorum.pipeline.run` takes, plus the cleaning log."""

    trades: pd.DataFrame
    cleaning_log: pd.DataFrame
    panel: prices.Panel
    cik_map: pd.Series
    riskfree: NDArray[np.float64]
    closes: pd.DataFrame  # symbol, date, close (split-adjusted), for the price check
    splits: pd.DataFrame  # symbol, date, factor


def load_raw(data_dir: Path, use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw P and S rows and their reporting persons, cached as parquet after the first read."""
    cache = data_dir / CACHE_DIR
    raw_path = cache / "raw_trades_layline.parquet"
    owner_path = cache / "raw_owners_layline.parquet"
    if use_cache and raw_path.exists() and owner_path.exists():
        return pd.read_parquet(raw_path), pd.read_parquet(owner_path)
    folder = data_dir / LAYLINE_DIR
    raw = layline.read_panel_trades(folder / layline.PANEL_ZIP)
    owners = layline.read_owners(folder / layline.OWNER_ZIP, set(raw["accessionNumber"]))
    cache.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(raw_path)
    owners.to_parquet(owner_path)
    return raw, owners


def load_inputs(data_dir: Path, use_cache: bool = True) -> Inputs:
    """Read the insider tables and the price files and build the pipeline inputs."""
    raw, owners = load_raw(data_dir, use_cache)
    cleaned = clean.clean_trades(raw, owners)

    price_dir = data_dir / PRICES_DIR
    cik_map = prices.load_cik_map(price_dir / prices.TICKER_FILE)
    issuers = cleaned.trades["issuer_cik"].unique()
    symbols = set(cik_map.reindex(issuers).dropna()) | set(BENCHMARK_SYMBOLS)
    px, dv = prices.load_long(price_dir, symbols, PRICE_START)
    px = px[px["date"] <= pd.Timestamp(PRICE_END)]
    calendar = pd.DatetimeIndex(
        np.sort(px.loc[px["symbol"] == CALENDAR_SYMBOL, "date"].unique()), name="date"
    )
    panel = prices.build_panel(px, dv, calendar, MIN_DOLLAR_VOLUME)
    riskfree = prices.load_monthly_riskfree(price_dir / prices.YIELD_FILE, panel.months)
    return Inputs(
        trades=cleaned.trades,
        cleaning_log=cleaned.log,
        panel=panel,
        cik_map=cik_map[cik_map.isin(panel.symbols)],
        riskfree=riskfree,
        closes=px[["symbol", "date", "close"]].reset_index(drop=True),
        splits=prices.load_splits(price_dir / prices.SPLIT_FILE, symbols),
    )
