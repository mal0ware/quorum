"""Shared helpers. Tests never touch the network or the data/ folder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from quorum import prices, synthetic
from quorum.dataset import Inputs


def buys_frame(rows: list[tuple[int, int, str, str]]) -> pd.DataFrame:
    """Purchases from (issuer, insider, trade date, public date) tuples."""
    frame = pd.DataFrame(rows, columns=["issuer_cik", "insider_cik", "trade_date", "public_date"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["public_date"] = pd.to_datetime(frame["public_date"])
    frame["side"] = "P"
    frame["shares"] = 100.0
    frame["value"] = 1000.0
    return frame


def panel_from_returns(
    returns: np.ndarray, symbols: list[str], start: str = "2010-01-04"
) -> prices.Panel:
    """A panel whose daily returns are exactly ``returns`` (rows are business days)."""
    calendar = pd.bdate_range(start, periods=returns.shape[0] + 1, name="date")
    frames = []
    for j, name in enumerate(symbols):
        close = 50.0 * np.concatenate([[1.0], np.cumprod(1.0 + returns[:, j])])
        frames.append(
            pd.DataFrame({"symbol": name, "date": calendar, "close": close, "volume": 1e6})
        )
    px = pd.concat(frames, ignore_index=True)
    dv = pd.DataFrame({"symbol": [], "date": pd.to_datetime([]), "amount": []})
    return prices.build_panel(px, dv, calendar, 100_000.0)


@pytest.fixture(scope="session")
def synthetic_inputs() -> Inputs:
    return synthetic.make(seed=0)
