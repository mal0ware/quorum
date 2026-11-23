"""Routine and opportunistic insiders (Cohen, Malloy and Pomorski 2012, section I).

At the start of calendar year Y an insider at a firm is classified if she traded
that firm's stock in each of the years Y-1, Y-2 and Y-3. She is **routine** if
some calendar month holds a trade in all three of those years, and
**opportunistic** otherwise. Every trade she makes in year Y carries that label.
Insiders without three straight years of trades are not classified.

A trade is an open-market purchase or sale. Only trades already public before
1 January of year Y are used, so the label is known when the year begins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK_YEARS = 3
ROUTINE = "routine"
OPPORTUNISTIC = "opportunistic"
UNCLASSIFIED = "unclassified"


def classify_insiders(trades: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (issuer_cik, insider_cik, year) that can be classified.

    ``trades`` needs ``issuer_cik``, ``insider_cik``, ``trade_date`` and ``public_date``.
    The result has a ``label`` column holding ``"routine"`` or ``"opportunistic"``.
    """
    base = pd.DataFrame(
        {
            "issuer_cik": trades["issuer_cik"].to_numpy(),
            "insider_cik": trades["insider_cik"].to_numpy(),
            "trade_year": trades["trade_date"].dt.year.to_numpy(),
            "month": trades["trade_date"].dt.month.to_numpy(),
            "public_year": trades["public_date"].dt.year.to_numpy(),
        }
    )
    parts = []
    for ahead in range(1, LOOKBACK_YEARS + 1):
        part = base.assign(year=base["trade_year"] + ahead)
        parts.append(part[part["public_year"] < part["year"]])
    rows = pd.concat(parts, ignore_index=True).drop_duplicates(
        ["issuer_cik", "insider_cik", "year", "trade_year", "month"]
    )
    who = ["issuer_cik", "insider_cik", "year"]

    years_traded = rows.drop_duplicates([*who, "trade_year"]).groupby(who).size()
    eligible = years_traded[years_traded == LOOKBACK_YEARS].index

    same_month = rows.groupby([*who, "month"]).size()
    routine = same_month[same_month == LOOKBACK_YEARS].reset_index()[who].drop_duplicates()
    routine_index = pd.MultiIndex.from_frame(routine)

    out = eligible.to_frame(index=False)
    out["label"] = np.where(eligible.isin(routine_index), ROUTINE, OPPORTUNISTIC)
    return out.sort_values(who).reset_index(drop=True)


def label_trades(trades: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Label each trade by its insider's class in the year the trade became public.

    Trades by insiders who cannot be classified get ``"unclassified"``.
    """
    key = pd.DataFrame(
        {
            "issuer_cik": trades["issuer_cik"].to_numpy(),
            "insider_cik": trades["insider_cik"].to_numpy(),
            "year": trades["public_date"].dt.year.to_numpy(),
        }
    )
    merged = key.merge(labels, on=["issuer_cik", "insider_cik", "year"], how="left")
    out = merged["label"].fillna(UNCLASSIFIED)
    out.index = trades.index
    return out
