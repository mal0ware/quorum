"""Cleaning rules for reported insider transactions.

Each rule is applied in order and the rows left after it are counted, so the
table in ``results/cleaning.csv`` shows what every rule removed.

Rules
-----
1. Form 4 only. Amendments (4/A) restate earlier filings and Form 5 is the late
   annual catch-all; both are dropped rather than reconciled.
2. Purchases must be coded P and "acquired"; sales must be coded S and "disposed".
3. The transaction date and the acceptance time must parse, the filing must be
   accepted on or after :data:`FIRST_PUBLIC_DATE`, and the trade cannot be dated
   after the filing that reports it.
4. Shares and price per share must be positive numbers.
5. The security must not be a preferred share, warrant, option, right, note,
   bond, debenture or convertible.
6. At least one reporting person on the filing must be an officer or a director.
   A filing counts as one insider: the officer or director on it with the lowest
   CIK. Joint filings by affiliated entities are therefore never a "cluster".
7. Rows are summed to one per issuer, insider, side, trade date and public date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Electronic filing of Forms 3, 4 and 5 became mandatory on 30 June 2003.
FIRST_PUBLIC_DATE = pd.Timestamp("2003-06-30")

EXCLUDED_TITLE = r"prefer|warrant|option|debenture|convertible|\bnotes?\b|\bbonds?\b|\brights?\b"

TRADE_COLUMNS = [
    "issuer_cik",
    "insider_cik",
    "side",
    "trade_date",
    "public_date",
    "shares",
    "value",
]


@dataclass(frozen=True)
class CleanResult:
    """Cleaned trades and the row count left after each rule."""

    trades: pd.DataFrame
    log: pd.DataFrame


def _flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0) > 0


def insider_per_filing(owners: pd.DataFrame) -> pd.Series:
    """Map accession number to the lowest officer-or-director CIK on the filing."""
    own = owners.drop_duplicates(["accessionNumber", "rptOwnerCik"])
    inside = own[_flag(own["isDirector"]) | _flag(own["isOfficer"])].copy()
    inside["cik"] = pd.to_numeric(inside["rptOwnerCik"], errors="coerce")
    inside = inside.dropna(subset=["cik"])
    out = inside.groupby("accessionNumber")["cik"].min().astype("int64")
    out.name = "insider_cik"
    return out


def clean_trades(raw: pd.DataFrame, owners: pd.DataFrame) -> CleanResult:
    """Apply the rules above to raw panel rows and their reporting persons."""
    steps: list[tuple[str, int]] = [("raw rows coded P or S", len(raw))]

    df = raw[raw["documentType"] == "4"]
    steps.append(("1 form 4 only", len(df)))

    code = df["transactionCode"]
    acq = df["transactionAcquiredDisposedCode"]
    df = df[((code == "P") & (acq == "A")) | ((code == "S") & (acq == "D"))]
    steps.append(("2 code matches acquired or disposed", len(df)))

    df = df.assign(
        trade_date=pd.to_datetime(df["transactionDate"], format="%Y-%m-%d", errors="coerce"),
        public_date=pd.to_datetime(
            df["acceptanceDatetime"].astype(str).str.slice(0, 8), format="%Y%m%d", errors="coerce"
        ),
    )
    df = df[
        df["trade_date"].notna()
        & df["public_date"].notna()
        & (df["public_date"] >= FIRST_PUBLIC_DATE)
        & (df["trade_date"] <= df["public_date"])
    ]
    steps.append(("3 dates valid, trade not after filing", len(df)))

    df = df.assign(
        shares=pd.to_numeric(df["transactionShares"], errors="coerce"),
        price=pd.to_numeric(df["transactionPricePerShare"], errors="coerce"),
    )
    shares = df["shares"].to_numpy(dtype=np.float64)
    price = df["price"].to_numpy(dtype=np.float64)
    df = df[np.isfinite(shares) & np.isfinite(price) & (shares > 0) & (price > 0)]
    steps.append(("4 positive shares and price", len(df)))

    title = df["securityTitle"].fillna("").str.lower()
    df = df[~title.str.contains(EXCLUDED_TITLE, regex=True)]
    steps.append(("5 common equity titles", len(df)))

    df = df.join(insider_per_filing(owners), on="accessionNumber", how="inner")
    steps.append(("6 filed by an officer or director", len(df)))

    df = df.assign(
        issuer_cik=pd.to_numeric(df["issuerCIK"], errors="coerce"),
        value=df["shares"] * df["price"],
        side=df["transactionCode"],
    ).dropna(subset=["issuer_cik"])
    df["issuer_cik"] = df["issuer_cik"].astype("int64")
    keys = ["issuer_cik", "insider_cik", "side", "trade_date", "public_date"]
    trades = df.groupby(keys, as_index=False, sort=True)[["shares", "value"]].sum()
    steps.append(("7 one row per insider, side, trade date, public date", len(trades)))

    log = pd.DataFrame(steps, columns=["rule", "rows_left"])
    return CleanResult(trades=trades[TRADE_COLUMNS].reset_index(drop=True), log=log)
