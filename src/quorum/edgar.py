"""Optional: read the SEC's own Insider Transactions Data Sets.

This path is switched off unless you choose to use it. The SEC asks automated
clients to identify themselves in the ``User-Agent`` header, so every request
here takes that header from the environment variable ``SEC_USER_AGENT`` and the
module refuses to make a request when it is unset. Nothing in this repository
sets it for you, and the published results were not produced with it.

The quarterly files are zips of tab-separated tables. Three are used:
``SUBMISSION.tsv``, ``REPORTINGOWNER.tsv`` and ``NONDERIV_TRANS.tsv``. They are
turned into the same raw columns as the Layline tables, so the cleaning rules in
:mod:`quorum.clean` apply unchanged. These files carry a filing date but no
acceptance time, so the filing date stands in for the public date.

The column names follow the SEC's published description of the data sets. They
have been tested against hand-built files only, never against a live download.
"""

from __future__ import annotations

import os
import shutil
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import pandas as pd

ENV_VAR = "SEC_USER_AGENT"
URL_TEMPLATE = (
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets/{year}q{quarter}_form345.zip"
)
# The SEC's fair-access limit is ten requests a second. One a second is plenty here.
SECONDS_BETWEEN_REQUESTS = 1.0

Opener = Callable[..., Any]


class MissingUserAgentError(RuntimeError):
    """Raised when ``SEC_USER_AGENT`` is not set."""


def user_agent() -> str:
    """The ``User-Agent`` to send, from the environment. Raises if it is not set."""
    value = os.environ.get(ENV_VAR, "").strip()
    if not value:
        raise MissingUserAgentError(
            f"{ENV_VAR} is not set. The SEC asks automated clients to send a contact "
            f"name and email in the User-Agent header. Set {ENV_VAR} yourself to use "
            "this path; no request has been made."
        )
    return value


def quarter_url(year: int, quarter: int) -> str:
    """URL of one quarterly data set."""
    if quarter not in (1, 2, 3, 4):
        raise ValueError("quarter must be 1, 2, 3 or 4")
    return URL_TEMPLATE.format(year=year, quarter=quarter)


def download_quarter(
    year: int,
    quarter: int,
    dest_dir: Path,
    opener: Opener = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download one quarter into ``dest_dir`` and return its path.

    Raises :class:`MissingUserAgentError` before any request if the environment
    variable is unset. A file already present is not fetched again.
    """
    agent = user_agent()
    dest = dest_dir / f"{year}q{quarter}_form345.zip"
    if dest.exists():
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(quarter_url(year, quarter), headers={"User-Agent": agent})
    partial = dest.with_suffix(".part")
    with opener(request, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(dest)
    sleep(SECONDS_BETWEEN_REQUESTS)
    return dest


def _table(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    member = next(n for n in zf.namelist() if n.upper().endswith(name))
    handle: IO[bytes]
    with zf.open(member) as handle:
        return pd.read_csv(handle, sep="\t", dtype=str, keep_default_na=False, na_values=[""])


def _sec_date(series: pd.Series, out_format: str) -> pd.Series:
    """Dates arrive as 31-MAR-2024; return them as text in ``out_format``."""
    parsed = pd.to_datetime(series, format="%d-%b-%Y", errors="coerce")
    return parsed.dt.strftime(out_format)


def read_quarter(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(raw trades, reporting owners)`` in the Layline column names."""
    with zipfile.ZipFile(zip_path) as zf:
        sub = _table(zf, "SUBMISSION.TSV")
        own = _table(zf, "REPORTINGOWNER.TSV")
        trans = _table(zf, "NONDERIV_TRANS.TSV")

    trans = trans[trans["TRANS_CODE"].isin(["P", "S"])]
    merged = trans.merge(sub, on="ACCESSION_NUMBER", how="inner")
    raw = pd.DataFrame(
        {
            "accessionNumber": merged["ACCESSION_NUMBER"],
            "acceptanceDatetime": _sec_date(merged["FILING_DATE"], "%Y%m%d") + "000000",
            "filingDate": _sec_date(merged["FILING_DATE"], "%Y%m%d"),
            "issuerCIK": merged["ISSUERCIK"],
            "documentType": merged["DOCUMENT_TYPE"],
            "issuerTradingSymbol": merged["ISSUERTRADINGSYMBOL"],
            "transactionType": "nonDerivativeTransaction",
            "securityTitle": merged["SECURITY_TITLE"],
            "transactionDate": _sec_date(merged["TRANS_DATE"], "%Y-%m-%d"),
            "transactionCode": merged["TRANS_CODE"],
            "transactionShares": merged["TRANS_SHARES"],
            "transactionPricePerShare": merged["TRANS_PRICEPERSHARE"],
            "transactionAcquiredDisposedCode": merged["TRANS_ACQUIRED_DISP_CD"],
        }
    ).reset_index(drop=True)

    relation = own["RPTOWNER_RELATIONSHIP"].fillna("").str.lower()
    owners = pd.DataFrame(
        {
            "accessionNumber": own["ACCESSION_NUMBER"],
            "rptOwnerCik": own["RPTOWNERCIK"],
            "isDirector": relation.str.contains("director").astype(int).astype(str),
            "isOfficer": relation.str.contains("officer").astype(int).astype(str),
            "isTenPercentOwner": relation.str.contains("tenpercentowner").astype(int).astype(str),
        }
    )
    owners = owners[owners["accessionNumber"].isin(set(raw["accessionNumber"]))]
    return raw, owners.reset_index(drop=True)


def read_all(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every quarterly zip in ``directory``."""
    paths = sorted(directory.glob("*_form345.zip"))
    if not paths:
        raise FileNotFoundError(f"no quarterly files in {directory}")
    pairs = [read_quarter(p) for p in paths]
    raw = pd.concat([p[0] for p in pairs], ignore_index=True)
    owners = pd.concat([p[1] for p in pairs], ignore_index=True)
    return raw, owners
