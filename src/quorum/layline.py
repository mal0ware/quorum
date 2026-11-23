"""Read the Layline insider trading dataset (Balogh 2023, doi:10.7910/DVN/VH6GVH).

Two of its tables are used:

``lit_panel``          one row per reported transaction, with the filing's dates and issuer
``lit_reportingowner`` one row per (filing, reporting person)

Both arrive as a zip holding one large CSV. They are streamed straight out of the
zip, a few columns at a time, so the 5 GB of text never has to sit on disk or in memory.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

PANEL_ZIP = "lit_panel.zip"
PANEL_CSV = "lit_panel.csv"
OWNER_ZIP = "lit_reportingowner.zip"
OWNER_CSV = "lit_reportingowner.csv"

PANEL_COLUMNS = [
    "accessionNumber",
    "acceptanceDatetime",
    "filingDate",
    "issuerCIK",
    "documentType",
    "issuerTradingSymbol",
    "transactionType",
    "securityTitle",
    "transactionDate",
    "transactionCode",
    "transactionShares",
    "transactionPricePerShare",
    "transactionAcquiredDisposedCode",
]
OWNER_COLUMNS = ["accessionNumber", "rptOwnerCik", "isDirector", "isOfficer", "isTenPercentOwner"]

# Open-market or private purchase, and open-market or private sale (Form 4 general instruction 8).
TRADE_CODES = ("P", "S")


def _batches(
    zip_path: Path, member: str, columns: list[str], block_mb: int = 64
) -> Iterator[pa.RecordBatch]:
    """Stream one CSV out of a zip as record batches of strings."""
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as raw:
        reader = pacsv.open_csv(
            io.BufferedReader(raw, buffer_size=1 << 22),  # type: ignore[type-var]
            read_options=pacsv.ReadOptions(block_size=block_mb << 20),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                include_columns=columns,
                column_types={c: pa.string() for c in columns},
                strings_can_be_null=True,
            ),
        )
        yield from reader


def read_panel_trades(zip_path: Path) -> pd.DataFrame:
    """Non-derivative purchases and sales (codes P and S) from ``lit_panel``.

    Every value is returned as text, exactly as filed. Cleaning happens in
    :mod:`quorum.clean`, where each rule is counted.
    """
    kept: list[pa.Table] = []
    for batch in _batches(zip_path, PANEL_CSV, PANEL_COLUMNS):
        mask = pc.and_(
            pc.equal(batch.column("transactionType"), "nonDerivativeTransaction"),
            pc.is_in(batch.column("transactionCode"), value_set=pa.array(TRADE_CODES)),
        )
        kept.append(pa.Table.from_batches([batch]).filter(pc.fill_null(mask, False)))
    table = (
        pa.concat_tables(kept)
        if kept
        else pa.table({c: pa.array([], pa.string()) for c in PANEL_COLUMNS})
    )
    frame: pd.DataFrame = table.to_pandas().reset_index(drop=True)
    return frame


def read_owners(zip_path: Path, accessions: set[str]) -> pd.DataFrame:
    """Reporting persons for the given filings, from ``lit_reportingowner``."""
    wanted = pa.array(sorted(accessions), pa.string())
    kept: list[pa.Table] = []
    for batch in _batches(zip_path, OWNER_CSV, OWNER_COLUMNS):
        mask = pc.is_in(batch.column("accessionNumber"), value_set=wanted)
        kept.append(pa.Table.from_batches([batch]).filter(mask))
    table = (
        pa.concat_tables(kept)
        if kept
        else pa.table({c: pa.array([], pa.string()) for c in OWNER_COLUMNS})
    )
    frame: pd.DataFrame = table.to_pandas().reset_index(drop=True)
    return frame
