"""Cleaning rules and the Layline reader, on small hand-built tables."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from quorum import layline
from quorum.clean import clean_trades, insider_per_filing

GOOD = {
    "accessionNumber": "A1",
    "acceptanceDatetime": "20150303170501",
    "filingDate": "20150303",
    "issuerCIK": "0000000042",
    "documentType": "4",
    "issuerTradingSymbol": "XYZ",
    "transactionType": "nonDerivativeTransaction",
    "securityTitle": "Common Stock",
    "transactionDate": "2015-03-02",
    "transactionCode": "P",
    "transactionShares": "100",
    "transactionPricePerShare": "12.50",
    "transactionAcquiredDisposedCode": "A",
}


def raw(*changes: dict[str, str | None]) -> pd.DataFrame:
    return pd.DataFrame([{**GOOD, **change} for change in changes])


def owners(*rows: tuple[str, str, int, int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["accessionNumber", "rptOwnerCik", "isDirector", "isOfficer", "isTenPercentOwner"],
    ).astype(str)


def test_a_clean_purchase_survives_with_the_right_values() -> None:
    out = clean_trades(raw({}), owners(("A1", "0000000007", 1, 0, 0))).trades
    assert len(out) == 1
    row = out.iloc[0]
    assert (row["issuer_cik"], row["insider_cik"], row["side"]) == (42, 7, "P")
    assert row["trade_date"] == pd.Timestamp("2015-03-02")
    assert row["public_date"] == pd.Timestamp("2015-03-03")
    assert row["value"] == 1250.0


def test_each_rule_removes_what_it_says() -> None:
    table = raw(
        {},
        {"accessionNumber": "A2", "documentType": "4/A"},
        {"accessionNumber": "A3", "transactionAcquiredDisposedCode": "D"},
        {"accessionNumber": "A4", "transactionDate": "2015-03-09"},  # after the filing
        {"accessionNumber": "A5", "transactionPricePerShare": "0"},
        {"accessionNumber": "A6", "securityTitle": "Series A Preferred Stock"},
        {"accessionNumber": "A7"},  # ten percent owner only
        {"accessionNumber": "A8", "acceptanceDatetime": "20030102120000"},  # before mid-2003
        {"accessionNumber": "A9", "transactionShares": "n/a"},
    )
    own = owners(
        *[(f"A{k}", "7", 0, 1, 0) for k in (1, 2, 3, 4, 5, 6, 8, 9)],
        ("A7", "8", 0, 0, 1),
    )
    result = clean_trades(table, own)
    assert result.log["rows_left"].tolist() == [9, 8, 7, 5, 3, 2, 1, 1]
    assert len(result.trades) == 1


def test_split_fills_are_summed_to_one_row() -> None:
    table = raw({}, {"transactionShares": "300", "transactionPricePerShare": "13"})
    out = clean_trades(table, owners(("A1", "7", 1, 0, 0))).trades
    assert len(out) == 1
    assert out["shares"].iloc[0] == 400.0
    assert out["value"].iloc[0] == 100 * 12.5 + 300 * 13


def test_a_joint_filing_counts_as_one_insider() -> None:
    own = owners(
        ("A1", "9", 0, 1, 0), ("A1", "5", 1, 0, 0), ("A1", "3", 0, 0, 1), ("A1", "9", 0, 1, 0)
    )
    assert insider_per_filing(own).to_dict() == {"A1": 5}


def test_sales_need_the_disposed_code() -> None:
    table = raw(
        {"transactionCode": "S", "transactionAcquiredDisposedCode": "D"},
        {"accessionNumber": "A2", "transactionCode": "S", "transactionAcquiredDisposedCode": "A"},
    )
    out = clean_trades(table, owners(("A1", "7", 1, 0, 0), ("A2", "7", 1, 0, 0))).trades
    assert out["side"].tolist() == ["S"]


def test_layline_reader_streams_from_the_zip(tmp_path: Path) -> None:
    panel = raw(
        {},
        {"accessionNumber": "A2", "transactionCode": "M"},  # option exercise: not read
        {"accessionNumber": "A3", "transactionType": "derivativeTransaction"},
        {"accessionNumber": "A4", "transactionCode": "S", "securityTitle": 'Common, "A"\nshares'},
    )
    panel["somethingElse"] = "x"
    panel_zip = tmp_path / layline.PANEL_ZIP
    with zipfile.ZipFile(panel_zip, "w") as zf:
        zf.writestr(layline.PANEL_CSV, panel.to_csv(index=False))
    got = layline.read_panel_trades(panel_zip)
    assert got["accessionNumber"].tolist() == ["A1", "A4"]
    assert list(got.columns) == layline.PANEL_COLUMNS
    assert got["issuerCIK"].iloc[0] == "0000000042"  # text, leading zeros kept

    own = owners(("A1", "7", 1, 0, 0), ("A1", "7", 1, 0, 0), ("ZZ", "8", 1, 0, 0))
    own["rptOwnerName"] = "SOMEONE"
    owner_zip = tmp_path / layline.OWNER_ZIP
    with zipfile.ZipFile(owner_zip, "w") as zf:
        zf.writestr(layline.OWNER_CSV, own.to_csv(index=False))
    got_owners = layline.read_owners(owner_zip, {"A1", "A4"})
    assert got_owners["accessionNumber"].tolist() == ["A1", "A1"]
    assert list(got_owners.columns) == layline.OWNER_COLUMNS
    assert len(clean_trades(got, got_owners).trades) == 1
