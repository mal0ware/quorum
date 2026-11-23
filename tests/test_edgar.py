"""The optional SEC path. The network is always mocked."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import pytest

import fetch_data
from quorum import edgar
from quorum.clean import clean_trades


class FakeResponse(io.BytesIO):
    def __enter__(self) -> FakeResponse:
        return self


class Recorder:
    """Stands in for urlopen and remembers what it was asked."""

    def __init__(self, payload: bytes = b"zip-bytes") -> None:
        self.payload = payload
        self.requests: list[Any] = []

    def __call__(self, request: Any, timeout: float = 0) -> FakeResponse:
        self.requests.append(request)
        return FakeResponse(self.payload)


def test_refuses_to_run_without_the_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(edgar.ENV_VAR, raising=False)
    opener = Recorder()
    with pytest.raises(edgar.MissingUserAgentError, match=edgar.ENV_VAR):
        edgar.download_quarter(2024, 1, tmp_path, opener=opener, sleep=lambda s: None)
    assert opener.requests == []
    assert list(tmp_path.iterdir()) == []


def test_a_blank_value_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(edgar.ENV_VAR, "   ")
    with pytest.raises(edgar.MissingUserAgentError):
        edgar.user_agent()


def test_sends_exactly_the_header_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(edgar.ENV_VAR, "placeholder-agent-for-tests")
    opener = Recorder()
    naps: list[float] = []
    path = edgar.download_quarter(2024, 3, tmp_path, opener=opener, sleep=naps.append)
    assert path.read_bytes() == b"zip-bytes"
    assert len(opener.requests) == 1
    request = opener.requests[0]
    assert request.full_url == edgar.quarter_url(2024, 3)
    assert request.full_url.endswith("/2024q3_form345.zip")
    assert request.get_header("User-agent") == "placeholder-agent-for-tests"
    assert naps == [edgar.SECONDS_BETWEEN_REQUESTS]

    # A file already on disk is not requested again.
    edgar.download_quarter(2024, 3, tmp_path, opener=opener, sleep=naps.append)
    assert len(opener.requests) == 1


def test_quarter_must_be_valid() -> None:
    with pytest.raises(ValueError):
        edgar.quarter_url(2024, 5)


def test_fetch_script_stops_before_any_request_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(edgar.ENV_VAR, raising=False)

    def no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the network was touched")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    monkeypatch.setattr(
        "sys.argv", ["fetch_data.py", "--source", "edgar", "--data-dir", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as stop:
        fetch_data.main()
    assert edgar.ENV_VAR in str(stop.value)
    assert list(tmp_path.iterdir()) == []


def test_default_downloads_carry_no_identity_and_avoid_the_sec() -> None:
    assert "@" not in fetch_data.CLIENT
    for remote in fetch_data.FILES:
        assert "sec.gov" not in remote.url
        assert remote.url.startswith("https://")
        assert len(remote.sha256) == 64


SUBMISSION = (
    "ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERCIK\tISSUERTRADINGSYMBOL\n"
    "0001-24-000001\t05-MAR-2024\t4\t0000000042\tXYZ\n"
    "0001-24-000002\t06-MAR-2024\t4\t0000000042\tXYZ\n"
)
OWNER = (
    "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNER_RELATIONSHIP\n"
    "0001-24-000001\t0000000007\tDirector,Officer\n"
    "0001-24-000002\t0000000008\tTenPercentOwner\n"
)
TRANS = (
    "ACCESSION_NUMBER\tSECURITY_TITLE\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\t"
    "TRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\n"
    "0001-24-000001\tCommon Stock\t04-MAR-2024\tP\t200\t10.5\tA\n"
    "0001-24-000001\tCommon Stock\t04-MAR-2024\tA\t50\t0\tA\n"
    "0001-24-000002\tCommon Stock\t05-MAR-2024\tP\t900\t10.6\tA\n"
)


def test_quarterly_file_is_read_into_the_layline_columns(tmp_path: Path) -> None:
    path = tmp_path / "2024q1_form345.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("SUBMISSION.tsv", SUBMISSION)
        zf.writestr("REPORTINGOWNER.tsv", OWNER)
        zf.writestr("NONDERIV_TRANS.tsv", TRANS)
    raw, owners = edgar.read_all(tmp_path)
    assert raw["transactionCode"].tolist() == ["P", "P"]
    assert raw["acceptanceDatetime"].tolist() == ["20240305000000", "20240306000000"]
    assert raw["transactionDate"].tolist() == ["2024-03-04", "2024-03-05"]
    assert owners["isDirector"].tolist() == ["1", "0"]
    assert owners["isTenPercentOwner"].tolist() == ["0", "1"]

    trades = clean_trades(raw, owners).trades  # the ten percent owner is dropped
    assert len(trades) == 1
    assert trades["insider_cik"].iloc[0] == 7
    assert trades["value"].iloc[0] == 2100.0
    assert str(trades["public_date"].iloc[0].date()) == "2024-03-05"


def test_reading_an_empty_folder_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        edgar.read_all(tmp_path)
