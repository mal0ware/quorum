"""The snapshot check in reproduce.py and the manifest reader in fetch_data.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import fetch_data
import reproduce


def frame(rows: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({"file": list(rows), "bytes": 1, "sha256": list(rows.values())})


def test_matching_hashes_report_no_problems() -> None:
    files = frame({"a.zip": "11", "b.parquet": "22"})
    assert reproduce.compare_with_manifest(files, files) == []


def test_a_changed_or_unknown_file_is_named() -> None:
    recorded = frame({"a.zip": "11", "b.parquet": "22"})
    found = frame({"a.zip": "11", "b.parquet": "99", "c.json": "33"})
    problems = reproduce.compare_with_manifest(found, recorded)
    assert len(problems) == 2
    assert problems[0].startswith("b.parquet") and "99" in problems[0] and "22" in problems[0]
    assert problems[1] == "c.json: not in the manifest"


def test_verify_says_match(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    files = frame({"a.zip": "11"})
    path = tmp_path / "manifest.csv"
    files.to_csv(path, index=False)
    assert reproduce.verify_snapshot(files, path, accept=False) is True
    assert "SNAPSHOT MATCHES" in capsys.readouterr().out


def test_verify_stops_on_a_different_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "manifest.csv"
    frame({"a.zip": "11"}).to_csv(path, index=False)
    with pytest.raises(SystemExit) as stop:
        reproduce.verify_snapshot(frame({"a.zip": "77"}), path, accept=False)
    assert "--accept-snapshot" in str(stop.value)
    out = capsys.readouterr().out
    assert "SNAPSHOT DIFFERS" in out and "a.zip" in out and "will not match" in out


def test_verify_runs_on_a_different_snapshot_when_told_to(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "manifest.csv"
    frame({"a.zip": "11"}).to_csv(path, index=False)
    assert reproduce.verify_snapshot(frame({"a.zip": "77"}), path, accept=True) is False
    assert "SNAPSHOT DIFFERS" in capsys.readouterr().out


def test_a_missing_manifest_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert reproduce.verify_snapshot(frame({"a.zip": "11"}), tmp_path / "none.csv", False) is False
    assert "SNAPSHOT NOT CHECKED" in capsys.readouterr().out


def test_missing_input_files_stop_the_run(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as stop:
        reproduce.manifest(tmp_path)
    assert "fetch_data.py" in str(stop.value)


def test_the_committed_manifest_lists_every_input_file() -> None:
    recorded = pd.read_csv(reproduce.MANIFEST)
    wanted = [f.name for f in reproduce.input_files(Path("data"))]
    assert recorded["file"].tolist() == wanted
    assert (recorded["sha256"].str.len() == 64).all()


def test_fetch_reads_the_same_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    frame({"a.zip": "11", "b.parquet": "22"}).to_csv(path, index=False)
    assert fetch_data.recorded_hashes(path) == {"a.zip": "11", "b.parquet": "22"}
    assert fetch_data.recorded_hashes(tmp_path / "none.csv") == {}
    names = {Path(remote.dest).name for remote in fetch_data.FILES}
    assert names == set(pd.read_csv(reproduce.MANIFEST)["file"])
    # The two Layline files are pinned in the script and agree with the manifest.
    recorded = fetch_data.recorded_hashes()
    for remote in fetch_data.FILES:
        if remote.sha256:
            assert recorded[Path(remote.dest).name] == remote.sha256
