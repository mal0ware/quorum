"""Download the public data into data/ (never committed).

    uv run python scripts/fetch_data.py

About 1.3 GB from two open archives. No account, no key, and no personal details
are sent: the only header set is a generic client name.

* Layline insider trading dataset (Harvard Dataverse, CC BY 4.0), two tables.
  Fetched by file id, so the bytes are fixed; the MD5 that Dataverse publishes
  and a SHA-256 are both checked, and a mismatch stops the script.
* Prices, dividends, splits, the CIK-to-ticker catalogue and Treasury yields from
  the defeatbeta/yahoo-finance-data dataset on Hugging Face (ODC-BY). That
  dataset is rewritten daily and keeps no history, so it cannot be pinned. Each
  file is compared with the hash in results/data_manifest.csv. A file that
  differs is kept and reported, and reproduce.py will then stop unless it is
  given --accept-snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "data_manifest.csv"
CLIENT = "quorum-fetch/0.1 (research script)"

DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
HUGGING_FACE = (
    "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/US/{name}"
)


@dataclass(frozen=True)
class Remote:
    """One file to download. A file with a hash here is pinned and must match it."""

    url: str
    dest: str
    sha256: str = ""
    md5: str = ""


FILES = [
    # Layline dataset, version 430.0 (released 2026-05-21), doi:10.7910/DVN/VH6GVH
    Remote(
        DATAVERSE.format(file_id=13148305),
        "layline/lit_panel.zip",
        "1eb80b20c4045dbce3f9373232334409bbaa80c5fbb05ea33a9ac8b5b84d6f33",
        "d1bc518ff455060e3f6bbc6bbc85f0f0",
    ),
    Remote(
        DATAVERSE.format(file_id=13148287),
        "layline/lit_reportingowner.zip",
        "4474992f5f2986a8dee0406f87cd22fb6f1c4ab76e6e8dbb0c0478c3f29c3108",
        "de4ab97a117d61ec320ab67581ea7038",
    ),
    # Not versioned at the source. The hashes of the snapshot behind the committed
    # results are in results/data_manifest.csv.
    Remote(HUGGING_FACE.format(name="stock_prices.parquet"), "prices/stock_prices.parquet"),
    Remote(
        HUGGING_FACE.format(name="stock_dividend_events.parquet"),
        "prices/stock_dividend_events.parquet",
    ),
    Remote(
        HUGGING_FACE.format(name="stock_split_events.parquet"),
        "prices/stock_split_events.parquet",
    ),
    Remote(HUGGING_FACE.format(name="company_tickers.json"), "prices/company_tickers.json"),
    Remote(
        HUGGING_FACE.format(name="daily_treasury_yield.parquet"),
        "prices/daily_treasury_yield.parquet",
    ),
]


def recorded_hashes(manifest: Path = MANIFEST) -> dict[str, str]:
    """File name to SHA-256 from the committed manifest."""
    if not manifest.exists():
        return {}
    with manifest.open(encoding="utf-8", newline="") as fh:
        return {row["file"]: row["sha256"] for row in csv.DictReader(fh)}


def digests(path: Path) -> tuple[str, str]:
    """SHA-256 and MD5 of a file."""
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": CLIENT})
    partial = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 22)
    partial.replace(dest)


def fetch(remote: Remote, data_dir: Path, force: bool, recorded: dict[str, str]) -> str:
    """Fetch one file unless it is already there. Returns a line for FETCHED.txt."""
    dest = data_dir / remote.dest
    if force or not dest.exists():
        print(f"downloading {remote.dest}", flush=True)
        download(remote.url, dest)
    sha, md5 = digests(dest)
    if remote.sha256:
        if sha != remote.sha256 or (remote.md5 and md5 != remote.md5):
            raise SystemExit(f"{remote.dest}: hash {sha} does not match the pinned file. Stopping.")
        status = "matches the pinned file"
    elif recorded.get(dest.name) == sha:
        status = "matches results/data_manifest.csv"
    else:
        status = "DIFFERS from results/data_manifest.csv (this source is not versioned)"
        print(f"note: {remote.dest} {status}", flush=True)
    return f"{remote.dest}\t{dest.stat().st_size}\t{sha}\t{status}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--force", action="store_true", help="download again even if present")
    args = parser.parse_args()

    recorded = recorded_hashes()
    lines = [fetch(remote, args.data_dir, args.force, recorded) for remote in FILES]
    (args.data_dir / "FETCHED.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done: {len(lines)} files in {args.data_dir}")
    if any("DIFFERS" in line for line in lines):
        print(
            "At least one file is not the snapshot behind the committed results. "
            "reproduce.py will say so and needs --accept-snapshot to run."
        )


if __name__ == "__main__":
    main()
