"""Download the public data into data/ (never committed).

    uv run python scripts/fetch_data.py

Default source: about 1.3 GB from two open archives. No account, no key, and no
personal details are sent: the only header set is a generic client name.

* Layline insider trading dataset (Harvard Dataverse, CC BY 4.0), two tables.
  Fetched by file id, so the bytes are fixed; the MD5 that Dataverse publishes
  and a SHA-256 are both checked.
* Prices, dividends, the CIK-to-ticker catalogue and Treasury yields from the
  defeatbeta/yahoo-finance-data dataset on Hugging Face (ODC-BY). That dataset
  is rewritten daily and keeps no history, so it cannot be pinned. A hash that
  differs from the published run is reported and the download is kept.

Optional source, off by default:

    uv run python scripts/fetch_data.py --source edgar --first-year 2006 --last-year 2025

This reads the SEC's own quarterly files instead of Layline. The SEC asks for a
contact in the User-Agent header, so this path reads it from the environment
variable SEC_USER_AGENT and stops before any request if it is not set.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from quorum import edgar

ROOT = Path(__file__).resolve().parents[1]
CLIENT = "quorum-fetch/0.1 (research script)"

DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
HUGGING_FACE = (
    "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/US/{name}"
)


@dataclass(frozen=True)
class Remote:
    """One file to download. ``pinned`` files must match their hash."""

    url: str
    dest: str
    sha256: str
    pinned: bool
    md5: str = ""


FILES = [
    # Layline dataset, version 430.0 (released 2026-05-21), doi:10.7910/DVN/VH6GVH
    Remote(
        DATAVERSE.format(file_id=13148305),
        "layline/lit_panel.zip",
        "1eb80b20c4045dbce3f9373232334409bbaa80c5fbb05ea33a9ac8b5b84d6f33",
        True,
        "d1bc518ff455060e3f6bbc6bbc85f0f0",
    ),
    Remote(
        DATAVERSE.format(file_id=13148287),
        "layline/lit_reportingowner.zip",
        "4474992f5f2986a8dee0406f87cd22fb6f1c4ab76e6e8dbb0c0478c3f29c3108",
        True,
        "de4ab97a117d61ec320ab67581ea7038",
    ),
    # Snapshot of 2026-11-23. Later downloads will differ.
    Remote(
        HUGGING_FACE.format(name="stock_prices.parquet"),
        "prices/stock_prices.parquet",
        "1db00824c155a7832b27ddadbedf72f84bc9997e9fb0a81c6d7b5590b633f472",
        False,
    ),
    Remote(
        HUGGING_FACE.format(name="stock_dividend_events.parquet"),
        "prices/stock_dividend_events.parquet",
        "a47b74649dbb3833157f31254cb90a57c3482cb2c22902b9a9e56c028568bf16",
        False,
    ),
    Remote(
        HUGGING_FACE.format(name="company_tickers.json"),
        "prices/company_tickers.json",
        "0b4457b35054fda2c8be37964d29f16e06c635d85575f8dd575695c519068817",
        False,
    ),
    Remote(
        HUGGING_FACE.format(name="daily_treasury_yield.parquet"),
        "prices/daily_treasury_yield.parquet",
        "7de3851415c57e0e169c88d5e415205c6f106c5734c8d60b5a907fb9b96b88f3",
        False,
    ),
]


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


def fetch(remote: Remote, data_dir: Path, force: bool) -> str:
    """Fetch one file unless it is already there. Returns a line for FETCHED.txt."""
    dest = data_dir / remote.dest
    if force or not dest.exists():
        print(f"downloading {remote.dest}", flush=True)
        download(remote.url, dest)
    sha, md5 = digests(dest)
    if sha == remote.sha256 and (not remote.md5 or md5 == remote.md5):
        status = "matches the published run"
    elif remote.pinned:
        raise SystemExit(f"{remote.dest}: hash {sha} does not match the pinned file. Stopping.")
    else:
        status = "DIFFERS from the published run (this source is not versioned)"
        print(f"note: {remote.dest} {status}; results may differ slightly", flush=True)
    return f"{remote.dest}\t{dest.stat().st_size}\t{sha}\t{status}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--source", choices=["layline", "edgar"], default="layline")
    parser.add_argument("--first-year", type=int, default=2006)
    parser.add_argument("--last-year", type=int, default=2025)
    parser.add_argument("--force", action="store_true", help="download again even if present")
    args = parser.parse_args()

    remotes = FILES
    if args.source == "edgar":
        try:
            edgar.user_agent()
        except edgar.MissingUserAgentError as err:
            sys.exit(str(err))
        for year in range(args.first_year, args.last_year + 1):
            for quarter in (1, 2, 3, 4):
                path = edgar.download_quarter(year, quarter, args.data_dir / "edgar")
                print(f"have {path.name}", flush=True)
        remotes = [r for r in FILES if r.dest.startswith("prices/")]

    lines = [fetch(remote, args.data_dir, args.force) for remote in remotes]
    (args.data_dir / "FETCHED.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done: {len(lines)} files in {args.data_dir}")


if __name__ == "__main__":
    main()
