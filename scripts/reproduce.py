"""Rebuild every table and figure.

    uv run python scripts/fetch_data.py     # once: downloads into data/
    uv run python scripts/reproduce.py      # writes results/
    uv run python scripts/reproduce.py --fast

Before anything runs, every input file is hashed and compared with
results/data_manifest.csv, the record of the files behind the committed results.
If they all match, the run rebuilds the committed tables. If any file differs
(the price source is rewritten daily and cannot be pinned) the script says
which, and stops. Pass --accept-snapshot to run anyway: the tables will then
differ from the ones quoted in the README, and the manifest is rewritten to
describe the files that were used.

``--fast`` needs no data and no network. It runs the whole pipeline on a small
seeded synthetic data set and writes to results_fast/ (gitignored). It checks
that the code runs end to end; its numbers are not findings.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd

from quorum import dataset, layline, pipeline, plots, prices, synthetic

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "data_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def input_files(data_dir: Path) -> list[Path]:
    """Every file the real run reads."""
    folder = data_dir / dataset.LAYLINE_DIR
    price_dir = data_dir / dataset.PRICES_DIR
    return [
        folder / layline.PANEL_ZIP,
        folder / layline.OWNER_ZIP,
        price_dir / prices.PRICE_FILE,
        price_dir / prices.DIVIDEND_FILE,
        price_dir / prices.SPLIT_FILE,
        price_dir / prices.TICKER_FILE,
        price_dir / prices.YIELD_FILE,
    ]


def manifest(data_dir: Path) -> pd.DataFrame:
    """Size and hash of every input file, so a reader can tell which snapshot was used."""
    files = input_files(data_dir)
    missing = [f.name for f in files if not f.exists()]
    if missing:
        sys.exit(f"missing input files: {', '.join(missing)}. Run scripts/fetch_data.py first.")
    rows = [{"file": f.name, "bytes": f.stat().st_size, "sha256": sha256(f)} for f in files]
    return pd.DataFrame(rows)


def compare_with_manifest(found: pd.DataFrame, recorded: pd.DataFrame) -> list[str]:
    """One line for every file whose hash is not the recorded one. Empty if all match."""
    expected = dict(zip(recorded["file"], recorded["sha256"], strict=True))
    problems = []
    for name, digest in zip(found["file"], found["sha256"], strict=True):
        if name not in expected:
            problems.append(f"{name}: not in the manifest")
        elif expected[name] != digest:
            problems.append(
                f"{name}: sha256 {digest[:16]}..., the manifest has {expected[name][:16]}..."
            )
    return problems


def verify_snapshot(found: pd.DataFrame, manifest_path: Path, accept: bool) -> bool:
    """Say whether the files on disk are the ones behind the committed results.

    Returns True on a match. On a mismatch the run stops unless ``accept`` is set.
    """
    if not manifest_path.exists():
        print(f"SNAPSHOT NOT CHECKED: {manifest_path} does not exist.")
        return False
    problems = compare_with_manifest(found, pd.read_csv(manifest_path, dtype=str))
    if not problems:
        print(f"SNAPSHOT MATCHES: all {len(found)} input files have the hashes in the manifest.")
        print("This run rebuilds the committed tables.")
        return True
    print("SNAPSHOT DIFFERS from results/data_manifest.csv:")
    for line in problems:
        print(f"  {line}")
    print("Tables built from these files will not match the numbers in README.md and PAPER.md.")
    if not accept:
        sys.exit(
            "Stopped without writing anything. Pass --accept-snapshot to run on these files; "
            "the manifest is then rewritten to describe them."
        )
    print("--accept-snapshot given: running, and rewriting the manifest for these files.")
    return False


def panel_notes(inputs: dataset.Inputs) -> pd.DataFrame:
    panel = inputs.panel
    items = {
        "trading_days": len(panel.dates),
        "first_day": str(panel.dates[0].date()),
        "last_day": str(panel.dates[-1].date()),
        "symbols_with_prices": len(panel.symbols),
        "issuers_mapped_to_a_symbol": len(inputs.cik_map),
        "cleaned_trade_rows": len(inputs.trades),
        **panel.notes,
    }
    return pd.DataFrame({"item": list(items), "value": [str(v) for v in items.values()]})


def write(
    results: pipeline.Results, inputs: dataset.Inputs, config: pipeline.Config, out: Path
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tables = dict(results.tables)
    tables["cleaning"] = inputs.cleaning_log
    tables["panel"] = panel_notes(inputs)
    tables["cluster_events"] = results.cluster_events
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False, float_format="%.6g", lineterminator="\n")
    plots.estimates(
        tables["robustness"],
        tables["calendar_time"],
        config.headline_horizon,
        out / "fig1_estimates.png",
    )
    plots.bhar_paths(tables["bhar_paths"], tables["event_study"], out / "fig2_after_a_cluster.png")
    split = pd.Timestamp(config.split_date)
    plots.by_year(
        tables["by_year"], config.headline_horizon, split.year + 0.5, out / "fig3_by_year.png"
    )
    plots.coverage(tables["sample"], out / "fig4_coverage.png")
    plots.survivorship(tables["survivorship_bounds"], out / "fig5_survivorship_bounds.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="synthetic data, no downloads")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true", help="re-read the raw zips")
    parser.add_argument(
        "--accept-snapshot",
        action="store_true",
        help="run even if the input files differ from results/data_manifest.csv",
    )
    parser.add_argument("--verify-only", action="store_true", help="check the hashes and stop")
    args = parser.parse_args()

    started = time.time()
    config = pipeline.Config()
    found = None
    if args.fast:
        out = args.out or ROOT / "results_fast"
        inputs = synthetic.make(seed=0)
        print("synthetic data: these numbers are not findings")
    else:
        out = args.out or ROOT / "results"
        found = manifest(args.data_dir)
        verify_snapshot(found, MANIFEST, args.accept_snapshot)
        if args.verify_only:
            return
        inputs = dataset.load_inputs(args.data_dir, not args.no_cache)
    results = pipeline.run(
        inputs.trades,
        inputs.panel,
        inputs.cik_map,
        inputs.riskfree,
        config,
        closes=inputs.closes,
        splits=inputs.splits,
    )
    write(results, inputs, config, out)
    if found is not None:
        found.to_csv(out / "data_manifest.csv", index=False, lineterminator="\n")

    study = results.tables["event_study"]
    head = study[
        (study["group"] == "all clusters")
        & (study["benchmark"] == "universe")
        & (study["horizon_days"] == config.headline_horizon)
    ]
    columns = ["period", "events", "mean_bhar", "t_month_cluster", "t_overlap", "median_bhar"]
    print(head[columns].to_string(index=False))
    print(f"wrote {out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
