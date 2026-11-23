"""Rebuild every table and figure.

    uv run python scripts/fetch_data.py     # once: downloads into data/
    uv run python scripts/reproduce.py      # writes results/
    uv run python scripts/reproduce.py --fast

``--fast`` needs no data and no network. It runs the whole pipeline on a small
seeded synthetic data set and writes to results_fast/ (gitignored). It checks
that the code runs end to end; its numbers are not findings.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import pandas as pd

from quorum import dataset, layline, pipeline, plots, prices, synthetic

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(data_dir: Path, source: str) -> pd.DataFrame:
    """Size and hash of every input file, so a reader can tell which snapshot was used."""
    if source == "edgar":
        files = sorted((data_dir / dataset.EDGAR_DIR).glob("*_form345.zip"))
    else:
        folder = data_dir / dataset.LAYLINE_DIR
        files = [folder / layline.PANEL_ZIP, folder / layline.OWNER_ZIP]
    price_dir = data_dir / dataset.PRICES_DIR
    files += [
        price_dir / prices.PRICE_FILE,
        price_dir / prices.DIVIDEND_FILE,
        price_dir / prices.TICKER_FILE,
        price_dir / prices.YIELD_FILE,
    ]
    rows = [{"file": f.name, "bytes": f.stat().st_size, "sha256": sha256(f)} for f in files]
    return pd.DataFrame(rows)


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
    plots.bhar_paths(tables["bhar_paths"], tables["event_study"], out / "fig1_after_a_cluster.png")
    split = pd.Timestamp(config.split_date)
    plots.by_year(
        tables["by_year"], config.headline_horizon, split.year + 0.5, out / "fig2_by_year.png"
    )
    plots.coverage(tables["sample"], out / "fig3_coverage.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="synthetic data, no downloads")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--source", choices=["layline", "edgar"], default="layline")
    parser.add_argument("--no-cache", action="store_true", help="re-read the raw zips")
    args = parser.parse_args()

    started = time.time()
    config = pipeline.Config()
    if args.fast:
        out = args.out or ROOT / "results_fast"
        inputs = synthetic.make(seed=0)
    else:
        out = args.out or ROOT / "results"
        inputs = dataset.load_inputs(args.data_dir, not args.no_cache, args.source)
    results = pipeline.run(inputs.trades, inputs.panel, inputs.cik_map, inputs.riskfree, config)
    write(results, inputs, config, out)
    if not args.fast:
        manifest(args.data_dir, args.source).to_csv(
            out / "data_manifest.csv", index=False, lineterminator="\n"
        )

    study = results.tables["event_study"]
    head = study[
        (study["group"] == "all clusters")
        & (study["benchmark"] == "universe")
        & (study["horizon_days"] == config.headline_horizon)
    ]
    print(head[["period", "events", "mean_bhar", "t", "median_bhar"]].to_string(index=False))
    print(f"wrote {out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
