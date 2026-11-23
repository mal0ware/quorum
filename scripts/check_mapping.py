"""Check that issuers were joined to the right price series.

    uv run python scripts/check_mapping.py      # after reproduce.py

Filings are joined to prices by SEC CIK, not by ticker. This compares two numbers
that come from different sources: the median price the insiders reported paying,
and the close in the price file on the day the cluster became public (or the last
close before it). If the join is right they should be near each other. The price
file is split-adjusted, so events before a later split will not match; the table
is by year so that shows.

Writes results/mapping_check.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quorum import dataset, prices

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    events = pd.read_csv(ROOT / "results" / "cluster_events.csv", parse_dates=["event_date"])
    px, _ = prices.load_long(
        ROOT / "data" / dataset.PRICES_DIR, set(events["symbol"]), dataset.PRICE_START
    )
    px = px[px["close"] > 0].sort_values("date")
    merged = pd.merge_asof(
        events.sort_values("event_date"),
        px[["symbol", "date", "close"]],
        left_on="event_date",
        right_on="date",
        by="symbol",
        tolerance=pd.Timedelta(days=7),
    )
    ratio = merged["median_price"] / merged["close"]
    off = np.abs(np.log(ratio))
    table = (
        pd.DataFrame(
            {
                "year": merged["event_date"].dt.year,
                "events": 1,
                "with_a_close": ratio.notna(),
                "within_10_percent": off < np.log(1.10),
                "within_25_percent": off < np.log(1.25),
            }
        )
        .groupby("year")
        .sum()
    )
    table["share_within_25_percent"] = table["within_25_percent"] / table["with_a_close"]
    table.to_csv(ROOT / "results" / "mapping_check.csv", float_format="%.4g", lineterminator="\n")
    print(table.to_string())
    recent = table[table.index >= 2021]
    print(
        "2021 on: within 25% in",
        f"{recent['within_25_percent'].sum() / recent['with_a_close'].sum():.1%} of events",
    )


if __name__ == "__main__":
    main()
