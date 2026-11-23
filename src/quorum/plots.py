"""Figures. Each takes result tables and writes one PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from quorum.events import UNIVERSE

PRE_COLOUR = "#1f4e79"
POST_COLOUR = "#c0504d"
GREY = "#7f7f7f"
LABELS = {"pre": "clusters before July 2012", "post": "clusters from July 2012"}


def _save(fig: Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def bhar_paths(paths: pd.DataFrame, study: pd.DataFrame, out: Path) -> None:
    """Mean abnormal return against matched survivors, by days since entry."""
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    table = study[(study["group"] == "all clusters") & (study["benchmark"] == UNIVERSE)]
    for period, colour in (("pre", PRE_COLOUR), ("post", POST_COLOUR)):
        line = paths[(paths["benchmark"] == UNIVERSE) & (paths["period"] == period)]
        marks = table[table["period"] == period]
        n = int(marks["events"].max()) if len(marks) else 0
        ax.plot(
            line["day"],
            100 * line["mean_bhar"],
            color=colour,
            lw=2,
            label=f"{LABELS[period]} (n = {n:,})",
        )
        ax.errorbar(
            marks["horizon_days"],
            100 * marks["mean_bhar"],
            yerr=196 * marks["se"],
            fmt="o",
            color=colour,
            capsize=3,
            lw=1,
        )
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel(
        "Trading days after the entry close (entry is the day after the filing is public)"
    )
    ax.set_ylabel("Mean abnormal return, %")
    ax.set_title(
        "After three or more insiders buy: return over an equal-weight\n"
        "portfolio of the other stocks in the price file",
        fontsize=11,
    )
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.99,
        0.02,
        "Bars: 95% interval, clustered by entry month",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=GREY,
    )
    _save(fig, out)


def by_year(table: pd.DataFrame, horizon: int, split_year: float, out: Path) -> None:
    """Mean and median abnormal return by event year."""
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    colours = [PRE_COLOUR if y < split_year else POST_COLOUR for y in table["year"]]
    ax.bar(table["year"], 100 * table["mean_bhar_universe"], color=colours, width=0.7, label="mean")
    ax.plot(
        table["year"], 100 * table["median_bhar_universe"], "o", color="black", ms=4, label="median"
    )
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel("Year the cluster became public")
    ax.set_ylabel(f"Abnormal return over {horizon} trading days, %")
    ax.set_title("Cluster events by year, against matched survivors", fontsize=11)
    ax.legend(frameon=False)
    _save(fig, out)


def coverage(sample: pd.DataFrame, out: Path) -> None:
    """How many clusters the price file can follow, by year."""
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(
        sample["year"],
        sample["clusters_found"],
        color="#d9d9d9",
        width=0.7,
        label="clusters found in the filings",
    )
    ax.bar(
        sample["year"],
        sample["clusters_priced"],
        color=PRE_COLOUR,
        width=0.7,
        label="clusters with usable prices",
    )
    ax.set_xlabel("Year the cluster became public")
    ax.set_ylabel("Cluster events")
    ax.set_title(
        "Most early clusters cannot be priced: the firms are no longer listed", fontsize=11
    )
    ax.legend(frameon=False)
    _save(fig, out)
