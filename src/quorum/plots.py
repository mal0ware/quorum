"""Figures. Each takes result tables and writes one PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from quorum.events import CONTROLS, UNIVERSE

PRE_COLOUR = "#1f4e79"
POST_COLOUR = "#c0504d"
GREY = "#7f7f7f"
LABELS = {"pre": "clusters before July 2012", "post": "clusters from July 2012"}


def _save(fig: Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def estimates(robustness: pd.DataFrame, calendar: pd.DataFrame, horizon: int, out: Path) -> None:
    """The headline estimate and its 95% interval under each sample and estimator."""
    picks: list[tuple[str, str, str, str]] = [
        ("Mean, all priced events", "all priced events", "mean", UNIVERSE),
        ("Mean against matched controls", "all priced events", "mean", CONTROLS),
        ("Mean, excluding 2009", "excluding 2009", "mean", UNIVERSE),
        (
            "Mean, winsorised at 1%",
            "all priced events",
            "mean, winsorised at 1% and 99%",
            UNIVERSE,
        ),
        (
            "Mean, winsorised at 5%",
            "all priced events",
            "mean, winsorised at 5% and 95%",
            UNIVERSE,
        ),
        ("Median", "all priced events", "median", UNIVERSE),
        ("Mean, reported price reconciled", "reported price reconciled only", "mean", UNIVERSE),
        ("Mean, 2007 to 2014 only", "2007 to 2014 only", "mean", UNIVERSE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharex=True, sharey=True)
    names: list[str] = []
    for ax, period, colour in zip(axes, ("pre", "post"), (PRE_COLOUR, POST_COLOUR), strict=True):
        names = []
        for label, sample, estimator, bench in picks:
            row = robustness[
                (robustness["sample"] == sample)
                & (robustness["estimator"] == estimator)
                & (robustness["benchmark"] == bench)
                & (robustness["period"] == period)
            ]
            if len(row) != 1:
                continue
            r = row.iloc[0]
            y = len(names)
            names.append(label)
            ax.plot([100 * r["ci_low"], 100 * r["ci_high"]], [y, y], color=colour, lw=1.6)
            ax.plot(100 * r["estimate"], y, "o", color=colour, ms=5)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_title(LABELS[period], fontsize=10)
        ax.set_xlabel(f"Abnormal return over {horizon} trading days, %")
        cal = calendar[
            (calendar["portfolio"] == "cluster of buyers") & (calendar["period"] == period)
        ]
        if len(cal) == 1:
            c = cal.iloc[0]
            ax.text(
                0.98,
                0.03,
                f"Monthly portfolio: {100 * c['excess_universe']:.2f}% a month"
                f" (t {c['excess_universe_t']:.1f})",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color=GREY,
            )
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_ylim(len(names) + 0.2, -0.6)
    fig.suptitle(
        "After three or more insiders buy: estimates with 95% intervals that allow for"
        " overlapping windows",
        fontsize=10.5,
    )
    _save(fig, out)


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
            yerr=196 * marks["se_overlap"],
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
        "Bars: 95% interval, allowing for overlapping windows",
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


def survivorship(bounds: pd.DataFrame, out: Path) -> None:
    """The all-event mean as the share of missing events assumed to have failed rises."""
    grid = bounds[bounds["other_missing_events_earn"] == "priced mean"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    styles = ["-", "--", "-.", ":"]
    for ax, column, colour, title in zip(
        axes,
        ("pre_mean", "post_mean"),
        (PRE_COLOUR, POST_COLOUR),
        (LABELS["pre"], LABELS["post"]),
        strict=True,
    ):
        for style, (loss, lines) in zip(
            styles, grid.groupby("loss_of_a_failed_event", sort=False), strict=False
        ):
            lines = lines[lines["share_of_missing_events_failed"] <= 0.2]
            ax.plot(
                100 * lines["share_of_missing_events_failed"],
                100 * lines[column],
                style,
                color=colour,
                lw=1.6,
                label=f"a failed event loses {-100 * float(str(loss)):.0f}%",
            )
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Share of unpriced clusters assumed to have failed, %")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Implied mean abnormal return, all clusters, %")
    fig.suptitle(
        "How few failures among the unpriced clusters would erase the survivors' mean",
        fontsize=10.5,
    )
    _save(fig, out)
