"""Cluster detection: several insiders buying the same stock within a window.

A cluster event is dated by what the public could see. On a day ``d`` when a
purchase becomes public, count the distinct insiders with a purchase that

* is public on or before ``d``, and
* was traded no more than ``window_days`` before ``d``.

If the count reaches ``min_insiders`` there is an event on ``d``, unless the
same issuer already had one in the previous ``cooldown_days``. The cooldown keeps
one burst of buying from being counted many times and keeps event windows for
one stock from overlapping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["issuer_cik", "event_date", "n_insiders", "value", "median_price", "insiders"]


def detect_clusters(
    buys: pd.DataFrame,
    min_insiders: int = 3,
    window_days: int = 30,
    cooldown_days: int = 182,
) -> pd.DataFrame:
    """Find cluster events in a table of purchases.

    ``buys`` needs ``issuer_cik``, ``insider_cik``, ``trade_date``, ``public_date``,
    ``shares`` and ``value``. Returns one row per event with the insiders involved.
    """
    if min_insiders < 1 or window_days < 0 or cooldown_days < 0:
        raise ValueError("min_insiders must be >= 1 and the day counts non-negative")
    records: list[dict[str, object]] = []
    ordered = buys.sort_values(["issuer_cik", "public_date", "trade_date", "insider_cik"])
    for issuer, grp in ordered.groupby("issuer_cik", sort=True):
        if grp["insider_cik"].nunique() < min_insiders:
            continue
        public = grp["public_date"].to_numpy(dtype="datetime64[D]")
        traded = grp["trade_date"].to_numpy(dtype="datetime64[D]")
        insider = grp["insider_cik"].to_numpy(dtype=np.int64)
        value = grp["value"].to_numpy(dtype=np.float64)
        shares = grp["shares"].to_numpy(dtype=np.float64)
        last_event: np.datetime64 | None = None
        for day in np.unique(public):
            if last_event is not None and (day - last_event).astype(int) <= cooldown_days:
                continue
            seen = (public <= day) & (traded >= day - np.timedelta64(window_days, "D"))
            who = np.unique(insider[seen])
            if len(who) < min_insiders:
                continue
            records.append(
                {
                    "issuer_cik": int(str(issuer)),
                    "event_date": pd.Timestamp(day),
                    "n_insiders": len(who),
                    "value": float(value[seen].sum()),
                    "median_price": float(np.median(value[seen] / shares[seen])),
                    "insiders": tuple(int(w) for w in who),
                }
            )
            last_event = day
    return pd.DataFrame.from_records(records, columns=EVENT_COLUMNS)


def count_labels(events: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Add ``n_opportunistic`` and ``n_routine``: how many cluster insiders carry each label.

    ``labels`` is the output of :func:`quorum.routine.classify_insiders`. The label
    used is the one for the calendar year of the event.
    """
    lookup = {
        (int(i), int(p), int(y)): str(lab)
        for i, p, y, lab in labels[["issuer_cik", "insider_cik", "year", "label"]].itertuples(
            index=False
        )
    }
    n_opp: list[int] = []
    n_rou: list[int] = []
    for issuer, date, insiders in events[["issuer_cik", "event_date", "insiders"]].itertuples(
        index=False
    ):
        found = [lookup.get((int(issuer), int(p), int(date.year))) for p in insiders]
        n_opp.append(sum(f == "opportunistic" for f in found))
        n_rou.append(sum(f == "routine" for f in found))
    return events.assign(n_opportunistic=n_opp, n_routine=n_rou)
