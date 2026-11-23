"""The routine-trader rule against the definition in Cohen, Malloy and Pomorski (2012).

Section I of the paper: an insider needs at least one trade in each of the three
preceding years to be classified; she is routine if she "placed a trade in the
same calendar month for at least three consecutive years", and opportunistic
otherwise.
"""

from __future__ import annotations

import pandas as pd

from quorum.routine import classify_insiders, label_trades


def trades(rows: list[tuple[int, str]], issuer: int = 1, lag_days: int = 2) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["insider_cik", "trade_date"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["public_date"] = frame["trade_date"] + pd.Timedelta(days=lag_days)
    frame["issuer_cik"] = issuer
    return frame


def label_of(frame: pd.DataFrame, insider: int, year: int) -> str | None:
    out = classify_insiders(frame)
    hit = out[(out["insider_cik"] == insider) & (out["year"] == year)]
    return None if hit.empty else str(hit["label"].iloc[0])


def test_same_month_three_years_running_is_routine() -> None:
    frame = trades([(7, "2010-03-15"), (7, "2011-03-02"), (7, "2012-03-28")])
    assert label_of(frame, 7, 2013) == "routine"


def test_three_years_of_trades_in_different_months_is_opportunistic() -> None:
    frame = trades([(7, "2010-03-15"), (7, "2011-06-02"), (7, "2012-11-28")])
    assert label_of(frame, 7, 2013) == "opportunistic"


def test_two_years_in_the_same_month_is_not_enough_to_be_routine() -> None:
    frame = trades([(7, "2010-03-15"), (7, "2011-03-02"), (7, "2012-04-02")])
    assert label_of(frame, 7, 2013) == "opportunistic"


def test_a_missing_year_means_no_classification() -> None:
    frame = trades([(7, "2010-03-15"), (7, "2012-03-28")])
    assert label_of(frame, 7, 2013) is None


def test_extra_trades_in_other_months_do_not_undo_a_routine_month() -> None:
    frame = trades(
        [
            (7, "2010-03-15"),
            (7, "2010-08-01"),
            (7, "2011-03-02"),
            (7, "2012-03-28"),
            (7, "2012-12-01"),
        ]
    )
    assert label_of(frame, 7, 2013) == "routine"


def test_only_the_three_preceding_years_count() -> None:
    # Routine on 2010-2012, so routine in 2013. In 2014 the window is 2011-2013 and
    # the 2013 trade is in another month.
    frame = trades([(7, "2010-03-15"), (7, "2011-03-02"), (7, "2012-03-28"), (7, "2013-07-09")])
    assert label_of(frame, 7, 2013) == "routine"
    assert label_of(frame, 7, 2014) == "opportunistic"
    assert label_of(frame, 7, 2015) is None


def test_the_label_is_per_insider_and_firm() -> None:
    a = trades([(7, "2010-03-15"), (7, "2011-03-02"), (7, "2012-03-28")], issuer=1)
    b = trades([(7, "2010-01-15"), (7, "2011-06-02"), (7, "2012-09-28")], issuer=2)
    out = classify_insiders(pd.concat([a, b], ignore_index=True))
    by_issuer = out[out["year"] == 2013].set_index("issuer_cik")["label"]
    assert by_issuer[1] == "routine"
    assert by_issuer[2] == "opportunistic"


def test_a_trade_not_yet_public_at_new_year_is_not_used() -> None:
    # The December 2012 trade is filed in January 2013, after the 2013 labels are set.
    frame = trades([(7, "2010-12-15"), (7, "2011-12-02"), (7, "2012-12-28")], lag_days=10)
    assert label_of(frame, 7, 2013) is None
    assert label_of(frame, 7, 2014) is None  # 2013 has no trade


def test_label_trades_marks_unclassified() -> None:
    frame = trades([(7, "2010-03-15"), (7, "2011-03-02"), (7, "2012-03-28"), (7, "2013-05-01")])
    labels = classify_insiders(frame)
    assert label_trades(frame, labels).tolist() == [
        "unclassified",
        "unclassified",
        "unclassified",
        "routine",
    ]
