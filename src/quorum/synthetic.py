"""A small seeded data set with a planted effect, for tests and ``--fast``.

Stocks follow a one-factor model. Each stock has eight insiders: two buy every
March and sell every September (routine), the rest buy at random. Now and then
several insiders buy within a few days. Before the split date, a planted cluster
is followed by a fixed daily drift; after it, by nothing. The pipeline should
find the drift before the split and none after.

Nothing here is a finding. The numbers only show that the code runs and that it
can recover an effect of known size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quorum import prices
from quorum.clean import TRADE_COLUMNS
from quorum.dataset import MIN_DOLLAR_VOLUME, Inputs

DRIFT_PER_DAY = 0.0015
DRIFT_DAYS = 63
SPLIT_DATE = "2012-07-01"
N_INSIDERS = 8
N_ILLIQUID = 10


def make(
    seed: int = 0, n_stocks: int = 160, start: str = "2004-01-01", end: str = "2016-12-30"
) -> Inputs:
    """Build synthetic pipeline inputs."""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start, end, name="date")
    n_days = len(cal)
    split = int(cal.searchsorted(pd.Timestamp(SPLIT_DATE)))
    years = sorted(set(cal.year))
    boost = np.zeros((n_days, n_stocks))
    rows: list[tuple[int, int, str, int, int]] = []  # issuer, insider, side, trade day, public day

    def add(issuer: int, insider: int, side: str, day: int) -> int:
        public = min(day + int(rng.integers(1, 3)), n_days - 1)
        rows.append((issuer, insider, side, day, public))
        return public

    for i in range(n_stocks):
        issuer = 1000 + i
        insiders = 50_000 + 10 * i + np.arange(N_INSIDERS)
        for year in years:
            march = int(cal.searchsorted(pd.Timestamp(year=year, month=3, day=10)))
            september = int(cal.searchsorted(pd.Timestamp(year=year, month=9, day=10)))
            for person in insiders[:2]:
                if march < n_days:
                    add(issuer, int(person), "P", march)
                if september < n_days:
                    add(issuer, int(person), "S", september)
        for person in insiders[2:]:
            for day in np.flatnonzero(rng.random(n_days) < 0.001):
                add(issuer, int(person), "P" if rng.random() < 0.5 else "S", int(day))
        for first in np.flatnonzero(rng.random(n_days) < 1.0 / 500.0):
            buyers = rng.choice(insiders[2:], size=int(rng.integers(3, 6)), replace=False)
            public_days = sorted(
                add(issuer, int(p), "P", min(int(first) + int(rng.integers(0, 8)), n_days - 1))
                for p in buyers
            )
            third = public_days[2]
            if third < split:
                boost[third + 2 : third + 2 + DRIFT_DAYS, i] += DRIFT_PER_DAY

    trades = pd.DataFrame(rows, columns=["issuer_cik", "insider_cik", "side", "t", "p"])
    trades = trades.assign(
        trade_date=cal[trades["t"].to_numpy(dtype=np.int64)],
        public_date=cal[trades["p"].to_numpy(dtype=np.int64)],
        shares=1000.0,
        value=20_000.0,
    )
    keys = ["issuer_cik", "insider_cik", "side", "trade_date", "public_date"]
    trades = trades.groupby(keys, as_index=False, sort=True)[["shares", "value"]].sum()

    market = rng.normal(0.0003, 0.01, n_days)
    beta = rng.uniform(0.6, 1.4, n_stocks)
    ret = beta[None, :] * market[:, None] + rng.normal(0.0, 0.02, (n_days, n_stocks)) + boost
    names = [f"S{i:03d}" for i in range(n_stocks)]
    frames = []
    dividends = []
    for i, name in enumerate(names):
        pays = i % 2 == 0
        close = np.empty(n_days)
        level = 20.0
        for t in range(n_days):
            cash = 0.05 if pays and t % 63 == 0 and t > 0 else 0.0
            level = level * (1.0 + ret[t, i]) - cash
            close[t] = level
            if cash:
                dividends.append((name, cal[t], cash))
        volume = 50.0 if i >= n_stocks - N_ILLIQUID else 100_000.0
        frames.append(pd.DataFrame({"symbol": name, "date": cal, "close": close, "volume": volume}))
    for name, scale in (("SPY", 1.0), ("IWM", 1.1)):
        index = 100.0 * np.cumprod(1.0 + scale * market)
        frames.append(pd.DataFrame({"symbol": name, "date": cal, "close": index, "volume": 1e7}))
    px = pd.concat(frames, ignore_index=True)
    dv = pd.DataFrame(dividends, columns=["symbol", "date", "amount"])

    panel = prices.build_panel(px, dv, cal, MIN_DOLLAR_VOLUME)
    cik_map = pd.Series(names, index=pd.Index(1000 + np.arange(n_stocks), name="cik"))
    log = pd.DataFrame({"rule": ["synthetic trades, no cleaning"], "rows_left": [len(trades)]})
    return Inputs(
        trades=trades[TRADE_COLUMNS],
        cleaning_log=log,
        panel=panel,
        cik_map=cik_map,
        riskfree=np.full(len(panel.months), 0.0002),
    )
