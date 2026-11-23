"""Calendar-time portfolios, rebalanced monthly.

Following Cohen, Malloy and Pomorski (2012, Table IV): a firm is in the portfolio
in month m+1 if it had a signal in month m. Here a signal is dated by the month
its filing became public, and the firm stays for ``hold`` months. Portfolios are
equal weighted over the firms that are liquid in the holding month.

Abnormal return is measured two ways:

* excess over the equal-weight return of all liquid stocks in the panel, and
* the intercept of ``R_p - R_f = alpha + beta (R_m - R_f) + e`` (Jensen 1968),
  with a broad index fund as the market.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quorum.prices import Panel
from quorum.stats import mean_nw, ols_nw

Float = NDArray[np.float64]
Bool = NDArray[np.bool_]

NW_LAGS = 3


def monthly_returns(panel: Panel) -> Float:
    """Compound daily returns to months. NaN where a stock has no return in the month."""
    n_months = len(panel.months)
    logs = np.log1p(np.nan_to_num(panel.ret.astype(np.float64), nan=0.0))
    has = np.isfinite(panel.ret)
    starts = np.flatnonzero(np.diff(panel.month_of_day, prepend=-1) != 0)
    total = np.add.reduceat(logs, starts, axis=0)
    count = np.add.reduceat(has.astype(np.int64), starts, axis=0)
    out: Float = np.expm1(total)
    out[count == 0] = np.nan
    assert out.shape[0] == n_months
    return out


def signal_matrix(panel: Panel, months: NDArray[np.int64], cols: NDArray[np.int64]) -> Bool:
    """Boolean [months, stocks] matrix with True at the given positions."""
    out = np.zeros((len(panel.months), len(panel.symbols)), dtype=np.bool_)
    ok = (months >= 0) & (months < out.shape[0]) & (cols >= 0)
    out[months[ok], cols[ok]] = True
    return out


def universe_return(monthly: Float, liquid: Bool) -> Float:
    """Equal-weight monthly return of all liquid stocks."""
    member = liquid & np.isfinite(monthly)
    count = member.sum(axis=1)
    total = np.where(member, monthly, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out: Float = np.where(count > 0, total / count, np.nan)
    return out


def calendar_time(signal: Bool, monthly: Float, liquid: Bool, hold: int) -> tuple[Float, Float]:
    """Monthly return and number of holdings of the portfolio that follows ``signal``.

    A stock is held in month m if it had a signal in any of months m-hold..m-1.
    """
    if hold < 1:
        raise ValueError("hold must be at least one month")
    held = np.zeros_like(signal)
    for lag in range(1, hold + 1):
        held[lag:] |= signal[:-lag]
    member = held & liquid & np.isfinite(monthly)
    count = member.sum(axis=1).astype(np.float64)
    total = np.where(member, monthly, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret: Float = np.where(count > 0, total / count, np.nan)
    return ret, count


@dataclass(frozen=True)
class PortfolioStats:
    """Summary of one portfolio over one period. Returns are per month."""

    months: int
    mean_holdings: float
    mean_return: float
    excess_universe: float
    excess_universe_t: float
    alpha: float
    alpha_t: float
    beta: float


def summarise(
    ret: Float, count: Float, universe: Float, market: Float, riskfree: Float, period: Bool
) -> PortfolioStats:
    """Statistics for the months in ``period`` where the portfolio holds something."""
    use = period & np.isfinite(ret) & np.isfinite(universe) & np.isfinite(market)
    r, u, m, f = ret[use], universe[use], market[use], riskfree[use]
    if len(r) < 3:
        nan = float("nan")
        return PortfolioStats(len(r), nan, nan, nan, nan, nan, nan, nan)
    excess = mean_nw(r - u, NW_LAGS)
    capm = ols_nw(r - f, np.column_stack([np.ones(len(r)), m - f]), NW_LAGS)
    return PortfolioStats(
        months=len(r),
        mean_holdings=float(count[use].mean()),
        mean_return=float(r.mean()),
        excess_universe=float(excess.coef[0]),
        excess_universe_t=float(excess.t[0]),
        alpha=float(capm.coef[0]),
        alpha_t=float(capm.t[0]),
        beta=float(capm.coef[1]),
    )
