"""Least squares with Newey-West standard errors.

Newey and West (1987), "A Simple, Positive Semi-definite, Heteroskedasticity and
Autocorrelation Consistent Covariance Matrix", Econometrica 55(3). The long-run
covariance of the moment vector ``u_t = x_t e_t`` is estimated as

    S = G_0 + sum_{j=1..L} (1 - j / (L + 1)) (G_j + G_j')
    G_j = sum_t u_t u_{t-j}'

and the coefficient covariance is ``(X'X)^-1 S (X'X)^-1``. No small-sample
correction is applied. With ``L = 0`` this is White's estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Fit:
    """Coefficients, their standard errors and the number of observations."""

    coef: Array
    se: Array
    nobs: int

    @property
    def t(self) -> Array:
        """t-statistics."""
        with np.errstate(divide="ignore", invalid="ignore"):
            out: Array = self.coef / self.se
        return out


def ols_nw(y: Array, x: Array, lags: int = 0) -> Fit:
    """Regress ``y`` on the columns of ``x`` (include your own constant)."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    t, k = x.shape
    if t <= k:
        nan = np.full(k, np.nan)
        return Fit(coef=nan, se=nan.copy(), nobs=t)
    xtx_inv = np.linalg.inv(x.T @ x)
    coef = xtx_inv @ x.T @ y
    u = x * (y - x @ coef)[:, None]
    s = u.T @ u
    for j in range(1, min(lags, t - 1) + 1):
        g = u[j:].T @ u[:-j]
        s += (1.0 - j / (lags + 1.0)) * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    return Fit(coef=coef, se=np.sqrt(np.diag(cov)), nobs=t)


def mean_nw(x: Array, lags: int = 0) -> Fit:
    """Mean of ``x`` with a Newey-West standard error. NaNs are dropped."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return ols_nw(x, np.ones((len(x), 1)), lags)
