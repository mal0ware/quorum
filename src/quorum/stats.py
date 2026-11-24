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


def ols_grouped(y: Array, x: Array, groups: NDArray[np.int64], lags: int = 0) -> Fit:
    """Least squares with scores summed within groups before the covariance is formed.

    ``groups`` is an integer time index such as a month number. The score
    ``u_i = x_i e_i`` is summed inside each group, ``s_g = sum_{i in g} u_i``, and

        S = sum_g s_g s_g' + sum_{j=1..L} (1 - j / (L + 1)) sum_g (s_g s_{g-j}' + s_{g-j} s_g')

    where ``g - j`` is the group ``j`` steps earlier on the integer index (a group
    with no observations has a zero score). The covariance is ``(X'X)^-1 S (X'X)^-1``.

    With ``lags = 0`` this is the usual cluster-robust estimator, clustered on
    ``groups``. With ``lags > 0`` observations in nearby groups may be correlated
    too, which is what overlapping return windows produce. Rows where ``y`` is not
    finite are dropped. No small-sample correction is applied.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    good = np.isfinite(y)
    y, x, groups = y[good], x[good], np.asarray(groups, dtype=np.int64)[good]
    n, k = x.shape
    nan = np.full(k, np.nan)
    if n <= k:
        return Fit(coef=nan, se=nan.copy(), nobs=n)
    xtx = x.T @ x
    if np.linalg.matrix_rank(xtx) < k:
        return Fit(coef=nan, se=nan.copy(), nobs=n)
    xtx_inv = np.linalg.inv(xtx)
    coef = xtx_inv @ x.T @ y
    u = x * (y - x @ coef)[:, None]
    first = int(groups.min())
    scores = np.zeros((int(groups.max()) - first + 1, k))
    np.add.at(scores, groups - first, u)
    s = scores.T @ scores
    for j in range(1, min(lags, len(scores) - 1) + 1):
        g = scores[j:].T @ scores[:-j]
        s += (1.0 - j / (lags + 1.0)) * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    return Fit(coef=coef, se=np.sqrt(np.maximum(np.diag(cov), 0.0)), nobs=n)


def block_bootstrap_medians(
    x: Array, groups: NDArray[np.int64], block: int, draws: int, seed: int
) -> Array:
    """Medians of ``x`` over moving-block bootstrap samples of whole groups.

    ``groups`` is an integer time index (months). Each draw picks blocks of
    ``block`` consecutive groups with replacement until the span of the data is
    covered, and takes every observation in the picked groups. Events in nearby
    months stay together, so overlap between their return windows is kept.
    """
    x = np.asarray(x, dtype=np.float64)
    good = np.isfinite(x)
    x, groups = x[good], np.asarray(groups, dtype=np.int64)[good]
    if len(x) == 0:
        return np.full(draws, np.nan)
    order = np.argsort(groups, kind="stable")
    x, groups = x[order], groups[order]
    first = int(groups[0])
    span = int(groups[-1]) - first + 1
    start = np.searchsorted(groups, first + np.arange(span + 1), side="left")
    block = max(1, min(block, span))
    n_blocks = -(-span // block)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, span - block + 1, size=(draws, n_blocks))
    out = np.full(draws, np.nan)
    for d in range(draws):
        parts = [x[start[p] : start[p + block]] for p in picks[d]]
        sample = np.concatenate(parts)
        if len(sample):
            out[d] = np.median(sample)
    return out
