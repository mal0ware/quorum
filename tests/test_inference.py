"""Standard errors that allow for overlapping return windows, and the bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from quorum import pipeline
from quorum.stats import block_bootstrap_medians, ols_grouped


def test_grouped_mean_with_no_lags_matches_a_hand_calculation() -> None:
    x = np.array([1.0, 3.0, 2.0, 6.0])
    fit = ols_grouped(x, np.ones((4, 1)), np.array([0, 0, 1, 1]), 0)
    # mean 3; group sums of deviations: (-2 + 0) and (-1 + 3); se = sqrt(4 + 4) / 4
    assert fit.coef[0] == pytest.approx(3.0)
    assert fit.se[0] == pytest.approx(np.sqrt(8.0) / 4.0)
    # One observation per group is the usual sandwich: sqrt(sum e^2) / n.
    single = ols_grouped(x, np.ones((4, 1)), np.arange(4), 0)
    assert single.se[0] == pytest.approx(np.sqrt(((x - 3.0) ** 2).sum()) / 4.0)


def test_lags_add_the_weighted_cross_products_of_neighbouring_groups() -> None:
    x = np.array([1.0, 3.0, 2.0, 6.0])
    groups = np.array([0, 0, 1, 1])
    fit = ols_grouped(x, np.ones((4, 1)), groups, 1)
    # scores s = (-2, 2); S = 4 + 4 + (1 - 1/2) * 2 * (2 * -2) = 4
    assert fit.se[0] == pytest.approx(np.sqrt(4.0) / 4.0)


def test_an_empty_group_between_two_groups_has_a_zero_score() -> None:
    x = np.array([1.0, 3.0, 2.0, 6.0])
    apart = np.array([0, 0, 2, 2])
    # The groups are two steps apart, so one lag adds nothing and two lags add
    # (1 - 2/3) * 2 * (2 * -2).
    one = ols_grouped(x, np.ones((4, 1)), apart, 1)
    two = ols_grouped(x, np.ones((4, 1)), apart, 2)
    assert one.se[0] == pytest.approx(np.sqrt(8.0) / 4.0)
    assert two.se[0] == pytest.approx(np.sqrt(8.0 - 8.0 / 3.0) / 4.0)


def test_missing_values_are_dropped() -> None:
    x = np.array([1.0, np.nan, 3.0, 2.0, 6.0])
    fit = ols_grouped(x, np.ones((5, 1)), np.array([0, 0, 0, 1, 1]), 0)
    assert fit.nobs == 4
    assert fit.coef[0] == pytest.approx(3.0)


def test_difference_regression_gives_the_difference_of_means() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    month = np.repeat(np.arange(50), 4)
    post = month >= 30
    d = pipeline.difference_tests(x, post, month, 3)
    assert d["pre_mean"] == pytest.approx(x[~post].mean())
    assert d["post_mean"] == pytest.approx(x[post].mean())
    assert d["post_minus_pre"] == pytest.approx(x[post].mean() - x[~post].mean())
    assert d["se_overlap"] > 0


def test_overlap_lags_cover_twice_the_window() -> None:
    assert pipeline.overlap_lags(21) == 2
    assert pipeline.overlap_lags(63) == 6
    assert pipeline.overlap_lags(126) == 12
    assert pipeline.overlap_lags(22) == 4


def overlapping_sample(rng: np.random.Generator, months: int, per_month: int, span: int):  # type: ignore[no-untyped-def]
    """Events whose value is the sum of the shocks of the next ``span`` months, plus noise."""
    shocks = rng.normal(0.0, 1.0, months + span)
    window = np.array([shocks[m : m + span].sum() for m in range(months)])
    month = np.repeat(np.arange(months), per_month)
    return window[month] + rng.normal(0.0, 1.0, len(month)), month


def test_month_clustering_rejects_too_often_when_windows_overlap_and_the_fix_does_not() -> None:
    # True mean is zero. Each event's value shares shocks with the next two months,
    # as a three-month return window does.
    rng = np.random.default_rng(1)
    reject_month = reject_overlap = 0
    trials = 400
    for _ in range(trials):
        x, month = overlapping_sample(rng, months=120, per_month=5, span=3)
        t = pipeline.mean_tests(x, month, lags=6)
        reject_month += abs(t["t_month_cluster"]) > 1.96
        reject_overlap += abs(t["t_overlap"]) > 1.96
        assert t["se_overlap"] > 0
    # Nominal rate is 5%. Month clustering ignores the shared shocks.
    assert reject_month / trials > 0.20
    assert reject_overlap / trials < 0.12
    assert reject_overlap < reject_month


def test_bootstrap_is_seeded_and_keeps_whole_months() -> None:
    rng = np.random.default_rng(2)
    month = np.repeat(np.arange(40), 3)
    x = rng.normal(size=len(month))
    a = block_bootstrap_medians(x, month, block=6, draws=200, seed=5)
    b = block_bootstrap_medians(x, month, block=6, draws=200, seed=5)
    c = block_bootstrap_medians(x, month, block=6, draws=200, seed=6)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.isfinite(a).all()
    assert x.min() <= a.min() and a.max() <= x.max()


def test_bootstrap_of_a_constant_is_that_constant() -> None:
    out = block_bootstrap_medians(np.full(30, 0.25), np.repeat(np.arange(10), 3), 4, 50, 0)
    np.testing.assert_allclose(out, 0.25)
    empty = block_bootstrap_medians(np.array([np.nan]), np.array([0]), 4, 5, 0)
    assert np.isnan(empty).all()


def test_bootstrap_interval_widens_when_months_move_together() -> None:
    rng = np.random.default_rng(3)
    month = np.repeat(np.arange(60), 10)
    independent = rng.normal(size=len(month))
    common = np.repeat(rng.normal(size=60), 10) + 0.1 * rng.normal(size=len(month))

    def width(x: np.ndarray) -> float:
        draws = block_bootstrap_medians(x, month, 6, 400, 0)
        return float(np.quantile(draws, 0.975) - np.quantile(draws, 0.025))

    assert width(common) > 2 * width(independent)
