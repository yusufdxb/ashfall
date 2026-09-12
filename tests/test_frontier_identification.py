"""A frontier threshold is reported only where the data identify one."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ashfall.frontier import (
    CENSORING_KINDS,
    FrontierEstimator,
    FrontierPoint,
    RobustnessMargin,
    SeverityMetric,
    frontier_shift,
    frontier_shift_with_reason,
)


def estimator(q=0.5):
    return FrontierEstimator(SeverityMetric("slip", "friction", "coefficient", -1, 1), q)


def points(counts, replicates=10):
    return [
        FrontierPoint(str(i), float(i), f, replicates, f / replicates, 0.2)
        for i, f in enumerate(counts)
    ]


def _legacy_estimate(counts, q=0.5):
    """The estimator as it was: a point from the left edge of a pooled block.

    Reproduced here as the mutant the schema must refuse. It returned
    ``xs[idx]`` when the first fitted value at or above ``q`` was the first
    element, and labelled every fitted crossing ``observed_crossing``.
    """
    xs = np.arange(len(counts), dtype=float)
    blocks = []
    for i, f in enumerate(counts):
        blocks.append([i, i, f, 10])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            b = blocks.pop()
            a = blocks.pop()
            blocks.append([a[0], b[1], a[2] + b[2], a[3] + b[3]])
    fitted = np.empty(len(xs))
    for first, last, f, n in blocks:
        fitted[first : last + 1] = f / n
    idx = int(np.flatnonzero(fitted >= q)[0])
    threshold = (
        float(xs[idx])
        if idx == 0
        else float(np.interp(q, fitted[idx - 1 : idx + 1], xs[idx - 1 : idx + 1]))
    )
    return threshold, "observed_crossing"


class TestIdentification:
    def test_interpolated_crossing_is_named_as_fitted(self):
        margin = estimator().estimate(points([0, 2, 8, 10]))
        assert margin.censoring == "interpolated_crossing"
        assert margin.threshold == pytest.approx(1.5)
        assert margin.threshold_interval is None
        assert margin.identified and margin.monotonic_assumption
        assert "observed_crossing" not in CENSORING_KINDS

    @pytest.mark.guard
    def test_plateau_at_quantile_is_an_interval_not_a_point(self):
        """Raw 0, .8, .2, 1 pools to 0, .5, .5, 1: the crossing lies in [1, 2]."""
        margin = estimator().estimate(points([0, 8, 2, 10]))
        assert margin.censoring == "identified_interval"
        assert margin.threshold is None
        assert margin.threshold_interval == (1.0, 2.0)
        assert margin.monotone_violations == 1 and not margin.monotonic_assumption
        # The legacy estimator answered 1.0 here: the left edge of a set.
        legacy_threshold, legacy_label = _legacy_estimate([0, 8, 2, 10])
        assert legacy_threshold == 1.0
        # The schema refuses that answer in both of its forms.
        with pytest.raises(ValueError, match="unknown censoring"):
            replace(margin, censoring=legacy_label)
        with pytest.raises(ValueError, match="point threshold"):
            replace(margin, threshold=legacy_threshold)

    @pytest.mark.guard
    def test_no_monotone_support_is_unidentifiable(self):
        """Raw .9, .1, .4 pools into ONE isotonic block: no crossing exists in the fit.

        Pool-adjacent-violators merges .9 and .1 into .5, then .5 > .4 merges
        again, leaving a single block at 14/30. The v2 audit described .9, .1,
        .8 as pooling to one block; with standard PAVA that input gives two
        blocks (.5, .5, .8), which the estimator now censors at the support
        boundary rather than reporting the manufactured point the legacy code
        returned. Both cases are pinned here.
        """
        margin = estimator().estimate(points([9, 1, 4]))
        assert margin.censoring == "unidentifiable"
        assert margin.threshold is None and margin.threshold_interval is None
        assert margin.monotone_violations == 1
        assert margin.fitted_probabilities == tuple((float(x), 14 / 30) for x in range(3))
        edge = estimator().estimate(points([9, 1, 8]))
        assert edge.censoring == "below_support" and edge.threshold is None
        legacy_threshold, _ = _legacy_estimate([9, 1, 8])
        assert legacy_threshold == 0.0  # a manufactured point
        flat = estimator().estimate(points([5, 5, 5]))
        assert flat.censoring == "unidentifiable" and flat.monotone_violations == 0

    def test_censoring_at_support_boundaries(self):
        assert estimator().estimate(points([0, 0])).censoring == "above_support"
        assert estimator().estimate(points([10, 10])).censoring == "below_support"
        # Fitted exactly at the quantile at the smallest severity: the crossing
        # is at or below the support, never reported as a point at xs[0].
        assert estimator().estimate(points([5, 8, 10])).censoring == "below_support"

    def test_plateau_reaching_the_end_of_support_is_still_an_interval(self):
        margin = estimator().estimate(points([0, 5, 5]))
        assert margin.censoring == "identified_interval"
        assert margin.threshold_interval == (1.0, 2.0)

    def test_raw_and_fitted_probabilities_are_reported(self):
        margin = estimator().estimate(points([0, 8, 2, 10]))
        assert margin.raw_probabilities == ((0.0, 0.0), (1.0, 0.8), (2.0, 0.2), (3.0, 1.0))
        assert margin.fitted_probabilities == ((0.0, 0.0), (1.0, 0.5), (2.0, 0.5), (3.0, 1.0))

    def test_monotonic_assumption_cannot_be_asserted_against_violations(self):
        margin = estimator().estimate(points([0, 8, 2, 10]))
        with pytest.raises(ValueError, match="derived"):
            replace(margin, monotonic_assumption=True)
        with pytest.raises(ValueError, match="threshold interval"):
            replace(estimator().estimate(points([0, 2, 8, 10])), threshold_interval=(1.0, 2.0))

    def test_other_quantiles(self):
        low = estimator(0.1).estimate(points([0, 2, 8, 10]))
        assert low.censoring == "interpolated_crossing" and low.threshold == pytest.approx(0.5)
        high = estimator(0.9).estimate(points([0, 2, 8, 10]))
        assert high.threshold == pytest.approx(2.5)


class TestShift:
    def test_shift_is_defined_only_between_interpolated_crossings(self):
        a = estimator().estimate(points([0, 2, 8, 10]))
        b = estimator().estimate(points([0, 0, 4, 10]))
        shift, reason = frontier_shift_with_reason(a, b)
        assert shift == pytest.approx(b.threshold - a.threshold) and shift > 0
        assert reason == "difference of interpolated crossings"
        interval = estimator().estimate(points([0, 8, 2, 10]))
        shift, reason = frontier_shift_with_reason(a, interval)
        assert shift is None and "identified_interval" in reason
        assert frontier_shift(interval, a) is None
        unident = estimator().estimate(points([9, 1, 8, 5]))
        assert frontier_shift(unident, a) is None

    def test_definitional_mismatch_is_an_error_not_a_none(self):
        a = estimator().estimate(points([0, 2, 8, 10]))
        b = estimator().estimate(points([0, 0, 4, 10]))
        with pytest.raises(ValueError, match="identical"):
            frontier_shift(a, replace(b, quantile=0.4))
        with pytest.raises(ValueError, match="identical"):
            frontier_shift(a, replace(b, scenario_ids=("other",)))

    def test_prioritize_unchanged(self):
        weights = estimator().prioritize(points([0, 5, 10]))
        assert weights.sum() == pytest.approx(1.0) and weights[1] > weights[0]
        with pytest.raises(ValueError):
            estimator().prioritize([])


def test_margin_schema_rejects_unknown_censoring():
    with pytest.raises(ValueError, match="unknown censoring"):
        RobustnessMargin(0.5, None, "observed_crossing", (0.0, 1.0), {})
