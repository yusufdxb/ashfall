"""Matched-scenario comparisons of observed episode records.

Intervals resample scenario identities as clusters, collapsing each cluster's
replicates to their mean first so a scenario with extra replicates does not
dominate. Training seeds must be analyzed independently, followed by an
across-training-seed analysis; this module does not pool training runs.

A cluster bootstrap has nothing to say when every cluster delta is identical:
the resampled distribution is a single point and the resulting zero-width
interval is not evidence of anything. That case is flagged as ``degenerate``
rather than reported as certainty; ``exact_binary_effect_bounds`` gives the
conservative alternative for binary metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta

PAIR_FIELDS = ("scenario_id", "scenario_seed", "evaluation_seed", "parameter_sample_id")

INTERVAL_METHODS = ("cluster_bootstrap_percentile", "degenerate_all_clusters_identical")


def matched_records(baseline: Sequence[Mapping], candidate: Sequence[Mapping]):
    def index(records):
        if not records:
            raise ValueError("evaluation requires observed episode records")
        result = {}
        training_seeds = {r.get("training_seed") for r in records}
        policy_ids = {r.get("policy_id") for r in records}
        if len(training_seeds) != 1 or len(policy_ids) != 1 or not next(iter(policy_ids)):
            raise ValueError("compare one independent training run and policy at a time")
        for record in records:
            key = tuple(record.get(field) for field in PAIR_FIELDS)
            if any(value is None or value == "" for value in key):
                raise ValueError("paired evaluation requires complete scenario identity")
            if key in result:
                raise ValueError("duplicate evaluation scenario/replicate")
            result[key] = record
        return result

    left, right = index(baseline), index(candidate)
    if left.keys() != right.keys():
        raise ValueError("paired evaluation requires exactly matching frozen scenarios")
    for key in left:
        if left[key].get("environment_parameters") != right[key].get("environment_parameters"):
            raise ValueError("matched scenario has different environment parameters")
    keys = sorted(left)
    return [(left[key], right[key]) for key in keys]


@dataclass(frozen=True)
class PairedEffect:
    metric: str
    pairs: int
    scenario_clusters: int
    candidate_minus_baseline: float
    ci_low: float
    ci_high: float
    confidence: float
    interval_method: str = "cluster_bootstrap_percentile"
    degenerate: bool = False
    n_bootstrap: int = 0

    def __post_init__(self):
        if self.interval_method not in INTERVAL_METHODS:
            raise ValueError(f"unknown interval method {self.interval_method!r}")
        if self.degenerate != (self.interval_method == "degenerate_all_clusters_identical"):
            raise ValueError("degenerate must agree with interval_method")


def _cluster_values(baseline, candidate, metric):
    matched = matched_records(baseline, candidate)
    clusters: dict[str, list[tuple[float, float]]] = {}
    for left, right in matched:
        values = (left.get(metric), right.get(metric))
        if any(v is None or not np.isfinite(float(v)) for v in values):
            raise ValueError(f"missing or nonfinite observed metric: {metric}")
        clusters.setdefault(left["scenario_id"], []).append((float(values[0]), float(values[1])))
    return matched, clusters


def paired_comparison(
    baseline: Sequence[Mapping],
    candidate: Sequence[Mapping],
    metric: str = "success",
    *,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
) -> PairedEffect:
    """Cluster-bootstrap percentile interval on the mean per-scenario delta.

    Assumes scenario clusters are exchangeable draws; replicates within a
    cluster are collapsed to their mean, not resampled. When every cluster
    delta is identical the interval degenerates to the point estimate and is
    marked so; the point estimate is unchanged.
    """
    if bootstrap_samples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid interval configuration")
    matched, clusters = _cluster_values(baseline, candidate, metric)
    # Equal scenario weighting prevents scenarios with extra replicates dominating.
    differences = np.array([np.mean([r - b for b, r in values]) for values in clusters.values()])
    point = float(differences.mean())
    if len(differences) < 2 or np.ptp(differences) == 0.0:
        return PairedEffect(
            metric,
            len(matched),
            len(clusters),
            point,
            point,
            point,
            confidence,
            interval_method="degenerate_all_clusters_identical",
            degenerate=True,
            n_bootstrap=0,
        )
    rng = np.random.default_rng(bootstrap_seed)
    sampled = rng.choice(differences, (bootstrap_samples, len(differences))).mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(sampled, [alpha, 1 - alpha])
    return PairedEffect(
        metric,
        len(matched),
        len(clusters),
        point,
        float(low),
        float(high),
        confidence,
        n_bootstrap=bootstrap_samples,
    )


def exact_binary_bounds(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval for a binomial proportion.

    Exact under independent identically distributed Bernoulli trials; it is
    conservative (coverage at least ``confidence``) and never zero-width.
    """
    if type(successes) is not int or type(n) is not int or n < 1 or not 0 <= successes <= n:
        raise ValueError("successes and n must be integers with 0 <= successes <= n >= 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")
    alpha = 1 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return lower, upper


def _one_sided_binary_bound(successes: int, n: int, level: float, side: str) -> float:
    if side == "lower":
        return 0.0 if successes == 0 else float(beta.ppf(1 - level, successes, n - successes + 1))
    return 1.0 if successes == n else float(beta.ppf(level, successes + 1, n - successes))


def exact_binary_effect_bounds(
    baseline_clusters: Sequence[Sequence[float]],
    candidate_clusters: Sequence[Sequence[float]],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Conservative bounds on the candidate-minus-baseline rate of a binary metric.

    Method. Each cluster is reduced to two indicators: ``all`` (every
    replicate is 1) and ``any`` (some replicate is 1). Their proportions
    bracket the per-episode rate, ``p_all <= p <= p_any``. One-sided
    Clopper-Pearson bounds at level ``1 - (1-confidence)/2`` are taken on the
    favourable indicator of each arm and combined by the union bound, so the
    returned lower bound satisfies ``P(lower <= p_cand - p_base) >= confidence``
    and the upper bound likewise, under arbitrary dependence between the arms
    and within a cluster. Independence is assumed only across clusters, which
    frozen scenario draws satisfy by construction.

    This is deliberately conservative. It states what a finite matched suite
    at ceiling can and cannot establish; it does not estimate the effect.
    """
    if len(baseline_clusters) != len(candidate_clusters) or not baseline_clusters:
        raise ValueError("matched cluster lists of equal positive length required")
    for arm in (baseline_clusters, candidate_clusters):
        for values in arm:
            if not values or any(v not in (0, 1, 0.0, 1.0, True, False) for v in values):
                raise ValueError("binary bounds need clusters of 0/1 replicate values")
    n = len(baseline_clusters)
    level = 1 - (1 - confidence) / 2

    def counts(arm):
        every = sum(1 for values in arm if all(bool(v) for v in values))
        some = sum(1 for values in arm if any(bool(v) for v in values))
        return every, some

    base_all, base_any = counts(baseline_clusters)
    cand_all, cand_any = counts(candidate_clusters)
    lower = _one_sided_binary_bound(cand_all, n, level, "lower") - _one_sided_binary_bound(
        base_any, n, level, "upper"
    )
    upper = _one_sided_binary_bound(cand_any, n, level, "upper") - _one_sided_binary_bound(
        base_all, n, level, "lower"
    )
    return float(lower), float(upper)


def paired_binary_effect_bounds(
    baseline: Sequence[Mapping],
    candidate: Sequence[Mapping],
    metric: str = "success",
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """``exact_binary_effect_bounds`` over matched records grouped by scenario."""
    _, clusters = _cluster_values(baseline, candidate, metric)
    base = [[b for b, _ in values] for values in clusters.values()]
    cand = [[r for _, r in values] for values in clusters.values()]
    return exact_binary_effect_bounds(base, cand, confidence)


def target_failure_probability(records: Sequence[Mapping], target_mode: str) -> float:
    """Observed probability target mode occurs in matched counterexample episodes.

    Call with the frozen counterexample suite, not nominal training episodes.
    Unknown event capture cannot be interpreted as absence of failure.
    """
    if not records:
        raise ValueError("no observed counterexample episodes")
    if any(r.get("failure_modes") is None for r in records):
        raise ValueError("failure mode capture unavailable")
    return float(np.mean([target_mode in record["failure_modes"] for record in records]))
