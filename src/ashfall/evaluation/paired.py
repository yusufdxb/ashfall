"""Matched-scenario comparisons of observed episode records.

Intervals resample scenario identities as clusters, retaining all stochastic
replicates together. Training seeds must be analyzed independently, followed
by an across-training-seed analysis; this module does not pool training runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

PAIR_FIELDS = ("scenario_id", "scenario_seed", "evaluation_seed", "parameter_sample_id")


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


def paired_comparison(
    baseline: Sequence[Mapping],
    candidate: Sequence[Mapping],
    metric: str = "success",
    *,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
) -> PairedEffect:
    if bootstrap_samples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid interval configuration")
    matched = matched_records(baseline, candidate)
    clusters: dict[str, list[float]] = {}
    for left, right in matched:
        values = (left.get(metric), right.get(metric))
        if any(v is None or not np.isfinite(float(v)) for v in values):
            raise ValueError(f"missing or nonfinite observed metric: {metric}")
        clusters.setdefault(left["scenario_id"], []).append(float(values[1]) - float(values[0]))
    # Equal scenario weighting prevents scenarios with extra replicates dominating.
    differences = np.array([np.mean(values) for values in clusters.values()])
    rng = np.random.default_rng(bootstrap_seed)
    sampled = rng.choice(differences, (bootstrap_samples, len(differences))).mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(sampled, [alpha, 1 - alpha])
    return PairedEffect(
        metric,
        len(matched),
        len(clusters),
        float(differences.mean()),
        float(low),
        float(high),
        confidence,
    )


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
