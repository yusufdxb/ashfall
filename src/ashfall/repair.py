"""Inspectable reset curriculum; PPO consumes fresh on-policy rollouts only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scenarios import ScenarioManifest, seed_state_id


@dataclass(frozen=True)
class FrontierSchedule:
    nominal_fraction: float = 0.5
    frontier_fraction: float = 0.4
    hard_failure_fraction: float = 0.1
    refresh_interval: int = 10

    def __post_init__(self):
        fractions = (self.nominal_fraction, self.frontier_fraction, self.hard_failure_fraction)
        if min(fractions) < 0 or not np.isclose(sum(fractions), 1) or self.refresh_interval <= 0:
            raise ValueError("Reset fractions must be nonnegative, sum to one, refresh positive")


class ScenarioOutcomeBuffer:
    def __init__(self, manifest: ScenarioManifest):
        self.manifest = manifest
        self.policy_id = None
        self.outcomes = {}

    def replace(self, policy_id, outcomes):
        train = {s.scenario_id for s in self.manifest.select("train")}
        if not policy_id or set(outcomes) != train:
            raise ValueError("Refresh requires all and only training scenarios and policy identity")
        for values in outcomes.values():
            if not values or any(type(v) is not bool for v in values):
                raise ValueError("Actual replicate boolean outcomes required")
        self.policy_id = policy_id
        self.outcomes = {k: tuple(v) for k, v in outcomes.items()}


class RepairCurriculum:
    def __init__(
        self, manifest, reproductions, *, schedule=None, training_seed=0, allow_mock=False
    ):
        self.manifest = manifest
        self.schedule = schedule or FrontierSchedule()
        self.scenarios = manifest.select("train")
        if not self.scenarios:
            raise ValueError("Repair requires training scenarios")
        eligible = {}
        for result in reproductions:
            result.assert_eligible(allow_mock=allow_mock)
            eligible[result.reproduction_id] = (
                result.candidate.capsule_id,
                seed_state_id(result.candidate.capsule_id, result.candidate.seed_row),
            )
        for scenario in self.scenarios:
            manifest.assert_training(scenario)
            if eligible.get(scenario.reproduction_id) != (
                scenario.capsule_id,
                scenario.initial_state_id,
            ):
                raise ValueError(
                    "Every training scenario must bind to a reproduced capsule and seed state"
                )
        self.buffer = ScenarioOutcomeBuffer(manifest)
        self.rng = np.random.default_rng(training_seed)
        self.last_refresh = None

    def refresh_due(self, iteration):
        return (
            self.last_refresh is None
            or iteration - self.last_refresh >= self.schedule.refresh_interval
        )

    def update(self, outcomes, policy_id, iteration=0):
        if iteration < 0 or (self.last_refresh is not None and iteration < self.last_refresh):
            raise ValueError("Curriculum iteration cannot go backwards")
        self.buffer.replace(policy_id, outcomes)
        self.last_refresh = iteration

    def sample(self, n, iteration=0):
        if (
            type(n) is not int
            or n < 0
            or (self.last_refresh is not None and iteration < self.last_refresh)
            or self.refresh_due(iteration)
        ):
            raise ValueError("Current-policy frontier refresh required before sampling")
        outcomes = [self.buffer.outcomes[s.scenario_id] for s in self.scenarios]
        probabilities = np.array([sum(v) / len(v) for v in outcomes])
        # Beta posterior variance includes uncertainty when replicates are sparse.
        variance = np.array(
            [
                (sum(v) + 1) * (len(v) - sum(v) + 1) / ((len(v) + 2) ** 2 * (len(v) + 3))
                for v in outcomes
            ]
        )
        boundary = 0.01 + 1 - 2 * np.abs(probabilities - 0.5) + np.sqrt(variance)
        hard = 0.01 + probabilities
        boundary, hard = boundary / boundary.sum(), hard / hard.sum()
        fractions = [
            self.schedule.nominal_fraction,
            self.schedule.frontier_fraction,
            self.schedule.hard_failure_fraction,
        ]
        categories = self.rng.choice(3, n, p=fractions)
        result = []
        for category in categories:
            if category == 0:
                result.append(None)
            else:
                s = self.scenarios[
                    int(self.rng.choice(len(self.scenarios), p=boundary if category == 1 else hard))
                ]
                self.manifest.assert_training(s)
                result.append(s)
        return result


def train_with_frontier(
    *, curriculum, iterations, train_iteration, evaluate_training, policy_identity
):
    """Backend seam: train_iteration collects fresh PPO experience and updates once.

    evaluate_training runs a separate evaluation environment. No observations,
    actions or rewards from that evaluation are supplied to PPO minibatches.
    """
    for iteration in range(iterations):
        if curriculum.refresh_due(iteration):
            policy_id = policy_identity()
            outcomes = evaluate_training(curriculum.manifest.select("train"), policy_id)
            curriculum.update(outcomes, policy_id, iteration)
        train_iteration(iteration, lambda n: curriculum.sample(n, iteration))
