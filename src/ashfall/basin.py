"""Reproduced counterexample neighborhoods with replicate-level outcomes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist

from .provenance import write_artifact
from .reproduction import ReproductionResult, sample_parameters
from .scenarios import Scenario, ScenarioManifest, seed_state_id


@dataclass(frozen=True)
class BasinObservation:
    scenario_id: str
    policy_id: str
    evaluation_seeds: tuple[int, ...]
    failures: tuple[bool, ...]
    descriptors: tuple[dict, ...]

    def __post_init__(self):
        if not self.scenario_id or not self.policy_id:
            raise ValueError("Basin observations require scenario and policy identities")
        if any(type(seed) is not int for seed in self.evaluation_seeds):
            raise ValueError("Evaluation seeds must be integer identifiers")
        if not self.failures or len(self.failures) != len(self.evaluation_seeds):
            raise ValueError("Every replicate needs an evaluation seed and outcome")
        if len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError("Replicate seeds must be independent identifiers")
        if len(self.descriptors) != len(self.failures) or any(
            type(v) is not bool for v in self.failures
        ):
            raise ValueError("Every replicate needs a descriptor and boolean outcome")

    @property
    def failure_probability(self):
        return sum(self.failures) / len(self.failures)

    @property
    def probability_interval(self):
        """Wilson 95% interval; conditional on this fixed scenario and policy."""
        n, p, z = len(self.failures), self.failure_probability, NormalDist().inv_cdf(0.975)
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
        return center - half, center + half

    def to_dict(self):
        return {
            **asdict(self),
            "failure_probability": self.failure_probability,
            "replicate_count": len(self.failures),
            "probability_interval": self.probability_interval,
        }


def discover_basin(
    reproduction: ReproductionResult,
    *,
    bounds: dict,
    counts: dict[str, int],
    search_seed=0,
    family="slip",
    allow_mock=False,
):
    """Freeze split assignments before collecting any policy outcomes.

    Bounds are explicitly declared local search support, not recovered terrain.
    Replication follows freezing via evaluate_basin or a simulator evaluator.
    """
    reproduction.assert_eligible(allow_mock=allow_mock)
    if set(counts) != {"train", "validation", "held_out"} or any(
        type(v) is not int or v <= 0 for v in counts.values()
    ):
        raise ValueError("Positive train, validation and held_out counts required")
    samples = iter(sample_parameters(bounds, sum(counts.values()), search_seed))
    scenarios = []
    index = 0
    state_id = seed_state_id(reproduction.candidate.capsule_id, reproduction.candidate.seed_row)
    for split in ("train", "validation", "held_out"):
        for _ in range(counts[split]):
            scenarios.append(
                Scenario(
                    reproduction.candidate.capsule_id,
                    tuple(next(samples).items()),
                    search_seed + index,
                    split,
                    reproduction.reproduction_id,
                    state_id,
                    family,
                )
            )
            index += 1
    return ScenarioManifest(tuple(scenarios))


def evaluate_basin(
    manifest, policy_id, evaluator, *, evaluation_seeds, splits=("train", "validation")
):
    if not evaluation_seeds or len(set(evaluation_seeds)) != len(evaluation_seeds):
        raise ValueError("Distinct replicate evaluation seeds required")
    result = []
    for scenario in manifest.scenarios:
        if scenario.split not in splits:
            continue
        descriptors = tuple(evaluator(scenario, policy_id, seed) for seed in evaluation_seeds)
        result.append(
            BasinObservation(
                scenario.scenario_id,
                policy_id,
                tuple(evaluation_seeds),
                tuple(d.mode is not None for d in descriptors),
                tuple(asdict(d) for d in descriptors),
            )
        )
    return result


def save_observations(path, observations):
    return write_artifact(
        path, {"schema_version": "2.0.0", "observations": [o.to_dict() for o in observations]}
    )
