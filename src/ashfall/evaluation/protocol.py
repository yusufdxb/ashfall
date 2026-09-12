"""Frozen experimental protocols, nominal suites and independent-seed inference.

This is the v2 frontier-repair acceptance protocol (arms ``A_baseline`` to
``E_frontier_repair``) behind ``ashfall verdict``. The Phase-II comparison is
registered in ``docs/phase2/PREREGISTRATION.md`` and implemented by
:mod:`ashfall.protocol.arms` (arms A to D) with :mod:`ashfall.selection`. The
two arm vocabularies are distinct and must not be mixed in one analysis.

A manifest declares an experiment; it is not evidence that its scenarios were
executed. Unsupported nominal cases block experiment readiness explicitly.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from ashfall.evaluation.paired import matched_records, paired_comparison
from ashfall.evaluation.regression import RegressionBudget, RepairVerdict, regression_verdict
from ashfall.provenance import content_hash, write_artifact

ARMS = (
    "A_baseline",
    "B_plain_finetune",
    "C_generic_domain_randomization",
    "D_legacy_failure_reset",
    "E_frontier_repair",
)
NOMINAL_GROUPS = ("flat", "rough", "moderate_slope", "turning", "velocity_range", "nominal_push")


@dataclass(frozen=True)
class NominalScenario:
    group: str
    parameters: tuple[tuple[str, float], ...]
    scenario_seed: int
    support: str = "supported"
    unsupported_reason: str | None = None

    def __post_init__(self):
        if self.group not in NOMINAL_GROUPS or self.support not in {"supported", "unsupported"}:
            raise ValueError("unknown nominal group or support status")
        if self.support == "unsupported" and not self.unsupported_reason:
            raise ValueError("unsupported scenarios need an explicit reason")
        if len(dict(self.parameters)) != len(self.parameters):
            raise ValueError("duplicate nominal parameter")
        if not self.parameters or not all(math.isfinite(value) for _, value in self.parameters):
            raise ValueError("nominal parameters must be explicit finite values")
        object.__setattr__(self, "parameters", tuple(sorted(self.parameters)))

    @property
    def scenario_id(self):
        return content_hash(
            dict(
                group=self.group, parameters=dict(self.parameters), scenario_seed=self.scenario_seed
            )
        )

    @property
    def parameter_sample_id(self):
        return content_hash(dict(self.parameters))

    def to_dict(self):
        return {
            **asdict(self),
            "scenario_id": self.scenario_id,
            "parameter_sample_id": self.parameter_sample_id,
        }


@dataclass(frozen=True)
class NominalSuite:
    scenarios: tuple[NominalScenario, ...]
    schema_version: str = "2.0.0"
    required_groups: tuple[str, ...] = NOMINAL_GROUPS

    def __post_init__(self):
        if self.schema_version != "2.0.0" or not self.scenarios:
            raise ValueError("empty or unsupported nominal suite")
        object.__setattr__(
            self, "scenarios", tuple(sorted(self.scenarios, key=lambda s: s.scenario_id))
        )
        ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate nominal scenario")
        if not self.required_groups or not set(self.required_groups) <= set(NOMINAL_GROUPS):
            raise ValueError("invalid required nominal groups")
        if {s.group for s in self.scenarios} != set(self.required_groups):
            raise ValueError("each nominal family must be defined, including unsupported cases")

    @property
    def ready(self):
        return all(s.support == "supported" for s in self.scenarios)

    @property
    def manifest_hash(self):
        return content_hash(self.to_dict())

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "required_groups": list(self.required_groups),
            "scenarios": [s.to_dict() for s in sorted(self.scenarios, key=lambda s: s.scenario_id)],
        }

    def save(self, path):
        return write_artifact(path, self.to_dict())

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        scenarios = []
        for row in value.pop("scenarios"):
            row = dict(row)
            identity = row.pop("scenario_id")
            parameter_identity = row.pop("parameter_sample_id")
            row["parameters"] = tuple(tuple(pair) for pair in row["parameters"])
            scenario = NominalScenario(**row)
            if (
                scenario.scenario_id != identity
                or scenario.parameter_sample_id != parameter_identity
            ):
                raise ValueError("nominal scenario content identity mismatch")
            scenarios.append(scenario)
        value["required_groups"] = tuple(value["required_groups"])
        return cls(tuple(scenarios), **value)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def default_nominal_suite(*, supported_parameters: Sequence[str] = (), scenario_seed_base=4000):
    """Explicit initial design; backend must declare each implemented parameter.

    Roughness is RMS height in metres, slope radians, push impulse N s. A
    backend that implements a different terrain generator must create its own
    suite. These values are protocol inputs, not empirically tuned difficulty.
    """
    definitions = {
        "flat": [{"command_vx": 0.4, "command_vy": 0.0, "command_yaw": 0.0, "slope_rad": 0.0}],
        "rough": [{"command_vx": 0.4, "terrain_rms_height_m": 0.02}],
        "moderate_slope": [{"command_vx": 0.4, "slope_rad": slope} for slope in (-0.1, 0.1)],
        "turning": [{"command_vx": 0.3, "command_yaw": yaw} for yaw in (-0.5, 0.5)],
        "velocity_range": [{"command_vx": speed} for speed in (0.1, 0.4, 0.7)],
        "nominal_push": [{"command_vx": 0.4, "push_impulse_ns": 2.0, "push_time_s": 2.0}],
    }
    cases = []
    for group, configurations in definitions.items():
        for configuration in configurations:
            missing = sorted(set(configuration) - set(supported_parameters))
            cases.append(
                NominalScenario(
                    group,
                    tuple(configuration.items()),
                    scenario_seed_base + len(cases),
                    "unsupported" if missing else "supported",
                    "backend lacks: " + ", ".join(missing) if missing else None,
                )
            )
    return NominalSuite(tuple(cases))


@dataclass(frozen=True)
class ExperimentProtocol:
    baseline_policy_id: str
    counterexample_manifest_hash: str
    nominal_manifest_hash: str
    training_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    budget: RegressionBudget
    training_steps_per_arm: int
    target_mode: str = "slip"
    arms: tuple[str, ...] = ARMS
    schema_version: str = "2.0.0"
    seed_selection_rule: str = "preselected_without_outcome_screening"

    def __post_init__(self):
        if self.schema_version != "2.0.0" or self.arms != ARMS:
            raise ValueError("protocol requires the five declared comparison arms")
        if not all(
            (
                self.baseline_policy_id,
                self.counterexample_manifest_hash,
                self.nominal_manifest_hash,
                self.target_mode,
            )
        ):
            raise ValueError("baseline, target and frozen suite identities required")
        if len(self.training_seeds) < 3 or len(set(self.training_seeds)) != len(
            self.training_seeds
        ):
            raise ValueError("at least three distinct preselected training seeds required")
        if not self.evaluation_seeds or len(set(self.evaluation_seeds)) != len(
            self.evaluation_seeds
        ):
            raise ValueError("distinct evaluation replicate seeds required")
        if self.training_steps_per_arm <= 0:
            raise ValueError("fine-tuning arms require identical positive fresh rollout budgets")
        if self.seed_selection_rule != "preselected_without_outcome_screening":
            raise ValueError("outcome-screened seed selection is not allowed")
        if not self.budget.required_nominal_groups or not set(
            self.budget.required_nominal_groups
        ) <= set(NOMINAL_GROUPS):
            raise ValueError("protocol must declare its required nominal strata")

    @property
    def protocol_id(self):
        return content_hash(asdict(self))

    def to_dict(self):
        return {**asdict(self), "protocol_id": self.protocol_id}

    def save(self, path):
        return write_artifact(path, self.to_dict())

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        identity = value.pop("protocol_id")
        for name in ("training_seeds", "evaluation_seeds", "arms"):
            value[name] = tuple(value[name])
        budget = dict(value["budget"])
        budget["required_nominal_groups"] = tuple(budget["required_nominal_groups"])
        value["budget"] = RegressionBudget(**budget)
        result = cls(**value)
        if result.protocol_id != identity:
            raise ValueError("protocol content identity mismatch")
        return result

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class IndependentSeedEffect:
    training_seeds: tuple[int, ...]
    mean_effect: float
    ci_low: float
    ci_high: float
    confidence: float
    inference_unit: str = "independent_training_seed"


def independent_seed_interval(
    effects: Mapping[int, float], *, expected_seeds: Sequence[int], confidence=0.95
) -> IndependentSeedEffect:
    """Student-t interval across independently trained policies, equal seed weights.

    Episodes/replicates must first be reduced to one paired effect per training
    seed. Small-n normal-effect approximation remains a reporting assumption.
    """
    if set(effects) != set(expected_seeds) or len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("all and only preregistered training seeds must be reported")
    if len(effects) < 3 or not 0 < confidence < 1:
        raise ValueError("at least three independent effects and valid confidence required")
    seeds = tuple(sorted(effects))
    values = np.array([effects[seed] for seed in seeds], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("training-seed effects must be finite")
    mean = float(values.mean())
    radius = float(
        student_t.ppf((1 + confidence) / 2, len(values) - 1)
        * values.std(ddof=1)
        / math.sqrt(len(values))
    )
    return IndependentSeedEffect(seeds, mean, mean - radius, mean + radius, confidence)


def equivalence_verdict(
    effects: Mapping[int, float],
    *,
    expected_seeds: Sequence[int],
    equivalence_margin: float,
    alpha=0.05,
) -> dict:
    """TOST decision via the (1-2 alpha) interval within explicit symmetric bounds.

    A nonsignificant difference is never interpreted as equivalence.
    """
    if not math.isfinite(equivalence_margin) or equivalence_margin <= 0 or not 0 < alpha < 0.5:
        raise ValueError("positive prespecified equivalence margin and valid alpha required")
    interval = independent_seed_interval(
        effects, expected_seeds=expected_seeds, confidence=1 - 2 * alpha
    )
    return {
        "equivalent": interval.ci_low > -equivalence_margin
        and interval.ci_high < equivalence_margin,
        "equivalence_margin": equivalence_margin,
        "alpha": alpha,
        "interval": asdict(interval),
    }


#: Keys a backend may add to ``environment_parameters`` beside the applied
#: scenario parameters. They describe HOW the episode was produced and are
#: recorded, not compared against the frozen scenario (audit finding A4).
BACKEND_METADATA_KEYS = frozenset(
    {
        "effective_env_config_hash",
        "rsl_rl_version",
        "simulator",
        "material_scope",
        "known_physical_context_reconstructed",
        "actuator_internal_state_restored",
        "action_history_restored",
        "command_strategy",
        "reset_state_id",
        "success_criterion",
        "nominal_group",
        "scene_seed",
    }
)


def applied_parameters(record: Mapping, frozen_parameters) -> dict:
    """The scenario parameters a record reports it applied, without backend metadata.

    Every frozen parameter must be present in the record: a missing key is
    reported as missing rather than dropped, so a backend that silently
    skipped a parameter still fails the comparison. Extra keys outside the
    declared metadata vocabulary are kept as well, because an undeclared
    manipulation is a difference from the frozen scenario.
    """
    environment = record.get("environment_parameters") or {}
    names = {name for name, _ in frozen_parameters}
    return {
        key: value
        for key, value in environment.items()
        if key in names or key not in BACKEND_METADATA_KEYS
    }


def nominal_group_of(record: Mapping):
    """The nominal stratum a record was evaluated in, top-level or stamped by the backend."""
    group = record.get("nominal_group")
    if group is None:
        group = (record.get("environment_parameters") or {}).get("nominal_group")
    return group


def evidence_verdict(
    baseline_target,
    candidate_target,
    baseline_nominal,
    candidate_nominal,
    *,
    counterexample_manifest,
    nominal_suite: NominalSuite,
    reproductions,
    protocol: ExperimentProtocol,
    allow_mock=False,
) -> RepairVerdict:
    """Derive target improvement from complete reproduced held-out episode evidence.

    The lower clustered interval bound for target incidence reduction must meet
    the practical improvement margin. This checks one candidate training seed;
    independent_seed_interval is needed for a campaign-level estimate.
    """
    try:
        if (
            counterexample_manifest.manifest_hash != protocol.counterexample_manifest_hash
            or nominal_suite.manifest_hash != protocol.nominal_manifest_hash
        ):
            raise ValueError("suite hash differs from frozen protocol")
        if set(nominal_suite.required_groups) != set(protocol.budget.required_nominal_groups):
            raise ValueError("nominal suite strata differ from frozen protocol budgets")
        if not nominal_suite.ready:
            raise ValueError("nominal suite contains unsupported scenarios")
        reproduced = {}
        for reproduction in reproductions:
            reproduction.assert_eligible(allow_mock=allow_mock)
            if reproduction.config.baseline_policy_id != protocol.baseline_policy_id:
                raise ValueError("reproduction baseline differs from evaluation baseline")
            reproduced[reproduction.reproduction_id] = reproduction
        scenarios = counterexample_manifest.select("held_out")
        if not scenarios or any(s.reproduction_id not in reproduced for s in scenarios):
            raise ValueError("held-out scenario lacks passed reproduction evidence")
        pairs = matched_records(baseline_target, candidate_target)
        if {r["policy_id"] for r in baseline_target} != {protocol.baseline_policy_id}:
            raise ValueError("evaluated baseline differs from frozen checkpoint")
        training_seeds = {r.get("training_seed") for r in candidate_target}
        if len(training_seeds) != 1 or not training_seeds <= set(protocol.training_seeds):
            raise ValueError("candidate training seed was not preselected")
        expected = {(s.scenario_id, seed) for s in scenarios for seed in protocol.evaluation_seeds}
        if {(r["scenario_id"], r["evaluation_seed"]) for r in baseline_target} != expected:
            raise ValueError("held-out evaluation incomplete or contains unregistered scenarios")
        lookup = {s.scenario_id: s for s in scenarios}
        for left, right in pairs:
            scenario = lookup[left["scenario_id"]]
            for record in (left, right):
                if (
                    record["parameter_sample_id"] != scenario.parameter_sample_id
                    or record["scenario_seed"] != scenario.scenario_seed
                    or applied_parameters(record, scenario.parameters) != dict(scenario.parameters)
                ):
                    raise ValueError("observed target scenario differs from frozen parameters")
                if record.get("failure_modes") is None:
                    raise ValueError("target failure capture missing")
        nominal_pairs = matched_records(baseline_nominal, candidate_nominal)
        expected_nominal = {
            (s.scenario_id, seed)
            for s in nominal_suite.scenarios
            for seed in protocol.evaluation_seeds
        }
        if {(r["scenario_id"], r["evaluation_seed"]) for r in baseline_nominal} != expected_nominal:
            raise ValueError("nominal evaluation incomplete or contains unregistered scenarios")
        nominal_lookup = {s.scenario_id: s for s in nominal_suite.scenarios}
        for left, right in nominal_pairs:
            scenario = nominal_lookup[left["scenario_id"]]
            for record in (left, right):
                if (
                    nominal_group_of(record) != scenario.group
                    or record["scenario_seed"] != scenario.scenario_seed
                    or record["parameter_sample_id"] != scenario.parameter_sample_id
                    or applied_parameters(record, scenario.parameters) != dict(scenario.parameters)
                ):
                    raise ValueError("observed nominal case differs from frozen parameters")
        if {r["policy_id"] for r in baseline_nominal} != {protocol.baseline_policy_id} or {
            r["policy_id"] for r in candidate_nominal
        } != {r["policy_id"] for r in candidate_target}:
            raise ValueError("nominal and target evaluations used different policies")

        if {r.get("training_seed") for r in candidate_nominal} != training_seeds:
            raise ValueError("nominal and target candidate training seeds differ")

        def target_free(records):
            return [
                {**r, "target_free": protocol.target_mode not in r["failure_modes"]}
                for r in records
            ]

        effect = paired_comparison(
            target_free(baseline_target), target_free(candidate_target), metric="target_free"
        )
        return regression_verdict(
            effect.ci_low, baseline_nominal, candidate_nominal, protocol.budget
        )
    except (ValueError, TypeError, KeyError) as exc:
        return RepairVerdict(False, False, False, (str(exc),))
