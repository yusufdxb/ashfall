from dataclasses import replace

import pytest

from ashfall.basin import BasinObservation, discover_basin
from ashfall.demo import AnalyticBackend, demo_capsule
from ashfall.frontier import FrontierEstimator, FrontierPoint, SeverityMetric, frontier_shift
from ashfall.repair import RepairCurriculum
from ashfall.reproduction import (
    FailureDescriptor,
    ReproductionCandidate,
    ReproductionConfig,
    ReproductionGate,
)
from ashfall.scenarios import ScenarioManifest


def reproduction():
    capsule = demo_capsule()
    config = ReproductionConfig("baseline", FailureDescriptor("slip", 0.5), 45)
    return capsule, ReproductionGate(config).run(
        capsule,
        ReproductionCandidate(capsule.capsule_id, (("dynamic_friction", 0.3),), 45),
        AnalyticBackend(),
    )


@pytest.mark.parametrize("damage", ["scores", "observations", "row", "backend"])
def test_in_memory_reproduction_cannot_bypass_evidence_validation(damage):
    _, result = reproduction()
    with pytest.raises(ValueError):
        if damage == "scores":
            replace(result, scores=(1.0, 1.0, 0.0))
        if damage == "observations":
            replace(result, observations=())
        if damage == "row":
            replace(result, candidate=replace(result.candidate, seed_row=0))
        if damage == "backend":
            replace(result, backend_id="")


def test_nonfinite_similarity_configuration_and_contact_signatures_rejected():
    _, result = reproduction()
    with pytest.raises(ValueError):
        replace(result.config, time_tolerance_s=float("inf"))
    with pytest.raises(ValueError):
        FailureDescriptor("slip", 0.5, contact_signature=(float("nan"),))
    with pytest.raises(ValueError):
        replace(result.candidate, parameters=(("friction", 0.2), ("friction", 0.3)))


def test_reproduction_target_cannot_be_changed_to_an_easier_mode():
    capsule, result = reproduction()
    config = replace(result.config, target=FailureDescriptor("collapse", 0.5))
    with pytest.raises(ValueError, match="original capsule"):
        ReproductionGate(config).run(capsule, result.candidate, AnalyticBackend())


def test_repair_cannot_bind_a_different_initial_state_to_passed_reproduction():
    _, result = reproduction()
    manifest = discover_basin(
        result,
        bounds={"dynamic_friction": (0.1, 0.9)},
        counts={"train": 4, "validation": 2, "held_out": 3},
        allow_mock=True,
    )
    scenario = manifest.select("train")[0]
    altered = replace(scenario, initial_state_id="unreproduced_seed_state")
    manifest = ScenarioManifest(tuple(altered if s == scenario else s for s in manifest.scenarios))
    with pytest.raises(ValueError, match="seed state"):
        RepairCurriculum(manifest, [result], allow_mock=True)


def test_r50_cannot_pool_changing_nuisance_parameters():
    _, result = reproduction()
    manifest = discover_basin(
        result,
        bounds={"dynamic_friction": (0.1, 0.9), "command_vx": (0.1, 0.8)},
        counts={"train": 4, "validation": 2, "held_out": 3},
        allow_mock=True,
    )
    observations = [
        BasinObservation(s.scenario_id, "baseline", (1, 2), (False, True), ({}, {}))
        for s in manifest.select("train")
    ]
    estimator = FrontierEstimator(SeverityMetric("slip", "dynamic_friction", "coefficient", -1, 1))
    with pytest.raises(ValueError, match="nuisance"):
        estimator.points(manifest, observations)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(severity=float("nan")),
        dict(failures=0.5),
        dict(probability=0.9),
        dict(uncertainty=float("inf")),
    ],
)
def test_frontier_invalid_empirical_probabilities_rejected(kwargs):
    arguments = dict(
        scenario_id="s1", severity=0.1, failures=1, replicates=2, probability=0.5, uncertainty=0.2
    )
    with pytest.raises(ValueError):
        FrontierPoint(**{**arguments, **kwargs})


def test_r50_shift_requires_same_scenarios_not_merely_same_endpoints():
    estimator = FrontierEstimator(SeverityMetric("slip", "friction", "coefficient", -1, 1))
    points = [FrontierPoint("s1", 0.0, 0, 10, 0.0, 0.2), FrontierPoint("s2", 1.0, 10, 10, 1.0, 0.2)]
    baseline = estimator.estimate(points)
    candidate = estimator.estimate(
        [replace(p, scenario_id="different-" + p.scenario_id) for p in points]
    )
    with pytest.raises(ValueError):
        frontier_shift(baseline, candidate)
    with pytest.raises(ValueError):
        estimator.estimate(points + [points[0]])
