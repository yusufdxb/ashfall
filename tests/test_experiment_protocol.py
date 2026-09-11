from dataclasses import replace

import pytest

from ashfall.basin import discover_basin
from ashfall.demo import AnalyticBackend, demo_capsule
from ashfall.evaluation.protocol import (
    NOMINAL_GROUPS,
    ExperimentProtocol,
    default_nominal_suite,
    equivalence_verdict,
    evidence_verdict,
    independent_seed_interval,
)
from ashfall.evaluation.regression import RegressionBudget
from ashfall.reproduction import (
    FailureDescriptor,
    ReproductionCandidate,
    ReproductionConfig,
    ReproductionGate,
)


def supported_suite():
    preliminary = default_nominal_suite()
    parameters = {name for scenario in preliminary.scenarios for name, _ in scenario.parameters}
    return default_nominal_suite(supported_parameters=parameters)


def test_nominal_definitions_frozen_and_unknown_backend_is_not_ready(tmp_path):
    unknown = default_nominal_suite()
    assert not unknown.ready
    assert all(case.unsupported_reason for case in unknown.scenarios)
    suite = supported_suite()
    assert suite.ready
    assert {s.group for s in suite.scenarios} == set(NOMINAL_GROUPS)
    assert suite.manifest_hash == supported_suite().manifest_hash
    path = suite.save(tmp_path / "nominal.json")
    suite.save(path)
    with pytest.raises(FileExistsError):
        unknown.save(path)


def protocol_fixture():
    capsule = demo_capsule()
    backend = AnalyticBackend()
    config = ReproductionConfig(
        "baseline", FailureDescriptor("slip", 0.5), capsule.resolve_seed_index()
    )
    candidate = ReproductionCandidate(
        capsule.capsule_id, (("dynamic_friction", 0.3),), config.seed_row
    )
    reproduction = ReproductionGate(config).run(capsule, candidate, backend)
    manifest = discover_basin(
        reproduction,
        bounds={"dynamic_friction": (0.1, 0.95)},
        counts={"train": 3, "validation": 2, "held_out": 5},
        allow_mock=True,
    )
    nominal = supported_suite()
    budget = RegressionBudget(required_nominal_groups=NOMINAL_GROUPS)
    protocol = ExperimentProtocol(
        "baseline",
        manifest.manifest_hash,
        nominal.manifest_hash,
        (11, 17, 23),
        (801, 809),
        budget,
        10000,
    )
    target, repaired, baseline_nominal, repaired_nominal = [], [], [], []
    for scenario in manifest.select("held_out"):
        for seed in protocol.evaluation_seeds:
            record = dict(
                scenario_id=scenario.scenario_id,
                scenario_seed=scenario.scenario_seed,
                evaluation_seed=seed,
                parameter_sample_id=scenario.parameter_sample_id,
                policy_id="baseline",
                training_seed=None,
                failure_modes=["slip"],
                environment_parameters=dict(scenario.parameters),
                success=False,
            )
            target.append(record)
            repaired.append(
                {
                    **record,
                    "failure_modes": [],
                    "success": True,
                    "policy_id": "candidate",
                    "training_seed": 11,
                }
            )
    for scenario in nominal.scenarios:
        for seed in protocol.evaluation_seeds:
            record = dict(
                scenario_id=scenario.scenario_id,
                scenario_seed=scenario.scenario_seed,
                evaluation_seed=seed,
                parameter_sample_id=scenario.parameter_sample_id,
                policy_id="baseline",
                training_seed=None,
                failure_modes=[],
                environment_parameters=dict(scenario.parameters),
                success=True,
                nominal_group=scenario.group,
                tracking_error=0.01,
                intervention_required=False,
            )
            baseline_nominal.append(record)
            repaired_nominal.append({**record, "policy_id": "candidate", "training_seed": 11})
    kwargs = dict(
        counterexample_manifest=manifest,
        nominal_suite=nominal,
        reproductions=[reproduction],
        protocol=protocol,
        allow_mock=True,
    )
    return [target, repaired, baseline_nominal, repaired_nominal], kwargs


def test_protocol_disallows_missing_baselines_screened_seeds_and_missing_strata():
    _, kwargs = protocol_fixture()
    protocol = kwargs["protocol"]
    with pytest.raises(ValueError):
        replace(protocol, training_seeds=(1, 1, 2))
    with pytest.raises(ValueError):
        replace(protocol, arms=("E_frontier_repair",))
    with pytest.raises(ValueError):
        replace(protocol, budget=RegressionBudget())
    with pytest.raises(ValueError):
        replace(protocol, seed_selection_rule="best_baseline_seeds")


def test_independent_training_seed_interval_and_explicit_equivalence():
    effects = {11: 0.1, 17: 0.2, 23: 0.3}
    interval = independent_seed_interval(effects, expected_seeds=(11, 17, 23))
    assert interval.mean_effect == pytest.approx(0.2)
    assert interval.ci_low < 0.1 and interval.ci_high > 0.3
    assert interval.inference_unit == "independent_training_seed"
    with pytest.raises(ValueError):
        independent_seed_interval(effects, expected_seeds=(11, 17, 23, 29))
    with pytest.raises(ValueError):
        independent_seed_interval({1: 0.1, 2: 0.2}, expected_seeds=(1, 2))
    assert not equivalence_verdict(effects, expected_seeds=(11, 17, 23), equivalence_margin=0.05)[
        "equivalent"
    ]
    assert equivalence_verdict(
        {11: 0.001, 17: 0.0, 23: -0.001}, expected_seeds=(11, 17, 23), equivalence_margin=0.05
    )["equivalent"]


def test_evidence_verdict_requires_reproduction_and_all_frozen_cases():
    records, kwargs = protocol_fixture()
    assert evidence_verdict(*records, **kwargs).accepted
    assert not evidence_verdict(*records, **{**kwargs, "allow_mock": False}).accepted
    assert not evidence_verdict(*records, **{**kwargs, "reproductions": []}).accepted
    records[1].pop()
    assert not evidence_verdict(*records, **kwargs).accepted


@pytest.mark.parametrize(
    "damage", ["policy", "parameters", "group", "seed", "target_labels", "nominal_missing"]
)
def test_evidence_tampering_rejected(damage):
    records, kwargs = protocol_fixture()
    if damage == "policy":
        records[2][0]["policy_id"] = "other"
    if damage == "parameters":
        records[0][0]["environment_parameters"] = records[1][0]["environment_parameters"] = {}
    if damage == "group":
        records[2][0]["nominal_group"] = records[3][0]["nominal_group"] = "invalid"
    if damage == "seed":
        for record in records[1]:
            record["training_seed"] = 999
    if damage == "target_labels":
        records[1][0]["failure_modes"] = None
    if damage == "nominal_missing":
        records[2].pop()
        records[3].pop()
    assert not evidence_verdict(*records, **kwargs).accepted


def test_protocol_and_suite_roundtrip_reject_tampered_content(tmp_path):
    _, kwargs = protocol_fixture()
    suite, protocol = kwargs["nominal_suite"], kwargs["protocol"]
    from ashfall.evaluation.protocol import NominalSuite

    assert NominalSuite.load(suite.save(tmp_path / "suite.json")) == suite
    assert ExperimentProtocol.load(protocol.save(tmp_path / "protocol.json")) == protocol
    value = protocol.to_dict()
    value["training_seeds"] = [101, 103, 107]
    with pytest.raises(ValueError):
        ExperimentProtocol.from_dict(value)
    value = suite.to_dict()
    value["scenarios"][0]["parameters"] = [["command_vx", 999]]
    with pytest.raises(ValueError):
        NominalSuite.from_dict(value)


def test_nominal_target_candidate_training_seed_must_match():
    records, kwargs = protocol_fixture()
    for record in records[3]:
        record["training_seed"] = 17
    assert not evidence_verdict(*records, **kwargs).accepted
