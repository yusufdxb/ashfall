"""D1/D2/D3 gates, the delivery verdict, and H0 calibration on the toy surrogate."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ashfall.backends.toy import ToyBackend
from ashfall.capsule import FailureCapsule
from ashfall.counterfactual import DepartureConfig
from ashfall.datasets import DatasetManifest, NotScientificData, assert_scientific
from ashfall.gates import InterventionReceipt, evaluate_delivery, gate_d1
from ashfall.h0 import (
    H0Spec,
    exact_mcnemar_p,
    h0_seed_plan,
    run_h0,
    select_states,
    summarise_attempts,
)
from ashfall.harvest import capsule_from_pair, harvest_h0_run
from ashfall.ontology import Intervention

FRICTION = Intervention.friction_reduction(0.1, 0.08)


@pytest.fixture(scope="module")
def backend():
    return ToyBackend()


@pytest.fixture(scope="module")
def state(backend):
    return backend.nominal_rollout(seed=1, horizon_steps=150, command=(0.6, 0.0, 0.0)).state_at(60)


@pytest.fixture(scope="module")
def slip_pair(backend, state):
    return backend.matched_pair(
        state, FRICTION, simulator_seed=7, replicate_seeds=(11, 13, 17, 19), horizon_steps=150
    )


class TestReceipts:
    def test_verified_is_derived_from_readback(self):
        ok = InterventionReceipt(
            FRICTION, dict(FRICTION.parameters), dict(FRICTION.parameters), "readback"
        )
        assert ok.verified
        off = InterventionReceipt(
            FRICTION,
            dict(FRICTION.parameters),
            {"static_friction": 0.1, "dynamic_friction": 0.5},
            "readback",
        )
        assert not off.verified
        missing = InterventionReceipt(FRICTION, dict(FRICTION.parameters), None, "none")
        assert not missing.verified
        with pytest.raises(ValueError, match="requested parameters"):
            InterventionReceipt(FRICTION, {"static_friction": 0.1}, None, "x")

    def test_d1_requires_both_receipts(self, slip_pair):
        good = gate_d1(slip_pair.treatment_receipt, slip_pair.control_receipt, expected=FRICTION)
        assert good.passed
        no_control = gate_d1(slip_pair.treatment_receipt, None, expected=FRICTION)
        assert not no_control.passed and "control" in no_control.detail["reason"]
        other = gate_d1(
            slip_pair.treatment_receipt,
            slip_pair.control_receipt,
            expected=Intervention.friction_reduction(0.5, 0.4),
        )
        assert not other.passed
        unverified = replace(
            slip_pair.treatment_receipt, readback={"static_friction": 0.9, "dynamic_friction": 0.9}
        )
        assert not gate_d1(unverified, slip_pair.control_receipt, expected=FRICTION).passed


class TestVerdicts:
    def test_friction_delivers_slip(self, slip_pair):
        verdict = evaluate_delivery(slip_pair, intervention=FRICTION, intended_phenotype="slip")
        assert verdict.status == "DELIVERED"
        assert verdict.d2_departure.detail["directional"]["moves_in_expected_direction"]

    def test_wrong_phenotype_and_unsupported(self, backend, state, slip_pair):
        assert (
            evaluate_delivery(
                slip_pair, intervention=FRICTION, intended_phenotype="collapse"
            ).status
            == "WRONG_PHENOTYPE"
        )
        terrain = Intervention("terrain_geometry", (("step_height_m", 0.1),))
        pair = backend.matched_pair(
            state, terrain, simulator_seed=7, replicate_seeds=(11, 13, 17), horizon_steps=100
        )
        assert (
            evaluate_delivery(pair, intervention=terrain, intended_phenotype="stumble").status
            == "UNSUPPORTED"
        )

    @pytest.mark.guard
    def test_skipped_intervention_is_not_applied_even_if_trajectories_differ(self, slip_pair):
        """Reintroduce the defect: use the treatment trajectory with no verified receipt."""
        fake_receipt = InterventionReceipt(FRICTION, dict(FRICTION.parameters), None, "not_applied")
        evidence = replace(slip_pair, treatment_receipt=fake_receipt)
        verdict = evaluate_delivery(evidence, intervention=FRICTION, intended_phenotype="slip")
        assert verdict.d2_departure.passed and verdict.d3_phenotype.passed
        assert verdict.status == "NOT_APPLIED" and not verdict.delivered

    def test_phenotype_in_both_arms_is_not_attributable(self, backend):
        # Overload the surrogate so even the control collapses.
        heavy_state = backend.spawn_state((0.3, 0, 0))
        mass = Intervention("payload_mass", (("mass_offset_kg", 4.5),))
        pair = backend.matched_pair(
            heavy_state, mass, simulator_seed=3, replicate_seeds=(4, 5, 6), horizon_steps=150
        )
        control_collapses = pair.control.termination_reason == "base_contact"
        verdict = evaluate_delivery(pair, intervention=mass, intended_phenotype="collapse")
        if control_collapses:
            assert verdict.status in ("PHENOTYPE_ABSENT", "WRONG_PHENOTYPE", "NO_DEPARTURE")
        else:
            assert verdict.status == "DELIVERED"

    def test_evidence_validation(self, slip_pair):
        with pytest.raises(ValueError, match="reuse"):
            replace(slip_pair, replicate_seeds=(7, 13, 17, 19))
        with pytest.raises(ValueError, match="distinct"):
            replace(slip_pair, replicate_seeds=(11, 11, 17, 19))


class TestH0:
    def test_mcnemar_exact(self):
        assert exact_mcnemar_p(0, 0) == 1.0
        assert exact_mcnemar_p(6, 0) == pytest.approx(2 / 64)
        assert exact_mcnemar_p(3, 3) == 1.0

    def test_seed_plan_is_disjoint_and_deterministic(self):
        spec = H0Spec(FRICTION, "slip", n_pairs=5, replicates_per_pair=3, horizon_steps=50)
        plan = h0_seed_plan(spec)
        seeds = [s["simulator_seed"] for s in plan] + [
            r for s in plan for r in s["replicate_seeds"]
        ]
        assert len(seeds) == len(set(seeds)) == 20
        assert plan == h0_seed_plan(spec)

    def test_spec_refuses_conflation(self):
        with pytest.raises(ValueError):
            H0Spec(
                FRICTION, "friction_reduction", n_pairs=4, replicates_per_pair=3, horizon_steps=50
            )
        with pytest.raises(ValueError):
            H0Spec(Intervention.none(), "slip", n_pairs=4, replicates_per_pair=3, horizon_steps=50)

    def test_friction_to_slip_passes_and_wrong_phenotype_fails(self, backend):
        spec = H0Spec(FRICTION, "slip", n_pairs=6, replicates_per_pair=4, horizon_steps=150)
        states = select_states(backend, spec, command=(0.6, 0, 0))
        result, records = run_h0(backend, spec, states=states)
        assert result.verdict == "PASS" and result.delivered_fraction == 1.0
        assert result.exact_p == pytest.approx(result.p_floor) == pytest.approx(2 / 64)
        wrong = H0Spec(FRICTION, "collapse", n_pairs=6, replicates_per_pair=4, horizon_steps=150)
        bad, _ = run_h0(backend, wrong, states=states)
        assert bad.verdict == "FAIL" and bad.statuses == {"WRONG_PHENOTYPE": 6}

    def test_too_few_pairs_cannot_pass(self, backend):
        spec = H0Spec(FRICTION, "slip", n_pairs=3, replicates_per_pair=3, horizon_steps=150)
        result, _ = run_h0(backend, spec, states=select_states(backend, spec, command=(0.6, 0, 0)))
        assert result.verdict == "FAIL" and any("floor" in r for r in result.reasons)

    @pytest.mark.guard
    def test_unverified_intervention_makes_h0_uninformative(self, backend):
        spec = H0Spec(FRICTION, "slip", n_pairs=6, replicates_per_pair=4, horizon_steps=150)
        _, records = run_h0(backend, spec, states=select_states(backend, spec, command=(0.6, 0, 0)))
        # Reintroduce the defect on one pair: the intervention was never applied.
        from ashfall.ontology import DeliveryVerdict, GateResult

        broken = records[0].attempt
        verdict = DeliveryVerdict(
            broken.intended_phenotype,
            broken.intervention,
            GateResult("D1_intervention", False, {"reason": "no receipt"}),
            broken.verdict.d2_departure,
            broken.verdict.d3_phenotype,
        )
        attempts = [replace(broken, verdict=verdict, attempt_id="")] + [
            r.attempt for r in records[1:]
        ]
        result = summarise_attempts(spec, attempts)
        assert result.verdict == "UNINFORMATIVE"

    def test_harvest_writes_capsules_and_a_fixture_manifest_for_mock_evidence(
        self, backend, tmp_path
    ):
        spec = H0Spec(FRICTION, "slip", n_pairs=6, replicates_per_pair=4, horizon_steps=150)
        result, records = run_h0(
            backend, spec, states=select_states(backend, spec, command=(0.6, 0, 0))
        )
        summary = harvest_h0_run(
            spec,
            result,
            records,
            tmp_path,
            robot="toy",
            provenance={
                "ashfall_sha": "1" * 40,
                "phoenix_sha": None,
                "policy_id": backend.policy_id,
                "env_config_hash": backend.env_config_hash,
                "simulator_version": backend.simulator_version,
            },
        )
        assert summary.capsules == 6 and summary.dataset_kind == "fixture"
        manifest = DatasetManifest.load(tmp_path)
        manifest.verify(tmp_path)
        with pytest.raises(NotScientificData):
            assert_scientific(manifest, purpose="H1 fidelity")
        capsule = FailureCapsule.load(next((tmp_path / "capsules").glob("*.json")))
        assert capsule.schema_version == "1.1" and capsule.cause == FRICTION
        assert capsule.phenotype == "slip" and capsule.window.label_source == "detector"
        assert (
            capsule.window.precursor_start is None
            or capsule.window.precursor_start <= capsule.failure_onset_index
        )
        assert capsule.provenance["attempt_id"] == records[0].attempt.attempt_id or any(
            capsule.provenance["attempt_id"] == r.attempt.attempt_id for r in records
        )
        assert (tmp_path / "h0_result.json").exists() and (tmp_path / "attempts.json").exists()
        recorded = json.loads((tmp_path / "h0_result.json").read_text())
        assert recorded["verdict"] == "PASS"

    def test_undelivered_pair_never_becomes_a_capsule(self, backend, state):
        pair = backend.matched_pair(
            state, FRICTION, simulator_seed=7, replicate_seeds=(11, 13, 17), horizon_steps=150
        )
        spec = H0Spec(FRICTION, "collapse", n_pairs=1, replicates_per_pair=3, horizon_steps=150)
        _, records = run_h0(backend, spec, states=[state])
        assert records[0].attempt.verdict.status == "WRONG_PHENOTYPE"
        assert capsule_from_pair(records[0], robot="toy") is None
        del pair

    def test_departure_config_flows_through_spec(self, backend):
        strict = H0Spec(
            FRICTION,
            "slip",
            n_pairs=6,
            replicates_per_pair=4,
            horizon_steps=150,
            departure=DepartureConfig(margin_ratio=50.0),
        )
        result, _ = run_h0(
            backend, strict, states=select_states(backend, strict, command=(0.6, 0, 0))
        )
        assert result.statuses.get("NO_DEPARTURE", 0) == 6 and result.verdict == "FAIL"
