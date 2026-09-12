"""The cause -> response -> phenotype vocabulary and its refusals."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ashfall.capsule import CapsuleFrame, FailureCapsule
from ashfall.ontology import (
    FIRST_STUDY_PHENOTYPES,
    INTERVENTION_KINDS,
    PHENOTYPE_NAMES,
    PHENOTYPES,
    PLAUSIBLE_PATHWAYS,
    DeliveryVerdict,
    FailureEpisode,
    GateResult,
    Intervention,
    OnsetWindow,
    PhenotypeObservation,
    ReproductionAttempt,
    assert_not_conflated,
    phenotype_windows_from_events,
    plausible_phenotypes,
)

PROVENANCE = {
    "simulator_version": "toy-0",
    "seed": 3,
    "env_config_hash": "e" * 64,
    "backend_id": "toy",
}


class TestIntervention:
    def test_names_never_overlap_phenotypes(self):
        assert not set(INTERVENTION_KINDS) & set(PHENOTYPE_NAMES)

    def test_phenotype_name_is_not_an_intervention(self):
        with pytest.raises(ValueError, match="phenotype, not an intervention"):
            Intervention("slip", (("static_friction", 0.1),))

    def test_control_carries_no_parameters_and_active_needs_one(self):
        assert Intervention.none().is_control
        with pytest.raises(ValueError, match="does not accept"):
            Intervention("none", (("static_friction", 0.1),))
        with pytest.raises(ValueError, match="at least one parameter"):
            Intervention("friction_reduction")

    def test_undeclared_parameter_refused(self):
        with pytest.raises(ValueError, match="does not accept"):
            Intervention("friction_reduction", (("slope_rad", 0.1),))

    def test_friction_constructor_orders_and_hashes_content(self):
        a = Intervention.friction_reduction(0.4, 0.2)
        b = Intervention(
            "friction_reduction",
            (("dynamic_friction", 0.2), ("static_friction", 0.4)),
            intensity=0.8,
            intensity_axis="1-dynamic_friction",
        )
        assert a == b and a.intervention_id == b.intervention_id
        assert Intervention.from_dict(json.loads(json.dumps(a.to_dict()))) == a
        with pytest.raises(ValueError, match="dynamic friction"):
            Intervention.friction_reduction(0.1, 0.2)

    def test_intensity_requires_axis(self):
        with pytest.raises(ValueError, match="intensity"):
            Intervention("payload_mass", (("mass_offset_kg", 2.0),), intensity=2.0)

    @given(st.floats(min_value=0.0, max_value=1.0), st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=50, deadline=None)
    def test_friction_round_trip_is_identity(self, static, dynamic):
        static, dynamic = max(static, dynamic), min(static, dynamic)
        intervention = Intervention.friction_reduction(static, dynamic)
        assert Intervention.from_dict(intervention.to_dict()) == intervention


class TestPhenotypes:
    def test_registry_is_consistent(self):
        assert set(PHENOTYPE_NAMES) == set(PHENOTYPES)
        for spec in PHENOTYPES.values():
            assert spec.observability.keys() == {"simulation", "go2_hardware"}
            assert spec.first_confirmatory_study in ("included", "excluded")
            if spec.first_confirmatory_study == "excluded":
                assert spec.exclusion_reason

    def test_first_study_excludes_unsupported_phenotypes(self):
        assert "contact_loss" not in FIRST_STUDY_PHENOTYPES
        assert "stumble" not in FIRST_STUDY_PHENOTYPES
        for name in FIRST_STUDY_PHENOTYPES:
            assert PHENOTYPES[name].supported_on("simulation")

    def test_hardware_observability_is_honest(self):
        assert not PHENOTYPES["collapse"].supported_on("go2_hardware")
        assert not PHENOTYPES["slip"].supported_on("go2_hardware")
        assert PHENOTYPES["attitude"].supported_on("go2_hardware")
        assert PHENOTYPES["collapse"].unsupported_reason("go2_hardware")
        with pytest.raises(ValueError):
            PHENOTYPES["collapse"].supported_on("mars")

    def test_pathways_are_many_to_many(self):
        # No intervention maps to exactly one phenotype, and at least one
        # phenotype has more than one plausible cause.
        assert all(len(v) > 1 for v in PLAUSIBLE_PATHWAYS.values())
        causes = {}
        for kind, names in PLAUSIBLE_PATHWAYS.items():
            for name in names:
                causes.setdefault(name, set()).add(kind)
        assert any(len(v) > 1 for v in causes.values())
        assert plausible_phenotypes("none") == ()
        assert (
            plausible_phenotypes("friction_reduction") == PLAUSIBLE_PATHWAYS["friction_reduction"]
        )

    def test_conflation_guard(self):
        assert_not_conflated("friction_reduction", "slip")
        with pytest.raises(ValueError, match="names a phenotype"):
            assert_not_conflated("slip", "slip")
        with pytest.raises(ValueError, match="names an intervention"):
            assert_not_conflated("friction_reduction", "friction_reduction")
        with pytest.raises(ValueError, match="control"):
            assert_not_conflated("none", "slip")


class TestWindows:
    def test_ordering_enforced(self):
        OnsetWindow(10, 20, precursor_start=5, end=40)
        with pytest.raises(ValueError, match="precursor <= transition"):
            OnsetWindow(10, 5)
        with pytest.raises(ValueError):
            OnsetWindow(10, 20, end=15)
        with pytest.raises(ValueError, match="label source"):
            OnsetWindow(1, 2, label_source="vibes")

    def test_gradual_failure_is_not_a_single_timestamp(self):
        window = OnsetWindow(30, 50, precursor_start=20, label_source="simulator_ground_truth")
        assert window.development_frames == 20

    def test_observation_refuses_intervention_names(self):
        with pytest.raises(ValueError, match="intervention kind"):
            PhenotypeObservation("friction_reduction", OnsetWindow(1, 1))
        with pytest.raises(ValueError, match="unknown phenotype"):
            PhenotypeObservation("tumble", OnsetWindow(1, 1))

    def test_detector_events_become_windows_without_claiming_precursors(self):
        events = [
            {"mode": "slip", "timestamp_s": 1.0},
            {"mode": "slip", "timestamp_s": 2.5},
            {"mode": "collapse", "timestamp_s": 0.5},
        ]
        observations = phenotype_windows_from_events(events, control_dt=0.02)
        assert [o.phenotype for o in observations] == ["collapse", "slip"]
        slip = observations[1]
        assert slip.window.transition_start == slip.window.established_start == 50
        assert slip.window.precursor_start is None


def episode(**overrides):
    values = dict(
        source="simulation",
        policy_id="p" * 64,
        intervention=Intervention.friction_reduction(0.3, 0.2),
        command=(0.5, 0.0, 0.0),
        initial_state_id="state-1",
        control_dt=0.02,
        n_frames=200,
        termination_reason="evaluation_horizon",
        phenotypes=(PhenotypeObservation("slip", OnsetWindow(60, 80, precursor_start=50)),),
        provenance=PROVENANCE,
        seed=3,
    )
    values.update(overrides)
    return FailureEpisode(**values)


class TestEpisode:
    def test_round_trip_and_identity(self):
        ep = episode()
        assert FailureEpisode.from_dict(json.loads(json.dumps(ep.to_dict()))) == ep
        assert ep.episode_id.startswith("ep_")
        assert ep.phenotype_names == ("slip",)
        data = ep.to_dict()
        data["n_frames"] = 201
        with pytest.raises(ValueError, match="episode_id"):
            FailureEpisode.from_dict(data)

    def test_untraceable_episode_refused(self):
        with pytest.raises(ValueError, match="provenance is missing"):
            episode(provenance={"seed": 1})

    def test_window_must_lie_inside_episode(self):
        with pytest.raises(ValueError, match="outside the episode"):
            episode(phenotypes=(PhenotypeObservation("slip", OnsetWindow(10, 250)),))

    def test_source_vocabulary(self):
        with pytest.raises(ValueError, match="unknown episode source"):
            episode(source="dream")


def gate(name, passed, **detail):
    return GateResult(name, passed, detail)


class TestVerdict:
    def test_only_all_three_gates_deliver(self):
        intervention = Intervention.friction_reduction(0.3, 0.2)
        d1, d2, d3 = (
            gate("D1_intervention", True),
            gate("D2_departure", True),
            gate("D3_phenotype", True),
        )
        assert DeliveryVerdict("slip", intervention, d1, d2, d3).status == "DELIVERED"
        assert (
            DeliveryVerdict("slip", intervention, gate("D1_intervention", False), d2, d3).status
            == "NOT_APPLIED"
        )
        assert (
            DeliveryVerdict("slip", intervention, d1, gate("D2_departure", False), d3).status
            == "NO_DEPARTURE"
        )
        absent = DeliveryVerdict("slip", intervention, d1, d2, gate("D3_phenotype", False))
        assert absent.status == "PHENOTYPE_ABSENT" and not absent.delivered
        wrong = DeliveryVerdict(
            "slip",
            intervention,
            d1,
            d2,
            gate("D3_phenotype", False, observed_phenotypes=["collapse"]),
        )
        assert wrong.status == "WRONG_PHENOTYPE"
        unsupported = DeliveryVerdict(
            "slip", intervention, gate("D1_intervention", False, unsupported=True), d2, d3
        )
        assert unsupported.status == "UNSUPPORTED"

    def test_status_cannot_be_asserted_against_gates(self):
        intervention = Intervention.friction_reduction(0.3, 0.2)
        with pytest.raises(ValueError, match="contradicts"):
            DeliveryVerdict(
                "slip",
                intervention,
                gate("D1_intervention", False),
                gate("D2_departure", True),
                gate("D3_phenotype", True),
                status="DELIVERED",
            )

    def test_verdict_refuses_conflated_names_and_control(self):
        d1, d2, d3 = (
            gate("D1_intervention", True),
            gate("D2_departure", True),
            gate("D3_phenotype", True),
        )
        with pytest.raises(ValueError, match="control"):
            DeliveryVerdict("slip", Intervention.none(), d1, d2, d3)
        with pytest.raises(ValueError, match="names an intervention"):
            DeliveryVerdict(
                "friction_reduction", Intervention.friction_reduction(0.3, 0.2), d1, d2, d3
            )

    def test_attempt_binds_verdict_to_its_intervention(self):
        intervention = Intervention.friction_reduction(0.3, 0.2)
        verdict = DeliveryVerdict(
            "slip",
            intervention,
            gate("D1_intervention", True),
            gate("D2_departure", True),
            gate("D3_phenotype", True),
        )
        attempt = ReproductionAttempt(
            "state-1", "p" * 64, intervention, "slip", 7, "ep_a", "ep_b", "toy", "mock", verdict
        )
        assert attempt.attempt_id.startswith("att_")
        with pytest.raises(ValueError, match="distinct"):
            replace(attempt, treatment_episode_id="ep_a", attempt_id="")
        with pytest.raises(ValueError, match="differs"):
            replace(attempt, intervention=Intervention.friction_reduction(0.9, 0.8), attempt_id="")


class TestCapsuleOntologyFields:
    def frames(self, n=60):
        return tuple(
            CapsuleFrame(
                i * 0.02,
                base_pos=(0.0, 0.0, 0.3 if i < 40 else 0.1),
                base_quat=(0.0, 0.0, 0.0, 1.0),
                base_lin_vel_body=(0.5, 0.0, 0.0),
                base_ang_vel_body=(0.0, 0.0, 0.0),
                joint_pos=(0.0,) * 12,
                joint_vel=(0.0,) * 12,
                command_vel=(0.5, 0.0, 0.0),
            )
            for i in range(n)
        )

    def capsule(self, **overrides):
        values = dict(
            source="simulation",
            robot="go2",
            policy_id="p" * 64,
            timestamp=None,
            control_dt=0.02,
            failure_mode="collapse",
            failure_onset_index=40,
            pre_failure_start_index=20,
            post_failure_end_index=59,
            frames=self.frames(),
        )
        values.update(overrides)
        return FailureCapsule(**values)

    def test_schema_1_1_carries_cause_and_effect_separately(self, tmp_path):
        capsule = self.capsule(
            intervention=Intervention.friction_reduction(0.3, 0.2).to_dict(),
            phenotype_label="collapse",
            onset_window=OnsetWindow(30, 40, precursor_start=25).to_dict(),
            provenance=PROVENANCE,
        )
        assert capsule.schema_version == "1.1"
        assert capsule.cause.kind == "friction_reduction"
        assert capsule.phenotype == "collapse"
        assert capsule.window.development_frames == 10
        loaded = FailureCapsule.load(capsule.save(tmp_path / "c.json"))
        assert loaded == capsule

    def test_legacy_1_0_still_loads_without_ontology_fields(self):
        capsule = self.capsule(schema_version="1.0")
        assert capsule.cause is None and capsule.window is None
        assert capsule.phenotype == "collapse"
        with pytest.raises(ValueError, match="schema 1.0"):
            self.capsule(schema_version="1.0", phenotype_label="collapse")

    def test_conflation_refused_at_the_capsule(self):
        with pytest.raises(ValueError, match="names an intervention"):
            self.capsule(failure_mode="friction_reduction")
        with pytest.raises(ValueError, match="intervention kind"):
            self.capsule(phenotype_label="friction_reduction")
        with pytest.raises(ValueError, match="phenotype, not an intervention"):
            self.capsule(intervention={"kind": "collapse", "parameters": []})

    def test_window_must_agree_with_onset_index(self):
        with pytest.raises(ValueError, match="established_start"):
            self.capsule(onset_window=OnsetWindow(30, 41).to_dict())
