"""Research arms A to D and the compute budget they share, enforced not declared."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ashfall.ontology import FIRST_STUDY_PHENOTYPES
from ashfall.protocol.arms import ArmSet, ArmSpec, SamplerConfig, default_arm_set
from ashfall.protocol.budget import (
    BUDGET_FIELDS,
    ComputeBudget,
    ProtocolViolation,
    Tolerance,
    TrainingArtifact,
    artifact_from_rsl_rl,
    assert_equal_compute,
    compare_arms,
    default_tolerances,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def budget(**overrides):
    values = dict(
        fresh_env_transitions=100 * 4096 * 24,
        replay_transitions=100 * 4096 * 24 // 2,
        episodes=0,
        optimizer_updates=100 * 5 * 4,
        rollout_length=24,
        num_envs=4096,
        wall_clock_s=0.0,
        gpu_time_s=0.0,
    )
    values.update(overrides)
    return ComputeBudget(**values)


def artifact(arm="A_standard", seed=11, **overrides):
    values = dict(
        arm=arm,
        training_seed=seed,
        iterations=100,
        num_envs=4096,
        num_steps_per_env=24,
        num_learning_epochs=5,
        num_mini_batches=4,
        episodes=1234,
        replay_transitions=0 if arm == "A_standard" else 100 * 4096 * 24 // 2,
        wall_clock_s=600.0,
        gpu_time_s=550.0,
        initial_checkpoint_sha256=SHA_A,
        final_checkpoint_sha256=SHA_B if arm == "A_standard" else SHA_C,
        config_hashes={"env": SHA_A, "train": SHA_B},
    )
    values.update(overrides)
    return artifact_from_rsl_rl(**values)


ARMS = ("A_standard", "B_uniform_replay", "C_intervention_conditioned", "D_phenotype_conditioned")


def artifacts(seeds=(11, 23, 47), **overrides):
    return [artifact(arm, seed, **overrides) for arm in ARMS for seed in seeds]


class TestTolerance:
    def test_kinds(self):
        assert Tolerance("exact").allows(5, 5) and not Tolerance("exact").allows(5, 6)
        assert Tolerance("absolute", 2).allows(5, 7) and not Tolerance("absolute", 2).allows(5, 8)
        assert Tolerance("relative", 0.1).allows(110, 100)
        assert not Tolerance("relative", 0.1).allows(111, 100)
        assert Tolerance("recorded").allows(0, 10**9)
        with pytest.raises(ValueError, match="carries no value"):
            Tolerance("exact", 1.0)
        with pytest.raises(ValueError, match="'exact'"):
            Tolerance("relative", 0.0)
        with pytest.raises(ValueError, match="unknown"):
            Tolerance("fuzzy")


class TestBudget:
    def test_every_field_has_a_tolerance_and_the_core_is_enforced(self):
        assert set(default_tolerances()) == set(BUDGET_FIELDS)
        b = budget()
        assert b.tolerances["fresh_env_transitions"].kind == "exact"
        with pytest.raises(ValueError, match="cannot be merely recorded"):
            budget(tolerances={**default_tolerances(), "optimizer_updates": Tolerance("recorded")})
        with pytest.raises(ValueError, match="missing"):
            budget(tolerances={"fresh_env_transitions": Tolerance("exact")})
        with pytest.raises(ValueError, match="unknown budget field"):
            budget(tolerances={**default_tolerances(), "gpus": Tolerance("exact")})

    def test_round_trip_and_identity(self):
        b = budget(tolerances={**default_tolerances(), "wall_clock_s": Tolerance("relative", 0.25)})
        loaded = ComputeBudget.from_dict(json.loads(json.dumps(b.to_dict())))
        assert loaded == b and loaded.budget_id == b.budget_id
        data = b.to_dict()
        data["num_envs"] = 1
        with pytest.raises(ValueError, match="identity"):
            ComputeBudget.from_dict(data)

    def test_positive_training_required(self):
        with pytest.raises(ValueError, match="positive transitions"):
            budget(fresh_env_transitions=0)
        with pytest.raises(ValueError, match="nonnegative integer"):
            budget(num_envs=True)


class TestTrainingArtifact:
    def test_rsl_rl_derivation(self):
        a = artifact()
        assert a.fresh_env_transitions == 100 * 4096 * 24
        assert a.optimizer_updates == 100 * 5 * 4
        assert a.rollout_length == 24 and a.num_envs == 4096
        assert a.artifact_id.startswith("train_")
        assert TrainingArtifact.from_dict(json.loads(json.dumps(a.to_dict()))) == a

    def test_every_field_required(self):
        data = artifact().to_dict()
        data.pop("gpu_time_s")
        with pytest.raises(ValueError, match="missing fields"):
            TrainingArtifact.from_dict(data)

    def test_checkpoint_consistency(self):
        with pytest.raises(ValueError, match="did not train"):
            artifact(final_checkpoint_sha256=SHA_A)
        base = artifact().to_dict()
        base.pop("artifact_id")
        base.update(optimizer_updates=0, final_checkpoint_sha256=SHA_B)
        with pytest.raises(ValueError, match="zero optimizer updates"):
            TrainingArtifact(**base)
        base.update(optimizer_updates=0, final_checkpoint_sha256=SHA_A)
        TrainingArtifact(**base)  # a zero-update run is a legal record of no training

    def test_replay_is_a_subset_of_fresh(self):
        with pytest.raises(ValueError, match="subset"):
            artifact("B_uniform_replay", replay_transitions=10**12)

    def test_hashes_validated(self):
        with pytest.raises(ValueError, match="SHA256"):
            artifact(initial_checkpoint_sha256="abc")
        with pytest.raises(ValueError, match="config_hashes"):
            artifact(config_hashes={})


class TestEqualCompute:
    def test_matching_artifacts_comply(self):
        report = assert_equal_compute(artifacts(), budget(), replay_arms=frozenset(ARMS[1:]))
        assert report.compliant and "episodes" in report.recorded_only_fields
        assert "fresh_env_transitions" in report.checked_fields

    @pytest.mark.guard
    def test_guard_unequal_compute_is_rejected(self):
        """Reintroduce the defect: one arm quietly trains with more environments."""
        cells = artifacts()
        cheat = artifact("D_phenotype_conditioned", 11, num_envs=8192)
        cells = [c for c in cells if not (c.arm == cheat.arm and c.training_seed == 11)] + [cheat]
        with pytest.raises(
            ProtocolViolation, match="D_phenotype_conditioned.*fresh_env_transitions"
        ):
            assert_equal_compute(cells, budget(), replay_arms=frozenset(ARMS[1:]))
        report = compare_arms(cells, budget(), replay_arms=frozenset(ARMS[1:]))
        fields = {v.field for v in report.violations}
        assert {"fresh_env_transitions", "num_envs"} <= fields

    def test_standard_arm_must_not_replay_and_treated_arms_must_match_dose(self):
        cells = artifacts()
        leak = artifact("A_standard", 11, replay_transitions=10)
        cells = [c for c in cells if not (c.arm == "A_standard" and c.training_seed == 11)] + [leak]
        with pytest.raises(ProtocolViolation, match="A_standard.*replay_transitions"):
            assert_equal_compute(cells, budget(), replay_arms=frozenset(ARMS[1:]))
        cells = artifacts()
        underdosed = artifact("B_uniform_replay", 23, replay_transitions=5)
        cells = [c for c in cells if not (c.arm == "B_uniform_replay" and c.training_seed == 23)]
        with pytest.raises(
            ProtocolViolation, match="'B_uniform_replay' seed 23: replay_transitions"
        ):
            assert_equal_compute(cells + [underdosed], budget(), replay_arms=frozenset(ARMS[1:]))

    def test_shared_baseline_and_paired_seeds(self):
        cells = artifacts()
        other_base = artifact("C_intervention_conditioned", 47, initial_checkpoint_sha256=SHA_B)
        cells = [
            c
            for c in cells
            if not (c.arm == "C_intervention_conditioned" and c.training_seed == 47)
        ] + [other_base]
        report = compare_arms(cells, budget(), replay_arms=frozenset(ARMS[1:]))
        assert any(v.field == "initial_checkpoint_sha256" for v in report.violations)
        unpaired = artifacts()[:-1]  # drop D seed 47
        report = compare_arms(unpaired, budget(), replay_arms=frozenset(ARMS[1:]))
        assert any(
            v.field == "training_seed" and v.arm == "D_phenotype_conditioned"
            for v in report.violations
        )

    def test_recorded_fields_are_reported_not_enforced(self):
        cells = artifacts()
        slow = artifact("A_standard", 11, wall_clock_s=99999.0, episodes=5)
        cells = [c for c in cells if not (c.arm == "A_standard" and c.training_seed == 11)] + [slow]
        assert assert_equal_compute(cells, budget(), replay_arms=frozenset(ARMS[1:])).compliant
        bounded = budget(
            wall_clock_s=600.0,
            tolerances={**default_tolerances(), "wall_clock_s": Tolerance("relative", 0.2)},
        )
        with pytest.raises(ProtocolViolation, match="wall_clock_s"):
            assert_equal_compute(cells, bounded, replay_arms=frozenset(ARMS[1:]))

    def test_duplicates_and_empty_rejected(self):
        with pytest.raises(ValueError, match="no training artifacts"):
            compare_arms([], budget())
        a = artifact()
        with pytest.raises(ValueError, match="duplicate"):
            compare_arms([a, a], budget())

    @given(
        iterations=st.integers(min_value=1, max_value=500),
        num_envs=st.integers(min_value=1, max_value=8192),
        steps=st.integers(min_value=1, max_value=64),
        scale=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=40, deadline=None)
    def test_property_equal_parameters_comply_and_scaled_envs_violate(
        self, iterations, num_envs, steps, scale
    ):
        b = ComputeBudget(
            iterations * num_envs * steps, 0, 0, iterations * 20, steps, num_envs, 0.0, 0.0
        )
        common = dict(
            iterations=iterations,
            num_steps_per_env=steps,
            num_learning_epochs=5,
            num_mini_batches=4,
            episodes=0,
            replay_transitions=0,
            wall_clock_s=1.0,
            gpu_time_s=1.0,
            initial_checkpoint_sha256=SHA_A,
            config_hashes={"env": SHA_A},
        )
        fair = [
            artifact_from_rsl_rl(
                arm="A_standard",
                training_seed=1,
                num_envs=num_envs,
                final_checkpoint_sha256=SHA_B,
                **common,
            ),
            artifact_from_rsl_rl(
                arm="B_uniform_replay",
                training_seed=1,
                num_envs=num_envs,
                final_checkpoint_sha256=SHA_C,
                **common,
            ),
        ]
        assert compare_arms(fair, b).compliant
        unfair = [
            fair[0],
            artifact_from_rsl_rl(
                arm="B_uniform_replay",
                training_seed=1,
                num_envs=num_envs * scale,
                final_checkpoint_sha256=SHA_C,
                **common,
            ),
        ]
        assert not compare_arms(unfair, b).compliant


class TestArms:
    def test_default_arm_set_round_trips(self, tmp_path):
        arms = default_arm_set(budget())
        assert [a.kind for a in arms.arms] == [
            "standard",
            "uniform_replay",
            "intervention_conditioned",
            "phenotype_conditioned",
        ]
        assert arms.standard_arm.name == "A_standard"
        assert arms.replay_arm_names == frozenset(ARMS[1:])
        assert arms.by_name("D_phenotype_conditioned").phenotypes == FIRST_STUDY_PHENOTYPES
        path = arms.save(tmp_path / "arms.json")
        assert ArmSet.load(path) == arms
        assert ArmSet.load(path).arm_set_id == arms.arm_set_id

    def test_exactly_one_standard_and_unique_names(self):
        arms = default_arm_set(budget())
        with pytest.raises(ValueError, match="exactly one standard"):
            ArmSet(arms.arms[1:], budget())
        with pytest.raises(ValueError, match="unique"):
            ArmSet(arms.arms + (arms.arms[1],), budget())
        with pytest.raises(ValueError, match="zero replay"):
            ArmSet(arms.arms, budget(replay_transitions=0))
        # A set with no replay arm cannot exist: every non-standard kind replays
        # and only one standard arm is allowed, so the "budget grants replay
        # but nothing replays" branch is only reachable through the
        # single-standard rule, which fires first.
        with pytest.raises(ValueError, match="exactly one standard"):
            ArmSet(arms.arms[:1] + (replace(arms.arms[0], name="A2"),), budget())

    @pytest.mark.guard
    def test_guard_phenotype_intervention_conflation_refused(self):
        """Reintroduce the defect: condition an arm on 'slip' as if it were a cause."""
        with pytest.raises(ValueError, match="is a phenotype"):
            ArmSpec(
                "C",
                "intervention_conditioned",
                "x",
                SamplerConfig(0.5, ("intervention_kind",)),
                intervention_kinds=("slip",),
            )
        with pytest.raises(ValueError, match="is an intervention kind"):
            ArmSpec(
                "D",
                "phenotype_conditioned",
                "x",
                SamplerConfig(0.5, ("phenotype",)),
                phenotypes=("friction_reduction",),
            )

    def test_excluded_phenotypes_cannot_be_conditioned_on(self):
        with pytest.raises(ValueError, match="excluded from the first confirmatory study"):
            ArmSpec(
                "D",
                "phenotype_conditioned",
                "x",
                SamplerConfig(0.5, ("phenotype",)),
                phenotypes=("contact_loss",),
            )
        with pytest.raises(ValueError, match="excluded"):
            ArmSpec(
                "D",
                "phenotype_conditioned",
                "x",
                SamplerConfig(0.5, ("phenotype",)),
                phenotypes=("slip", "stumble"),
            )

    def test_kind_rules(self):
        with pytest.raises(ValueError, match="replays nothing"):
            ArmSpec("A", "standard", "x", SamplerConfig(0.1))
        with pytest.raises(ValueError, match="positive replay_fraction"):
            ArmSpec("B", "uniform_replay", "x", SamplerConfig(0.0))
        with pytest.raises(ValueError, match="unconditioned"):
            ArmSpec("B", "uniform_replay", "x", SamplerConfig(0.5, ("phenotype",)))
        with pytest.raises(ValueError, match="must name its intervention kinds"):
            ArmSpec(
                "C", "intervention_conditioned", "x", SamplerConfig(0.5, ("intervention_kind",))
            )
        with pytest.raises(ValueError, match="stratify on phenotype"):
            ArmSpec("D", "phenotype_conditioned", "x", SamplerConfig(0.5), phenotypes=("slip",))
        with pytest.raises(ValueError, match="unknown or empty intervention kind"):
            ArmSpec(
                "C",
                "intervention_conditioned",
                "x",
                SamplerConfig(0.5, ("intervention_kind",)),
                intervention_kinds=("none",),
            )

    @pytest.mark.guard
    def test_guard_row_zero_seeding_is_not_a_strategy(self):
        with pytest.raises(ValueError, match="row 0"):
            SamplerConfig(0.5, seed_row_strategy="first")
        with pytest.raises(ValueError, match="unknown seed_row_strategy"):
            SamplerConfig(0.5, seed_row_strategy="failure_onset_minus_seconds")
