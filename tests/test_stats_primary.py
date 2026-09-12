"""Primary endpoint, paired-seed inference, secondaries, recurrence and the BCa guard."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ashfall.analysis.multiseed import _exact_sign_flip_p as multiseed_sign_flip_p
from ashfall.analysis.recurrence import (
    IncidenceRow,
    RecurrenceRow,
    episode_recurrence,
    incidence_deltas_by_mode,
    paired_incidence_table,
    paired_recurrence_table,
    recurrence_deltas_by_mode,
)
from ashfall.evaluation.significance import bca_acceleration, bernoulli_arrays
from ashfall.ontology import FIRST_STUDY_PHENOTYPES
from ashfall.stats import (
    exact_sign_flip_p,
    incidence_by_training_seed,
    paired_seed_effect,
    primary_endpoint,
    reference_two_sample_bca_acceleration,
    secondary_endpoints,
    sign_flip_p_floor,
)


def record(seed, scenario, modes, *, policy=None, eval_seed=1):
    return dict(
        policy_id=policy or f"policy-{seed}",
        training_seed=seed,
        scenario_id=scenario,
        evaluation_seed=eval_seed,
        failure_modes=modes,
    )


def arm(seed_to_modes, *, prefix="p"):
    rows = []
    for seed, episodes in seed_to_modes.items():
        for i, modes in enumerate(episodes):
            rows.append(record(seed, f"s{i}", modes, policy=f"{prefix}-{seed}"))
    return rows


class TestBcaGuard:
    @pytest.mark.guard
    @pytest.mark.parametrize(
        "counts", [(7, 10, 3, 5), (90, 100, 95, 100), (1, 10, 9, 10), (50, 128, 60, 140)]
    )
    def test_production_acceleration_matches_the_jackknife_definition(self, counts):
        a, b = bernoulli_arrays(*counts)
        ours = bca_acceleration(a, b)
        reference = reference_two_sample_bca_acceleration(a, b)
        assert reference != 0.0
        assert np.sign(ours) == np.sign(reference)
        assert ours == pytest.approx(reference, rel=1e-6)
        # The historical defect was the negated influence values. It is not
        # close to the reference, so this guard would catch its return.
        assert -ours != pytest.approx(reference, rel=1e-6)

    def test_reference_rejects_degenerate_input(self):
        with pytest.raises(ValueError):
            reference_two_sample_bca_acceleration([1.0], [0.0, 1.0])


class TestIncidenceUnit:
    @pytest.mark.guard
    def test_evaluation_episodes_are_not_replicates(self):
        rows = arm({1: [["slip"], []]})
        for unit in ("evaluation_episode", "evaluation_seed", "episode"):
            with pytest.raises(ValueError, match="not independent replicates"):
                incidence_by_training_seed(rows, arm="baseline", replicate_unit=unit)
            with pytest.raises(ValueError, match="not independent replicates"):
                primary_endpoint(rows, rows, replicate_unit=unit)
            with pytest.raises(ValueError, match="not independent replicates"):
                secondary_endpoints(rows, rows, replicate_unit=unit)

    def test_one_incidence_per_training_seed(self):
        rows = arm({1: [["slip"], [], ["collapse", "attitude"], []], 2: [[], []]})
        by_seed = incidence_by_training_seed(rows, arm="baseline")
        assert by_seed[1].episodes == 4 and by_seed[1].failures == 2
        assert by_seed[1].incidence == pytest.approx(0.5)
        assert by_seed[2].incidence == 0.0
        # A phenotype outside the preregistered set does not count.
        rows = arm({1: [["contact_loss"], []]})
        assert incidence_by_training_seed(rows, arm="x")[1].failures == 0

    def test_unknown_capture_missing_identity_and_mixed_policies_refused(self):
        with pytest.raises(ValueError, match="failure_modes is None"):
            incidence_by_training_seed(arm({1: [None]}), arm="baseline")
        bad = arm({1: [[]]})
        bad[0]["training_seed"] = None
        with pytest.raises(ValueError, match="integer training_seed"):
            incidence_by_training_seed(bad, arm="baseline")
        mixed = arm({1: [[], []]})
        mixed[1]["policy_id"] = "other"
        with pytest.raises(ValueError, match="mixes policies"):
            incidence_by_training_seed(mixed, arm="baseline")
        duplicate = arm({1: [[]]}) * 2
        with pytest.raises(ValueError, match="duplicate"):
            incidence_by_training_seed(duplicate, arm="baseline")
        with pytest.raises(ValueError, match="unknown phenotypes"):
            incidence_by_training_seed(arm({1: [[]]}), ("tumble",), arm="baseline")


class TestPairedSeedEffect:
    def test_unpaired_seeds_raise(self):
        with pytest.raises(ValueError, match="unpaired training seeds"):
            paired_seed_effect({1: 0.5, 2: 0.4}, {1: 0.3, 3: 0.2})
        with pytest.raises(ValueError, match="at least two"):
            paired_seed_effect({1: 0.5}, {1: 0.3})

    def test_known_values(self):
        effect = paired_seed_effect({1: 0.5, 2: 0.4, 3: 0.6}, {1: 0.3, 2: 0.3, 3: 0.3})
        assert effect.deltas == pytest.approx((-0.2, -0.1, -0.3))
        assert effect.mean == pytest.approx(-0.2)
        assert effect.p_exact_two_sided == pytest.approx(0.25)
        assert effect.p_floor == pytest.approx(0.25) and effect.sign_carrying_seeds == 3
        assert not effect.alpha_reachable(0.05)
        assert effect.ci_low < effect.mean < effect.ci_high
        assert effect.inference_unit == "training_seed"

    def test_matches_the_archived_multiseed_test(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            deltas = list(rng.normal(size=rng.integers(1, 9)))
            assert exact_sign_flip_p(deltas) == pytest.approx(multiseed_sign_flip_p(deltas))

    @given(st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False), min_size=1, max_size=9))
    @settings(max_examples=60, deadline=None)
    def test_sign_flip_p_is_symmetric_and_bounded_by_its_floor(self, deltas):
        p = exact_sign_flip_p(deltas)
        floor = sign_flip_p_floor(deltas)
        assert p == pytest.approx(exact_sign_flip_p([-d for d in deltas]))
        assert floor - 1e-12 <= p <= 1.0
        assert floor == (1.0 if not any(deltas) else 2.0 / 2 ** sum(1 for d in deltas if d != 0))

    def test_zero_deltas_raise_the_floor(self):
        assert sign_flip_p_floor([0.1, 0.2, 0.0, -0.3]) == pytest.approx(2 / 8)
        assert sign_flip_p_floor([0.0, 0.0]) == 1.0
        assert exact_sign_flip_p([0.0, 0.0]) == 1.0


class TestEndpoints:
    def baseline(self):
        return arm(
            {
                11: [["slip"], ["slip"], [], []],
                23: [["collapse"], [], [], []],
                47: [["slip"], ["attitude"], [], []],
            },
            prefix="base",
        )

    def treatment(self):
        return arm(
            {11: [[], ["slip"], [], []], 23: [[], [], [], []], 47: [[], [], [], []]}, prefix="t"
        )

    def test_primary_endpoint_statement_and_shape(self):
        result = primary_endpoint(self.baseline(), self.treatment())
        assert result.phenotypes == FIRST_STUDY_PHENOTYPES
        assert result.n_seeds == 3 and result.inference_unit == "training_seed"
        assert result.effect.deltas == pytest.approx((-0.25, -0.25, -0.5))
        assert result.effect.p_exact_two_sided == pytest.approx(0.25)
        assert not result.rejected_at_alpha
        assert "NOT reachable" in result.statement and "p-floor 0.2500" in result.statement
        payload = result.to_dict()
        assert payload["effect"]["inference_unit"] == "training_seed"
        assert [s["incidence"] for s in payload["baseline"]] == pytest.approx([0.5, 0.25, 0.5])

    def test_unpaired_seed_between_arms_raises(self):
        treatment = [r for r in self.treatment() if r["training_seed"] != 47]
        with pytest.raises(ValueError, match="unpaired"):
            primary_endpoint(self.baseline(), treatment)

    @pytest.mark.guard
    def test_secondaries_are_holm_corrected_over_the_declared_family(self):
        result = secondary_endpoints(self.baseline(), self.treatment())
        assert result.family == FIRST_STUDY_PHENOTYPES and result.correction == "holm_step_down"
        raw = {name: effect.p_exact_two_sided for name, effect in result.effects.items()}
        for name in result.family:
            assert result.holm_adjusted_p[name] >= raw[name]
        # Every raw p at n=3 is at its 0.25 floor or 1.0; Holm over four
        # secondaries pushes the floor cases to 1.0, so nothing is
        # "significant" per phenotype where the family would not allow it.
        assert min(result.holm_adjusted_p.values()) >= min(raw.values()) * 2
        payload = result.to_dict()
        assert payload["family_size"] == len(FIRST_STUDY_PHENOTYPES)


class TestRecurrenceTerminology:
    def test_deprecated_names_warn_and_delegate(self):
        from ashfall.analysis.recurrence import LabeledEpisode
        from ashfall.evaluation.episode_records import EpisodeRecord

        def labeled(seed, modes):
            out = []
            for i, mode in enumerate(modes):
                record = EpisodeRecord(
                    run_id="r",
                    episode_id=i,
                    seed=seed,
                    env_index=0,
                    terrain_id="t",
                    challenge_id="c",
                    success=mode is None,
                    termination_reason="x",
                    time_to_failure_steps=-1,
                    time_to_failure_s=-1.0,
                    episode_return=0.0,
                    episode_length_steps=10,
                    episode_length_s=0.2,
                    mean_lin_vel_error_mps=0.0,
                    max_lin_vel_error_mps=0.0,
                    mean_ang_vel_error_radps=0.0,
                    max_ang_vel_error_radps=0.0,
                    control_dt_s=0.02,
                    policy_path="p",
                    policy_sha256="0" * 64,
                )
                out.append(LabeledEpisode(record=record, mode=mode))
            return out

        baseline = labeled(1, ["slip", None])
        treatment = labeled(1, [None, None])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old_rows = paired_recurrence_table(baseline, treatment)
            old_deltas = recurrence_deltas_by_mode(old_rows)
        assert len(caught) == 2 and all(w.category is DeprecationWarning for w in caught)
        assert "INCIDENCE" in str(caught[0].message)
        new_rows = paired_incidence_table(baseline, treatment)
        assert [r.to_dict() for r in old_rows] == [r.to_dict() for r in new_rows]
        assert old_deltas == incidence_deltas_by_mode(new_rows)
        assert RecurrenceRow is IncidenceRow

    def test_true_recurrence_needs_recovery_information(self):
        episode = dict(
            failure_modes=["slip"], failure_events=[{"mode": "slip", "timestamp_s": 1.0}]
        )
        counts = episode_recurrence(episode)
        assert counts["slip"].count is None and "recurrence is not" in counts["slip"].reason

    def test_true_recurrence_from_analyzer_counts_and_event_timestamps(self):
        analyzer = dict(
            failure_modes=["slip", "collapse"],
            repeated_events_after_recovery={"slip": 2},
        )
        counts = episode_recurrence(analyzer)
        assert counts["slip"].count == 2 and counts["collapse"].count == 0
        events = dict(
            failure_modes=["slip"],
            failure_events=[
                {"mode": "slip", "timestamp_s": 1.0, "recovered_at_s": 2.0},
                {"mode": "slip", "timestamp_s": 3.0, "recovered_at_s": None},
                {"mode": "slip", "timestamp_s": 4.0, "recovered_at_s": 5.0},
            ],
        )
        # Second event follows a recovery: one recurrence. Third follows an
        # unrecovered event: not a recurrence.
        assert episode_recurrence(events)["slip"].count == 1
        assert episode_recurrence(dict(failure_modes=None)) == {}
