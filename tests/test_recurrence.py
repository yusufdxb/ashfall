"""Tests for the paired per-seed, per-mode recurrence table."""

from __future__ import annotations

import numpy as np
import pytest

from ashfall.analysis.recurrence import (
    MODE_UNKNOWN,
    REQUIRED_SIGNALS,
    TABLE_MODES,
    EpisodeTelemetry,
    LabeledEpisode,
    RecurrenceError,
    attach_modes,
    detect_episode_mode,
    episode_key,
    missing_signals,
    paired_recurrence_table,
    recurrence_deltas_by_mode,
    unknown_fraction,
)
from ashfall.evaluation.episode_records import EpisodeRecord
from ashfall.taxonomy.detector import FailureMode


def make_record(*, seed: int, episode_id: int, success: bool, run_id: str = "run") -> EpisodeRecord:
    steps = 100
    return EpisodeRecord(
        run_id=run_id,
        episode_id=episode_id,
        seed=seed,
        env_index=episode_id % 4,
        terrain_id="rough",
        challenge_id="ff0p0",
        success=success,
        termination_reason="time_out" if success else "termination",
        time_to_failure_steps=-1 if success else steps,
        time_to_failure_s=-1.0 if success else steps * 0.02,
        episode_return=10.0,
        episode_length_steps=steps,
        episode_length_s=steps * 0.02,
        mean_lin_vel_error_mps=0.2,
        max_lin_vel_error_mps=0.6,
        mean_ang_vel_error_radps=0.1,
        max_ang_vel_error_radps=0.3,
        control_dt_s=0.02,
        policy_path="/ckpt/a.pt",
        policy_sha256="ab" * 32,
    )


def flat_telemetry(n: int = 60, **overrides) -> EpisodeTelemetry:
    """A nominal, non-failing episode: upright, at height, tracking its command."""
    base = dict(
        dt_s=0.02,
        pitch_rad=np.zeros(n),
        roll_rad=np.zeros(n),
        base_height_m=np.full(n, 0.32),
        cmd_lin_vel=np.tile(np.array([0.5, 0.0]), (n, 1)),
        actual_lin_vel=np.tile(np.array([0.5, 0.0]), (n, 1)),
    )
    base.update(overrides)
    return EpisodeTelemetry(**base)


class TestModeDetection:
    def test_attitude_detected(self):
        n = 60
        pitch = np.zeros(n)
        pitch[30:] = 1.2  # beyond the 0.8 rad attitude threshold
        mode, reason = detect_episode_mode(flat_telemetry(n, pitch_rad=pitch))
        assert mode == FailureMode.ATTITUDE.value
        assert reason == ""

    def test_collapse_outranks_attitude_by_severity(self):
        n = 60
        pitch = np.zeros(n)
        pitch[30:] = 1.2
        height = np.full(n, 0.32)
        height[30:] = 0.05
        mode, _ = detect_episode_mode(flat_telemetry(n, pitch_rad=pitch, base_height_m=height))
        assert mode == FailureMode.COLLAPSE.value

    def test_slip_detected(self):
        n = 120
        actual = np.zeros((n, 2))  # commanded 0.5 m/s, achieving nothing
        mode, _ = detect_episode_mode(flat_telemetry(n, actual_lin_vel=actual))
        assert mode == FailureMode.SLIP.value

    def test_nominal_episode_is_unknown_not_guessed(self):
        mode, reason = detect_episode_mode(flat_telemetry())
        assert mode == MODE_UNKNOWN
        assert "detector fired no event" in reason


class TestUnobservableSignals:
    def test_no_telemetry_is_unknown(self):
        mode, reason = detect_episode_mode(None)
        assert mode == MODE_UNKNOWN
        for name in REQUIRED_SIGNALS:
            assert name in reason

    def test_missing_single_signal_is_unknown(self):
        telem = flat_telemetry()
        telem.base_height_m = None
        mode, reason = detect_episode_mode(telem)
        assert mode == MODE_UNKNOWN
        assert "base_height_m" in reason

    def test_hardware_zero_filled_base_height_is_unknown(self):
        # The GO2 hardware path hardcodes base_pos to zeros. An all-zero base
        # height must not be read as a collapse.
        n = 60
        telem = flat_telemetry(n, base_height_m=np.zeros(n))
        mode, reason = detect_episode_mode(telem)
        assert mode == MODE_UNKNOWN
        assert "hardware zero-filled" in reason

    def test_zero_filled_contact_forces_do_not_trigger_contact_loss(self):
        # Zero contact forces would otherwise read as all four feet airborne.
        n = 120
        telem = flat_telemetry(n, contact_forces=np.zeros((n, 4)))
        mode, _ = detect_episode_mode(telem)
        assert mode == MODE_UNKNOWN

    def test_missing_signals_lists_every_required_name(self):
        assert missing_signals(None) == list(REQUIRED_SIGNALS)
        assert missing_signals(flat_telemetry()) == []


class TestAttachModes:
    def test_success_carries_no_mode(self):
        rec = make_record(seed=1, episode_id=0, success=True)
        labeled = attach_modes([rec])
        assert labeled[0].mode is None

    def test_failure_without_telemetry_is_unknown(self):
        rec = make_record(seed=1, episode_id=0, success=False)
        labeled = attach_modes([rec])
        assert labeled[0].mode == MODE_UNKNOWN
        assert labeled[0].unknown_reason

    def test_mapping_lookup_by_episode_key(self):
        rec = make_record(seed=3, episode_id=7, success=False)
        n = 120
        actual = np.zeros((n, 2))
        telem = {episode_key(rec): flat_telemetry(n, actual_lin_vel=actual)}
        labeled = attach_modes([rec], telem)
        assert labeled[0].mode == FailureMode.SLIP.value

    def test_callable_lookup(self):
        rec = make_record(seed=3, episode_id=7, success=False)
        n = 120
        labeled = attach_modes([rec], lambda r: flat_telemetry(n, actual_lin_vel=np.zeros((n, 2))))
        assert labeled[0].mode == FailureMode.SLIP.value

    def test_unknown_fraction_counts_only_failures(self):
        eps = [
            LabeledEpisode(make_record(seed=1, episode_id=0, success=True), None),
            LabeledEpisode(make_record(seed=1, episode_id=1, success=False), MODE_UNKNOWN),
            LabeledEpisode(
                make_record(seed=1, episode_id=2, success=False), FailureMode.SLIP.value
            ),
        ]
        assert unknown_fraction(eps) == pytest.approx(0.5)

    def test_unknown_fraction_empty_is_zero(self):
        assert unknown_fraction([]) == 0.0


def labeled(seed: int, modes: list) -> list[LabeledEpisode]:
    """Build episodes for one seed. ``None`` in ``modes`` means a success."""
    out = []
    for i, mode in enumerate(modes):
        out.append(
            LabeledEpisode(
                record=make_record(seed=seed, episode_id=i, success=mode is None),
                mode=mode,
            )
        )
    return out


class TestPairedRecurrenceTable:
    def test_synthetic_fixture_computes_correctly(self):
        slip = FailureMode.SLIP.value
        collapse = FailureMode.COLLAPSE.value
        # Seed 1: baseline 2/4 slip, treatment 1/4 slip.
        # Seed 2: baseline 1/4 collapse, treatment 0/4 collapse.
        baseline = labeled(1, [slip, slip, None, None]) + labeled(2, [collapse, None, None, None])
        treatment = labeled(1, [slip, None, None, None]) + labeled(2, [None, None, None, None])

        rows = paired_recurrence_table(baseline, treatment)
        cells = {(r.seed, r.mode): r for r in rows}

        s1 = cells[(1, slip)]
        assert (s1.baseline_episodes, s1.baseline_failures) == (4, 2)
        assert s1.baseline_rate == pytest.approx(0.5)
        assert (s1.treatment_episodes, s1.treatment_failures) == (4, 1)
        assert s1.treatment_rate == pytest.approx(0.25)
        assert s1.delta_rate == pytest.approx(-0.25)

        s2 = cells[(2, collapse)]
        assert s2.baseline_rate == pytest.approx(0.25)
        assert s2.treatment_rate == pytest.approx(0.0)
        assert s2.delta_rate == pytest.approx(-0.25)

        # A mode nothing fired on is still present, as an explicit zero.
        stumble = cells[(1, FailureMode.STUMBLE.value)]
        assert stumble.baseline_failures == 0
        assert stumble.delta_rate == pytest.approx(0.0)

    def test_every_mode_and_unknown_emitted_per_seed(self):
        baseline = labeled(1, [None, None])
        treatment = labeled(1, [None, None])
        rows = paired_recurrence_table(baseline, treatment)
        assert {r.mode for r in rows} == set(TABLE_MODES)
        assert MODE_UNKNOWN in TABLE_MODES
        assert len(rows) == len(TABLE_MODES)

    def test_unequal_episode_counts_use_rates(self):
        slip = FailureMode.SLIP.value
        baseline = labeled(1, [slip, None])  # 1/2
        treatment = labeled(1, [slip, None, None, None])  # 1/4
        row = next(r for r in paired_recurrence_table(baseline, treatment) if r.mode == slip)
        assert row.baseline_rate == pytest.approx(0.5)
        assert row.treatment_rate == pytest.approx(0.25)
        assert row.delta_rate == pytest.approx(-0.25)

    def test_only_paired_seeds_are_emitted(self):
        baseline = labeled(1, [None]) + labeled(2, [None])
        treatment = labeled(2, [None]) + labeled(3, [None])
        rows = paired_recurrence_table(baseline, treatment)
        assert {r.seed for r in rows} == {2}

    def test_no_shared_seed_raises(self):
        with pytest.raises(RecurrenceError, match="no seed appears in both arms"):
            paired_recurrence_table(labeled(1, [None]), labeled(2, [None]))

    def test_deltas_by_mode_groups_per_seed_values(self):
        slip = FailureMode.SLIP.value
        baseline = labeled(1, [slip, None]) + labeled(2, [slip, None])
        treatment = labeled(1, [None, None]) + labeled(2, [slip, None])
        deltas = recurrence_deltas_by_mode(paired_recurrence_table(baseline, treatment))
        assert deltas[slip] == [pytest.approx(-0.5), pytest.approx(0.0)]

    def test_row_to_dict_round_trip(self):
        rows = paired_recurrence_table(labeled(1, [None]), labeled(1, [None]))
        d = rows[0].to_dict()
        assert d["seed"] == 1
        assert set(d) == {
            "seed",
            "mode",
            "baseline_episodes",
            "baseline_failures",
            "baseline_rate",
            "treatment_episodes",
            "treatment_failures",
            "treatment_rate",
            "delta_rate",
        }


class TestEndToEnd:
    def test_records_to_table_via_attach(self):
        slip = FailureMode.SLIP.value
        n = 120
        slipping = flat_telemetry(n, actual_lin_vel=np.zeros((n, 2)))

        base_recs = [make_record(seed=5, episode_id=i, success=i > 0) for i in range(4)]
        treat_recs = [make_record(seed=5, episode_id=i, success=True) for i in range(4)]
        telem = {episode_key(base_recs[0]): slipping}

        rows = paired_recurrence_table(
            attach_modes(base_recs, telem), attach_modes(treat_recs, telem)
        )
        row = next(r for r in rows if r.mode == slip)
        assert row.baseline_failures == 1
        assert row.treatment_failures == 0
        assert row.delta_rate == pytest.approx(-0.25)
