"""Tests for the extended failure taxonomy detector."""

import numpy as np

from ashfall.taxonomy.detector import (
    FailureDetector,
    FailureMode,
    FailureThresholds,
)
from ashfall.taxonomy.schema import TAXONOMY, taxonomy_table_rows


class TestFailureDetector:
    def setup_method(self):
        self.detector = FailureDetector()
        self.stable_kwargs = {
            "timestamp_s": 0.0,
            "pitch_rad": 0.0,
            "roll_rad": 0.0,
            "base_height_m": 0.30,
            "cmd_lin_vel": np.array([0.5, 0.0]),
            "actual_lin_vel": np.array([0.48, 0.01]),
            "joint_vel": np.zeros(12),
            "contact_forces": np.array([50.0, 50.0, 50.0, 50.0]),
        }

    def test_no_failure_on_stable_telemetry(self):
        events = self.detector.step(**self.stable_kwargs)
        assert events == []

    def test_attitude_pitch(self):
        kwargs = {**self.stable_kwargs, "pitch_rad": 0.9}
        events = self.detector.step(**kwargs)
        assert len(events) == 1
        assert events[0].mode == FailureMode.ATTITUDE
        assert events[0].detail["pitch"] == 0.9

    def test_attitude_roll(self):
        kwargs = {**self.stable_kwargs, "roll_rad": -0.7}
        events = self.detector.step(**kwargs)
        assert len(events) == 1
        assert events[0].mode == FailureMode.ATTITUDE

    def test_collapse(self):
        kwargs = {**self.stable_kwargs, "base_height_m": 0.10}
        events = self.detector.step(**kwargs)
        assert len(events) == 1
        assert events[0].mode == FailureMode.COLLAPSE

    def test_slip_requires_sustained_duration(self):
        # First step: slip starts but not yet sustained
        kwargs = {
            **self.stable_kwargs,
            "cmd_lin_vel": np.array([0.5, 0.0]),
            "actual_lin_vel": np.array([0.01, 0.01]),
        }
        events = self.detector.step(**kwargs)
        # May detect command_mismatch, but not slip yet
        slip_events = [e for e in events if e.mode == FailureMode.SLIP]
        assert len(slip_events) == 0

        # Step at t=0.6 — past slip_min_duration_s
        kwargs["timestamp_s"] = 0.6
        events = self.detector.step(**kwargs)
        slip_events = [e for e in events if e.mode == FailureMode.SLIP]
        assert len(slip_events) == 1

    def test_stumble_joint_vel_spike(self):
        jv = np.zeros(12)
        jv[3:6] = 20.0  # spike on second leg
        kwargs = {
            **self.stable_kwargs,
            "joint_vel": jv,
            "contact_forces": np.array([50, 50, 50, 50]),
        }
        events = self.detector.step(**kwargs)
        stumble_events = [e for e in events if e.mode == FailureMode.STUMBLE]
        assert len(stumble_events) == 1
        assert stumble_events[0].detail["max_joint_vel"] == 20.0

    def test_stumble_not_triggered_midair(self):
        """Stumble requires feet in contact — all feet airborne should not trigger."""
        jv = np.zeros(12)
        jv[0:3] = 20.0
        kwargs = {
            **self.stable_kwargs,
            "joint_vel": jv,
            "contact_forces": np.array([1.0, 1.0, 1.0, 1.0]),  # below threshold
        }
        events = self.detector.step(**kwargs)
        stumble_events = [e for e in events if e.mode == FailureMode.STUMBLE]
        assert len(stumble_events) == 0

    def test_contact_loss_sustained(self):
        kwargs = {
            **self.stable_kwargs,
            "contact_forces": np.array([1.0, 1.0, 50.0, 50.0]),
        }
        events = self.detector.step(**kwargs)
        contact_events = [e for e in events if e.mode == FailureMode.CONTACT_LOSS]
        assert len(contact_events) == 0  # not yet sustained

        kwargs["timestamp_s"] = 0.15  # past contact_loss_min_duration_s
        events = self.detector.step(**kwargs)
        contact_events = [e for e in events if e.mode == FailureMode.CONTACT_LOSS]
        assert len(contact_events) == 1

    def test_command_mismatch_sustained(self):
        kwargs = {
            **self.stable_kwargs,
            "cmd_lin_vel": np.array([0.5, 0.2]),
            "actual_lin_vel": np.array([0.1, -0.1]),
        }
        events = self.detector.step(**kwargs)
        mismatch = [e for e in events if e.mode == FailureMode.COMMAND_MISMATCH]
        assert len(mismatch) == 0  # not yet sustained

        kwargs["timestamp_s"] = 1.1
        events = self.detector.step(**kwargs)
        mismatch = [e for e in events if e.mode == FailureMode.COMMAND_MISMATCH]
        assert len(mismatch) == 1

    def test_command_mismatch_not_when_slipping(self):
        """Slip takes priority over command mismatch when actual vel is near zero."""
        kwargs = {
            **self.stable_kwargs,
            "cmd_lin_vel": np.array([0.5, 0.0]),
            "actual_lin_vel": np.array([0.01, 0.0]),
            "timestamp_s": 2.0,
        }
        events = self.detector.step(**kwargs)
        mismatch = [e for e in events if e.mode == FailureMode.COMMAND_MISMATCH]
        assert len(mismatch) == 0  # should be classified as slip, not mismatch

    def test_event_suppression(self):
        """Same mode should not fire twice within min_event_gap_s."""
        kwargs = {**self.stable_kwargs, "pitch_rad": 0.9}
        events1 = self.detector.step(**kwargs)
        assert len(events1) >= 1

        kwargs["timestamp_s"] = 0.5  # within gap
        events2 = self.detector.step(**kwargs)
        attitude_events = [e for e in events2 if e.mode == FailureMode.ATTITUDE]
        assert len(attitude_events) == 0

        kwargs["timestamp_s"] = 1.5  # past gap
        events3 = self.detector.step(**kwargs)
        attitude_events = [e for e in events3 if e.mode == FailureMode.ATTITUDE]
        assert len(attitude_events) == 1

    def test_multiple_modes_co_occur(self):
        """High pitch + low height = attitude + collapse simultaneously."""
        kwargs = {
            **self.stable_kwargs,
            "pitch_rad": 1.0,
            "base_height_m": 0.10,
        }
        events = self.detector.step(**kwargs)
        modes = {e.mode for e in events}
        assert FailureMode.ATTITUDE in modes
        assert FailureMode.COLLAPSE in modes

    def test_severity_ordering(self):
        from ashfall.taxonomy.detector import SEVERITY

        assert SEVERITY[FailureMode.COLLAPSE] > SEVERITY[FailureMode.ATTITUDE]
        assert SEVERITY[FailureMode.ATTITUDE] > SEVERITY[FailureMode.SLIP]
        assert SEVERITY[FailureMode.SLIP] > SEVERITY[FailureMode.STUMBLE]

    def test_custom_thresholds(self):
        custom = FailureThresholds(pitch_rad=0.3)
        det = FailureDetector(custom)
        kwargs = {**self.stable_kwargs, "pitch_rad": 0.4}
        events = det.step(**kwargs)
        assert any(e.mode == FailureMode.ATTITUDE for e in events)


class TestTaxonomySchema:
    def test_all_modes_in_taxonomy(self):
        for mode in FailureMode:
            assert mode in TAXONOMY

    def test_taxonomy_table_rows(self):
        rows = taxonomy_table_rows()
        assert len(rows) == len(FailureMode)
        # Sorted by severity descending
        severities = [int(r["Severity"]) for r in rows]
        assert severities == sorted(severities, reverse=True)
