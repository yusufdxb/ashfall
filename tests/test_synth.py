"""Tests for synthetic failure generation."""

import tempfile

import numpy as np
import pyarrow.parquet as pq
import pytest

from ashfall.synth.generator import (
    GENERATORS,
    SCHEMA,
    generate_all_failures,
    generate_attitude_failure,
    generate_collapse_failure,
    generate_command_mismatch_failure,
    generate_contact_loss_failure,
    generate_slip_failure,
    generate_stumble_failure,
)
from ashfall.taxonomy.detector import FailureDetector, FailureMode


class TestGenerators:
    @pytest.mark.parametrize(
        "gen_fn,mode",
        [
            (generate_attitude_failure, FailureMode.ATTITUDE),
            (generate_collapse_failure, FailureMode.COLLAPSE),
            (generate_slip_failure, FailureMode.SLIP),
            (generate_stumble_failure, FailureMode.STUMBLE),
            (generate_contact_loss_failure, FailureMode.CONTACT_LOSS),
            (generate_command_mismatch_failure, FailureMode.COMMAND_MISMATCH),
        ],
    )
    def test_generator_produces_valid_rows(self, gen_fn, mode):
        rows = gen_fn(n_stable=10, n_failure=10, seed=0)
        assert len(rows) == 20

        # Check schema fields present
        for row in rows:
            assert "step" in row
            assert "timestamp_s" in row
            assert "base_pos" in row
            assert "failure_flag" in row
            assert "failure_mode" in row

        # At least some rows should be flagged as failures
        failure_rows = [r for r in rows if r["failure_flag"]]
        assert len(failure_rows) > 0

        # Failure mode label should match
        for r in failure_rows:
            assert r["failure_mode"] == mode.value

    @pytest.mark.parametrize(
        "gen_fn",
        [
            generate_attitude_failure,
            generate_collapse_failure,
            generate_slip_failure,
            generate_stumble_failure,
            generate_contact_loss_failure,
            generate_command_mismatch_failure,
        ],
    )
    def test_generator_deterministic(self, gen_fn):
        rows1 = gen_fn(seed=42)
        rows2 = gen_fn(seed=42)
        for r1, r2 in zip(rows1, rows2):
            assert r1["step"] == r2["step"]
            assert r1["failure_flag"] == r2["failure_flag"]
            np.testing.assert_allclose(r1["base_pos"], r2["base_pos"])


class TestGenerateAllFailures:
    def test_generates_correct_number_of_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_all_failures(tmpdir, n_variants=2, seed=0)
            assert len(paths) == len(GENERATORS) * 2
            for p in paths:
                assert p.exists()
                assert p.suffix == ".parquet"

    def test_parquet_schema_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_all_failures(tmpdir, n_variants=1, seed=0)
            for p in paths:
                table = pq.read_table(p)
                for field in SCHEMA:
                    assert field.name in table.column_names


class TestDetectorOnSyntheticData:
    """Validate that the failure detector fires correctly on synthetic trajectories."""

    def test_attitude_detected(self):
        rows = generate_attitude_failure(n_stable=20, n_failure=20, seed=0)
        detector = FailureDetector()
        all_events = []
        for row in rows:
            events = detector.step(
                timestamp_s=row["timestamp_s"],
                pitch_rad=_euler_pitch_from_quat(row["base_quat"]),
                roll_rad=_euler_roll_from_quat(row["base_quat"]),
                base_height_m=row["base_pos"][2],
                cmd_lin_vel=np.array(row["command_vel"][:2]),
                actual_lin_vel=np.array(row["base_lin_vel_body"][:2]),
                joint_vel=np.array(row["joint_vel"]),
                contact_forces=np.array(row["contact_forces"]),
            )
            all_events.extend(events)
        attitude_events = [e for e in all_events if e.mode == FailureMode.ATTITUDE]
        assert len(attitude_events) >= 1

    def test_collapse_detected(self):
        rows = generate_collapse_failure(n_stable=20, n_failure=20, seed=0)
        detector = FailureDetector()
        all_events = []
        for row in rows:
            events = detector.step(
                timestamp_s=row["timestamp_s"],
                pitch_rad=0.0,
                roll_rad=0.0,
                base_height_m=row["base_pos"][2],
                cmd_lin_vel=np.array(row["command_vel"][:2]),
                actual_lin_vel=np.array(row["base_lin_vel_body"][:2]),
                joint_vel=np.array(row["joint_vel"]),
                contact_forces=np.array(row["contact_forces"]),
            )
            all_events.extend(events)
        collapse_events = [e for e in all_events if e.mode == FailureMode.COLLAPSE]
        assert len(collapse_events) >= 1

    def test_stumble_detected(self):
        rows = generate_stumble_failure(n_stable=20, n_failure=10, seed=0)
        detector = FailureDetector()
        all_events = []
        for row in rows:
            events = detector.step(
                timestamp_s=row["timestamp_s"],
                pitch_rad=0.0,
                roll_rad=0.0,
                base_height_m=row["base_pos"][2],
                cmd_lin_vel=np.array(row["command_vel"][:2]),
                actual_lin_vel=np.array(row["base_lin_vel_body"][:2]),
                joint_vel=np.array(row["joint_vel"]),
                contact_forces=np.array(row["contact_forces"]),
            )
            all_events.extend(events)
        stumble_events = [e for e in all_events if e.mode == FailureMode.STUMBLE]
        assert len(stumble_events) >= 1


def _euler_pitch_from_quat(q: list[float]) -> float:
    """Extract pitch from xyzw quaternion (approximate)."""
    x, y, z, w = q
    return float(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))


def _euler_roll_from_quat(q: list[float]) -> float:
    """Extract roll from xyzw quaternion (approximate)."""
    x, y, z, w = q
    return float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
