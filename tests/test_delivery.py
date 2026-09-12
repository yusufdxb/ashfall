"""Tests for the delivery gate.

These are the tests the suite did not have. Before this file, nothing in 252
tests would have failed if a seeded reset state were indistinguishable from an
ordinary nominal state, which is the defect that made the Phase-I result
uninformative, and nothing noticed that one failure mode cannot be delivered
at all.

The six-mode table below is the important one: it pins, from this repo's own
generator, which modes a fixed 0.5 s rollback actually delivers. It
independently reproduces the Gate A (H0) verdict on the CPU.
"""

from __future__ import annotations

import math

import pytest

from ashfall.capsule import CapsuleFrame, FailureCapsule, capsules_from_rows
from ashfall.delivery import (
    RESTORABLE_CHANNELS,
    DeliveryEvidence,
    NominalReference,
    assert_deliverable,
    assert_delivers,
    frame_features,
    implausible_reset_fields,
    is_deliverable,
    measure_delivery,
    quat_xyzw_to_roll_pitch,
    resolve_developing_seed_row,
    signature_channels,
    undeliverable_channels,
    zero_filled_channels,
)
from ashfall.synth import generator as g

_GENERATORS = {
    "attitude": g.generate_attitude_failure,
    "collapse": g.generate_collapse_failure,
    "slip": g.generate_slip_failure,
    "stumble": g.generate_stumble_failure,
    "contact_loss": g.generate_contact_loss_failure,
    "command_mismatch": g.generate_command_mismatch_failure,
}


def _capsule(mode: str, seed: int = 42) -> FailureCapsule:
    rows = _GENERATORS[mode](seed=seed)
    return capsules_from_rows(
        rows, source="synthetic_fixture", robot="go2", control_dt=0.02
    )[0]


def _frame(
    *,
    height: float = 0.30,
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    vx: float = 0.5,
    cmd_vx: float = 0.5,
    joint_vel: float = 0.0,
    contact: float = 50.0,
    t: float = 0.0,
) -> CapsuleFrame:
    return CapsuleFrame(
        timestamp_s=t,
        base_pos=(0.0, 0.0, height),
        base_quat=quat,
        base_lin_vel_body=(vx, 0.0, 0.0),
        base_ang_vel_body=(0.0, 0.0, 0.0),
        joint_pos=(0.0,) * 12,
        joint_vel=(joint_vel,) * 12,
        command_vel=(cmd_vx, 0.0, 0.0),
        contact_forces=(contact,) * 4,
    )


class TestRestoreContract:
    def test_restorable_channels_match_the_simulator_contract(self):
        """The declared contract must equal what the restore API can write.

        Declared locally so the gate runs without the simulator installed, so
        this is the cross-check. If Phoenix gains or loses a writable channel
        the gate's deliverability verdicts silently go stale without it.
        """
        pytest.importorskip("torch")
        pytest.importorskip("phoenix")
        from phoenix.replay.trajectory_reader import InitialState

        from ashfall.backends.phoenix_compat import (
            INITIAL_STATE_METADATA_FIELDS,
            KINEMATIC_RESTORE_FIELDS,
            check_phoenix_interface,
        )

        # InitialState carries metadata about HOW to restore (position frame,
        # controller history, declared environment parameters) beside the
        # kinematic channels that are actually written. Only the latter are
        # the restore contract; the metadata list is pinned separately so a new
        # metadata field is a deliberate classification, not a silent widening.
        kinematic = {
            name
            for name in InitialState.__dataclass_fields__
            if name not in INITIAL_STATE_METADATA_FIELDS
        }
        assert set(RESTORABLE_CHANNELS) == kinematic == set(KINEMATIC_RESTORE_FIELDS)
        report = check_phoenix_interface()
        assert report.compatible, report.problems

    def test_contact_forces_is_not_restorable(self):
        # Contact force is an output of the physics engine given pose, joint
        # state and friction. No restore call writes it, which is why one mode
        # is undeliverable.
        assert "contact_forces" not in RESTORABLE_CHANNELS


class TestDeliverability:
    @pytest.mark.parametrize(
        "mode", ["attitude", "collapse", "slip", "stumble", "command_mismatch"]
    )
    def test_five_modes_are_deliverable(self, mode):
        assert is_deliverable(mode)
        assert undeliverable_channels(mode) == ()
        assert_deliverable(mode)  # does not raise

    def test_contact_loss_is_undeliverable_by_construction(self):
        assert not is_deliverable("contact_loss")
        assert undeliverable_channels("contact_loss") == ("contact_forces",)
        with pytest.raises(ValueError, match="undeliverable"):
            assert_deliverable("contact_loss")

    def test_every_mode_has_registered_signature_channels(self):
        for mode in _GENERATORS:
            assert signature_channels(mode)

    def test_unknown_mode_raises_rather_than_defaulting_to_deliverable(self):
        with pytest.raises(ValueError, match="unknown failure mode"):
            signature_channels("unclassified")


class TestSixModeDeliveryTable:
    """What a fixed 0.5 s rollback actually delivers, per mode.

    Measured from this repo's own generator at dt=0.02. Gate A reached the same
    verdict against the simulator: command_mismatch was the only mode that
    passed, contact_loss was not deliverable, and the rest seeded states
    indistinguishable from nominal.
    """

    # mode -> (onset row, fixed-0.5s seed row, seed departure in sigma)
    EXPECTED = {
        "attitude": (69, 44, -1.52),
        "collapse": (62, 37, -0.98),
        "slip": (70, 45, -0.12),
        "stumble": (50, 25, -1.11),
        "contact_loss": (55, 30, -0.19),
        "command_mismatch": (80, 55, +22.21),
    }

    @pytest.mark.parametrize("mode", list(EXPECTED))
    def test_onset_and_fixed_rollback_rows(self, mode):
        onset, seed_row, _z = self.EXPECTED[mode]
        capsule = _capsule(mode)
        assert capsule.failure_onset_index == onset
        assert capsule.resolve_seed_index() == seed_row

    @pytest.mark.parametrize("mode", list(EXPECTED))
    def test_fixed_rollback_departure_is_pinned(self, mode):
        _onset, seed_row, expected_z = self.EXPECTED[mode]
        evidence = measure_delivery(_capsule(mode), seed_row)
        assert evidence.seed_departure_z == pytest.approx(expected_z, abs=0.05)

    @pytest.mark.parametrize(
        "mode", ["attitude", "collapse", "slip", "stumble"]
    )
    def test_fixed_rollback_is_refused_as_indistinguishable(self, mode):
        capsule = _capsule(mode)
        with pytest.raises(ValueError, match="indistinguishable"):
            assert_delivers(capsule, capsule.resolve_seed_index())

    def test_fixed_rollback_is_refused_for_contact_loss_before_anything_is_measured(
        self,
    ):
        capsule = _capsule("contact_loss")
        with pytest.raises(ValueError, match="undeliverable"):
            assert_delivers(capsule, capsule.resolve_seed_index())

    def test_command_mismatch_is_the_only_mode_the_fixed_rollback_delivers(self):
        capsule = _capsule("command_mismatch")
        evidence = assert_delivers(capsule, capsule.resolve_seed_index())
        assert evidence.delivers
        assert evidence.feature == "speed_error_mps"
        assert evidence.seed_departure_z > 3.0

        delivered = []
        for mode in self.EXPECTED:
            candidate = _capsule(mode)
            try:
                assert_delivers(candidate, candidate.resolve_seed_index())
                delivered.append(mode)
            except ValueError:
                pass
        assert delivered == ["command_mismatch"]


class TestDevelopmentStrategy:
    @pytest.mark.parametrize(
        "mode,expected_row", [("attitude", 60), ("collapse", 57), ("slip", 60)]
    )
    def test_fraction_of_development_finds_a_departed_row(self, mode, expected_row):
        capsule = _capsule(mode)
        row = resolve_developing_seed_row(capsule)
        assert row == expected_row
        assert row > capsule.resolve_seed_index(), (
            "the development strategy should sit later than the fixed rollback, "
            "since the fixed rollback lands in the nominal prefix"
        )
        evidence = assert_delivers(capsule, row)
        assert evidence.delivers
        assert abs(evidence.seed_departure_z) > 3.0

    def test_strategy_is_reachable_through_the_capsule(self):
        capsule = _capsule("slip")
        assert capsule.resolve_seed_index("failure_onset_minus_fraction") == 60

    def test_instantaneous_failure_has_no_deliverable_pre_onset_state(self):
        # stumble's flag fires on the first failure row, so there is no frame
        # where the stumble is developing but has not happened. Refusing loudly
        # is the correct answer; silently seeding row 25 is what shipped.
        capsule = _capsule("stumble")
        with pytest.raises(ValueError, match="no pre-onset frame"):
            resolve_developing_seed_row(capsule)

    def test_fraction_zero_seeds_the_start_of_development(self):
        capsule = _capsule("slip")
        early = resolve_developing_seed_row(capsule, fraction=0.0)
        late = resolve_developing_seed_row(capsule, fraction=0.9)
        assert early < late < capsule.failure_onset_index


class TestDistinguishability:
    def test_a_constant_trajectory_delivers_nothing(self):
        """A fixture whose frames are all identical cannot deliver a failure.

        The canonical demo fixture was exactly this: 80 byte-identical frames
        modulo timestamp, so its "pre-failure" seed row equalled row 0. It was
        the fixture for 36 tests.
        """
        frames = tuple(_frame(t=i * 0.02) for i in range(80))
        capsule = FailureCapsule(
            source="synthetic_fixture",
            robot="go2",
            policy_id="baseline",
            timestamp="2026-01-01T00:00:00+00:00",
            control_dt=0.02,
            failure_mode="slip",
            failure_onset_index=50,
            pre_failure_start_index=0,
            post_failure_end_index=79,
            frames=frames,
        )
        with pytest.raises(ValueError, match="indistinguishable"):
            assert_delivers(capsule, 25)

    def test_departure_in_the_wrong_direction_is_refused(self):
        # Labelled collapse, but the base rises before onset.
        frames = tuple(_frame(height=0.30 + 0.002 * i, t=i * 0.02) for i in range(60))
        capsule = FailureCapsule(
            source="synthetic_fixture",
            robot="go2",
            policy_id="baseline",
            timestamp="2026-01-01T00:00:00+00:00",
            control_dt=0.02,
            failure_mode="collapse",
            failure_onset_index=50,
            pre_failure_start_index=25,
            post_failure_end_index=59,
            frames=frames,
        )
        with pytest.raises(ValueError, match="not toward"):
            assert_delivers(capsule, 45)

    def test_seeding_at_or_past_onset_is_refused(self):
        capsule = _capsule("command_mismatch")
        with pytest.raises(ValueError, match="does not hold its departure"):
            assert_delivers(capsule, capsule.failure_onset_index)

    def test_evidence_records_what_was_measured(self):
        capsule = _capsule("command_mismatch")
        evidence = measure_delivery(capsule, capsule.resolve_seed_index())
        assert isinstance(evidence, DeliveryEvidence)
        payload = evidence.to_dict()
        assert payload["delivers"] is True
        assert payload["failure_mode"] == "command_mismatch"
        assert payload["reference_frames"] == 32  # 40 percent of onset row 80
        assert set(payload) >= {
            "seed_row",
            "onset_row",
            "feature",
            "seed_departure_z",
            "onset_departure_z",
            "expected_sign",
            "distinguishable",
            "directional",
            "sustained",
        }


class TestZeroFilledCaptures:
    """The real GO2 captures carry identically-zero channels where nothing was
    measured. The seeding path accepted them; the labelling path already did
    not."""

    @staticmethod
    def _hardware_shaped_frame(t: float) -> CapsuleFrame:
        # Exactly the Gate 7 live-capture pattern: quaternion, angular rate,
        # joint position and joint velocity recorded; everything else zeroed.
        return CapsuleFrame(
            timestamp_s=t,
            base_pos=(0.0, 0.0, 0.0),
            base_quat=(0.0, 0.0, 0.0, 1.0),
            base_lin_vel_body=(0.0, 0.0, 0.0),
            base_ang_vel_body=(0.01, 0.0, 0.0),
            joint_pos=(0.1,) * 12,
            joint_vel=(0.2,) * 12,
            command_vel=(0.0, 0.0, 0.0),
            contact_forces=(0.0, 0.0, 0.0, 0.0),
        )

    def test_zero_filled_channels_are_reported(self):
        frame = self._hardware_shaped_frame(0.0)
        reported = zero_filled_channels(frame)
        assert set(reported) == {
            "base_pos",
            "base_lin_vel_body",
            "command_vel",
            "contact_forces",
        }

    def test_zero_base_height_is_implausible_not_merely_missing(self):
        frame = self._hardware_shaped_frame(0.0)
        assert frame.missing_reset_fields == ()  # presence is not plausibility
        assert implausible_reset_fields(frame) == ("base_pos",)

    def test_reset_frame_refuses_a_zero_filled_capture(self):
        frames = tuple(self._hardware_shaped_frame(i * 0.02) for i in range(60))
        capsule = FailureCapsule(
            source="hardware",
            robot="go2",
            policy_id="baseline",
            timestamp="2026-01-01T00:00:00+00:00",
            control_dt=0.02,
            failure_mode="slip",
            failure_onset_index=50,
            pre_failure_start_index=25,
            post_failure_end_index=59,
            frames=frames,
        )
        with pytest.raises(ValueError, match="physically impossible"):
            capsule.reset_frame()

    def test_a_stationary_robot_is_reported_but_not_refused(self):
        """A count of zero channels cannot be the refusal criterion.

        A robot standing still legitimately has zero velocity, zero command
        and zero joint speed, so a count-based rule refuses correct data. The
        channels are reported for review; the refusal keys on the impossible
        value instead.
        """
        standing = _frame(vx=0.0, cmd_vx=0.0)
        assert len(zero_filled_channels(standing)) >= 3  # reported
        assert implausible_reset_fields(standing) == ()  # but not impossible

        frames = tuple(_frame(vx=0.0, cmd_vx=0.0, t=i * 0.02) for i in range(60))
        capsule = FailureCapsule(
            source="synthetic_fixture",
            robot="go2",
            policy_id="baseline",
            timestamp="2026-01-01T00:00:00+00:00",
            control_dt=0.02,
            failure_mode="slip",
            failure_onset_index=50,
            pre_failure_start_index=25,
            post_failure_end_index=59,
            frames=frames,
        )
        assert capsule.reset_frame() is frames[25]  # accepted


class TestFeatures:
    def test_features_are_absent_rather_than_zero_when_unrecorded(self):
        bare = CapsuleFrame(timestamp_s=0.0, base_quat=(0.0, 0.0, 0.0, 1.0))
        features = frame_features(bare)
        assert "tilt_rad" in features
        assert "base_height_m" not in features
        assert "speed_error_mps" not in features

    def test_roll_pitch_from_identity_quaternion_is_level(self):
        roll, pitch = quat_xyzw_to_roll_pitch((0.0, 0.0, 0.0, 1.0))
        assert roll == pytest.approx(0.0)
        assert pitch == pytest.approx(0.0)

    def test_roll_pitch_recovers_a_known_pitch(self):
        angle = 0.4
        quat = (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))
        roll, pitch = quat_xyzw_to_roll_pitch(quat)
        assert pitch == pytest.approx(angle, abs=1e-9)
        assert roll == pytest.approx(0.0, abs=1e-9)

    def test_reference_needs_two_frames(self):
        with pytest.raises(ValueError, match="at least two frames"):
            NominalReference.from_frames([_frame()])

    def test_std_floor_prevents_infinite_departure_on_a_constant_reference(self):
        reference = NominalReference.from_frames(
            [_frame(t=i * 0.02) for i in range(10)]
        )
        departure = reference.departure(_frame(height=0.30))
        assert all(math.isfinite(z) for z in departure.values())
        assert departure["base_height_m"] == pytest.approx(0.0)


def test_transient_excursion_that_recovers_is_refused() -> None:
    """A departure that lapses back to nominal before onset is not this failure.

    Sustained is the weaker of the two possible requirements, since most of
    these failures plateau rather than keep growing, so it has to at least
    catch a spike that recovered.
    """
    heights = [0.30] * 20 + [0.18] * 5 + [0.30] * 20 + [0.10] * 15
    frames = tuple(
        _frame(height=h, t=i * 0.02) for i, h in enumerate(heights)
    )
    capsule = FailureCapsule(
        source="synthetic_fixture",
        robot="go2",
        policy_id="baseline",
        timestamp="2026-01-01T00:00:00+00:00",
        control_dt=0.02,
        failure_mode="collapse",
        failure_onset_index=45,
        pre_failure_start_index=20,
        post_failure_end_index=59,
        frames=frames,
    )
    # Row 22 sits in the transient dip: distinguishable and directional, but it
    # recovers to nominal before the real collapse at row 45.
    evidence = measure_delivery(capsule, 22)
    assert evidence.distinguishable
    assert evidence.directional
    assert not evidence.sustained
    with pytest.raises(ValueError, match="does not hold its departure"):
        assert_delivers(capsule, 22)
