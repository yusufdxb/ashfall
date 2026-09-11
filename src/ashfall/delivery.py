"""Whether a counterexample can actually be delivered to the simulator.

Three questions have to be answered before a capsule is allowed to seed a
training reset, and none of them was asked before this module existed.

1. Can the failure's signature be written at all? The state-restore API writes
   seven channels. A mode whose signature lives outside them cannot be
   delivered by any choice of seed row, parameter or strategy.
2. Is the seeded state measurably different from an ordinary nominal state? A
   fixed-seconds rollback answers this by accident at best: failure
   development time varies by a factor of six across the modes in this repo,
   so one global offset lands inside the nominal prefix for most of them.
3. Is it moving toward the failure, and does it stay there? A state can be
   distinguishable and still be departing along the wrong axis, or be a
   transient excursion that returns to nominal before the failure.

Gate A (hypothesis H0) measured exactly these and failed. This module turns
that measurement into a gate that runs on the CPU, so the failure cannot recur
silently.

Feature conventions: quaternions are xyzw, root velocities are body frame,
tilt is the magnitude of roll and pitch combined, and speed error is the
planar difference between commanded and achieved body velocity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from ashfall.capsule import CapsuleFrame, FailureCapsule

# The channels phoenix.replay.state_adapter.restore_state can write, which is
# the whole restore contract: root pose, root velocity, joint state, command.
# Declared here rather than imported so the gate runs without the simulator
# installed; tests/test_delivery.py cross-checks it against the real
# InitialState when phoenix is importable.
RESTORABLE_CHANNELS: tuple[str, ...] = (
    "base_pos",
    "base_quat",
    "base_lin_vel_body",
    "base_ang_vel_body",
    "joint_pos",
    "joint_vel",
    "command_vel",
)

# Standard deviation floors, one per scalar feature, in the feature's own
# units. They are measurement resolutions, not tuning knobs: without them a
# constant reference window divides by zero and any state looks infinitely
# distinguishable.
_STD_FLOOR: Mapping[str, float] = {
    "base_height_m": 0.005,
    "tilt_rad": 0.01,
    "speed_error_mps": 0.02,
    "max_joint_speed_radps": 0.5,
    "min_contact_n": 1.0,
}

# Which scalar feature carries each failure mode's signature, and which way it
# moves as the failure develops. Sign is +1 when the feature grows.
_MODE_SIGNATURE: Mapping[str, tuple[str, int]] = {
    "attitude": ("tilt_rad", +1),
    "collapse": ("base_height_m", -1),
    "slip": ("speed_error_mps", +1),
    "stumble": ("max_joint_speed_radps", +1),
    "contact_loss": ("min_contact_n", -1),
    "command_mismatch": ("speed_error_mps", +1),
}

# The raw channels each mode's signature is computed from. A mode is
# deliverable only if all of them are restorable.
_MODE_SIGNATURE_CHANNELS: Mapping[str, tuple[str, ...]] = {
    "attitude": ("base_quat",),
    "collapse": ("base_pos",),
    "slip": ("base_lin_vel_body", "command_vel"),
    "stumble": ("joint_vel",),
    "contact_loss": ("contact_forces",),
    "command_mismatch": ("base_lin_vel_body", "command_vel"),
}


def signature_channels(failure_mode: str) -> tuple[str, ...]:
    """Raw channels a mode's detector signature is computed from."""
    try:
        return _MODE_SIGNATURE_CHANNELS[failure_mode]
    except KeyError:
        raise ValueError(f"unknown failure mode: {failure_mode!r}") from None


def undeliverable_channels(failure_mode: str) -> tuple[str, ...]:
    """Signature channels the restore API cannot write."""
    return tuple(
        c for c in signature_channels(failure_mode) if c not in RESTORABLE_CHANNELS
    )


def is_deliverable(failure_mode: str) -> bool:
    """True when every channel carrying the signature can be written."""
    return not undeliverable_channels(failure_mode)


def assert_deliverable(failure_mode: str) -> None:
    """Refuse a mode whose signature cannot reach the simulator.

    ``contact_loss`` is the case this exists for: its generator perturbs only
    ``contact_forces``, contact force is an output of the physics engine given
    pose, joint state and friction, and no restore call writes it. Seeding such
    a capsule produces a nominal-looking episode that scores as though the
    treatment were applied.
    """
    missing = undeliverable_channels(failure_mode)
    if missing:
        raise ValueError(
            f"failure mode {failure_mode!r} is undeliverable: its signature needs "
            f"{', '.join(missing)}, which the state-restore contract "
            f"({', '.join(RESTORABLE_CHANNELS)}) cannot write. Regenerate the mode "
            "so its signature lives in a restorable channel, or exclude it."
        )


def quat_xyzw_to_roll_pitch(quat: Sequence[float]) -> tuple[float, float]:
    """Roll and pitch in radians from an xyzw quaternion rotating body to world."""
    x, y, z, w = (float(v) for v in quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(sin_pitch)


def frame_features(frame: "CapsuleFrame") -> dict[str, float]:
    """Scalar features a failure signature can be measured on.

    A feature is absent rather than zero when its channel was not recorded, so
    an unmeasured capture cannot masquerade as a measured one.
    """
    features: dict[str, float] = {}
    if frame.base_pos is not None:
        features["base_height_m"] = float(frame.base_pos[2])
    if frame.base_quat is not None:
        roll, pitch = quat_xyzw_to_roll_pitch(frame.base_quat)
        features["tilt_rad"] = math.hypot(roll, pitch)
    if frame.base_lin_vel_body is not None and frame.command_vel is not None:
        features["speed_error_mps"] = math.hypot(
            frame.command_vel[0] - frame.base_lin_vel_body[0],
            frame.command_vel[1] - frame.base_lin_vel_body[1],
        )
    if frame.joint_vel is not None:
        features["max_joint_speed_radps"] = max(abs(v) for v in frame.joint_vel)
    if frame.contact_forces is not None:
        features["min_contact_n"] = min(frame.contact_forces)
    return features


def zero_filled_channels(frame: "CapsuleFrame") -> tuple[str, ...]:
    """Channels present but identically zero.

    The GO2 logging path writes hardcoded zeros where a signal was never
    measured, so a present-and-zero channel is evidence of an unrecorded
    signal rather than a measurement of rest. ``analysis.recurrence`` already
    refuses these in the labelling path; this is the same rule for the seeding
    path, which accepted them.

    Reporting only. A count of zero channels is not a refusal criterion: a
    robot standing still legitimately has zero velocity, zero command and zero
    joint speed. The refusal is driven by ``implausible_reset_fields``, which
    keys on a value that cannot occur, and by the distinguishability
    measurement, which refuses a signature channel that never moves.
    """
    names = (*RESTORABLE_CHANNELS, "contact_forces")
    return tuple(
        name
        for name in names
        if (value := getattr(frame, name, None)) is not None
        and all(v == 0.0 for v in value)
    )


def implausible_reset_fields(frame: "CapsuleFrame") -> tuple[str, ...]:
    """Reset channels whose recorded value is physically impossible.

    Zero velocity and zero command are legal: a stationary robot has both.
    A base height of exactly zero is not, since it places the body origin on
    the ground plane, and it is the signature of the capture path filling
    ``base_pos`` with zeros.
    """
    bad = []
    if frame.base_pos is not None and frame.base_pos[2] == 0.0:
        bad.append("base_pos")
    return tuple(bad)


@dataclass(frozen=True)
class NominalReference:
    """Per-feature mean and spread over a capsule's own nominal window.

    The reference is the capsule's own early frames, not an external model of
    nominal locomotion, so the comparison holds whatever gait the recording
    used.
    """

    means: Mapping[str, float]
    stds: Mapping[str, float]
    n_frames: int

    @classmethod
    def from_frames(cls, frames: Sequence["CapsuleFrame"]) -> NominalReference:
        if len(frames) < 2:
            raise ValueError("nominal reference needs at least two frames")
        per_feature: dict[str, list[float]] = {}
        for frame in frames:
            for name, value in frame_features(frame).items():
                per_feature.setdefault(name, []).append(value)
        means, stds = {}, {}
        for name, values in per_feature.items():
            if len(values) != len(frames):
                # A channel that appears in only some frames cannot anchor a
                # reference; treat it as unmeasured rather than interpolate.
                continue
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            means[name] = mean
            stds[name] = max(math.sqrt(variance), _STD_FLOOR.get(name, 1e-6))
        return cls(means=means, stds=stds, n_frames=len(frames))

    def departure(self, frame: "CapsuleFrame") -> dict[str, float]:
        """Signed standardized deviation from nominal, per feature."""
        features = frame_features(frame)
        return {
            name: (features[name] - self.means[name]) / self.stds[name]
            for name in self.means
            if name in features
        }


@dataclass(frozen=True)
class DeliveryEvidence:
    """What a candidate seed row actually delivers, measured not assumed."""

    seed_row: int
    onset_row: int
    failure_mode: str
    feature: str
    seed_departure_z: float
    onset_departure_z: float
    expected_sign: int
    reference_frames: int
    distinguishable: bool
    directional: bool
    sustained: bool

    @property
    def delivers(self) -> bool:
        return self.distinguishable and self.directional and self.sustained

    def to_dict(self) -> dict:
        return {
            "seed_row": self.seed_row,
            "onset_row": self.onset_row,
            "failure_mode": self.failure_mode,
            "feature": self.feature,
            "seed_departure_z": self.seed_departure_z,
            "onset_departure_z": self.onset_departure_z,
            "expected_sign": self.expected_sign,
            "reference_frames": self.reference_frames,
            "distinguishable": self.distinguishable,
            "directional": self.directional,
            "sustained": self.sustained,
            "delivers": self.delivers,
        }


def _reference_frames(
    capsule: "FailureCapsule", reference_fraction: float
) -> Sequence["CapsuleFrame"]:
    """Earliest frames of the recording, used as that capsule's own nominal.

    Anchored at row 0 and measured as a fraction of the distance to onset, not
    as everything before ``pre_failure_start_index``. That index is itself
    derived from a rollback, and for a mode whose failure develops slowly the
    rollback lands inside the failure, so using it would build the nominal
    reference out of failure frames and hide the very departure being measured.
    """
    if not 0 < reference_fraction <= 1:
        raise ValueError("reference_fraction must lie in (0,1]")
    count = int(round(capsule.failure_onset_index * reference_fraction))
    if count < 2:
        raise ValueError(
            f"capsule records only {capsule.failure_onset_index} frames before onset, so "
            f"{reference_fraction:.0%} of them is under the two needed for a nominal "
            "reference; record more time before onset"
        )
    return capsule.frames[:count]


def measure_delivery(
    capsule: "FailureCapsule",
    seed_row: int,
    *,
    min_departure_z: float = 3.0,
    reference_fraction: float = 0.4,
) -> DeliveryEvidence:
    """Measure whether ``seed_row`` delivers this capsule's failure mode.

    Distinguishable means the seeded state differs from the capsule's own
    nominal window by at least ``min_departure_z`` standard deviations on the
    feature that carries the mode's signature. Directional means it differs in
    the direction the mode develops. Sustained means the departure holds on
    every frame from the seed through onset, which is what makes the seed a
    point on the trajectory toward the failure rather than a transient
    excursion that recovered.

    Sustained deliberately does not require the onset frame to depart further
    than the seed. Most of these failures plateau: a slipping robot reaches
    near-zero velocity and stays there, so its departure is flat after
    development and a monotone-growth requirement would reject a correct seed.
    """
    if type(seed_row) is not int:
        raise ValueError("seed_row must be an integer")
    if not 0 <= seed_row < len(capsule.frames):
        raise ValueError("seed_row is outside the trajectory")
    feature, expected_sign = _MODE_SIGNATURE.get(capsule.failure_mode, ("", 0))
    if not feature:
        raise ValueError(
            f"no signature feature registered for mode {capsule.failure_mode!r}; "
            "an unclassified capsule cannot be shown to deliver anything"
        )
    reference = NominalReference.from_frames(
        _reference_frames(capsule, reference_fraction)
    )
    if feature not in reference.means:
        raise ValueError(
            f"capsule does not record the channels needed for {feature}, so "
            f"delivery of {capsule.failure_mode!r} cannot be measured"
        )
    seed_z = reference.departure(capsule.frames[seed_row]).get(feature)
    onset_z = reference.departure(capsule.frames[capsule.failure_onset_index]).get(feature)
    if seed_z is None or onset_z is None:
        raise ValueError(f"{feature} is missing at the seed or onset frame")
    signed_seed = expected_sign * seed_z
    onset = capsule.failure_onset_index
    sustained = seed_row < onset and all(
        (z := reference.departure(capsule.frames[i]).get(feature)) is not None
        and expected_sign * z >= min_departure_z
        for i in range(seed_row, onset + 1)
    )
    return DeliveryEvidence(
        seed_row=seed_row,
        onset_row=onset,
        failure_mode=capsule.failure_mode,
        feature=feature,
        seed_departure_z=seed_z,
        onset_departure_z=onset_z,
        expected_sign=expected_sign,
        reference_frames=reference.n_frames,
        distinguishable=abs(seed_z) >= min_departure_z,
        directional=signed_seed >= min_departure_z,
        sustained=sustained,
    )


def resolve_developing_seed_row(
    capsule: "FailureCapsule",
    *,
    fraction: float = 0.5,
    min_departure_z: float = 3.0,
    reference_fraction: float = 0.4,
) -> int:
    """Seed row at a fraction of this capsule's own failure development.

    Development starts at the earliest pre-onset frame that has departed from
    nominal in the mode's direction and stays departed through onset, and ends
    at onset. Taking a fraction of that per-capsule window is what a fixed
    number of seconds cannot do: development time spans a factor of six across
    the six modes here, so one global offset cannot sit inside all of them.
    """
    if not 0 <= fraction < 1:
        raise ValueError("fraction must lie in [0,1)")
    onset = capsule.failure_onset_index
    feature, expected_sign = _MODE_SIGNATURE.get(capsule.failure_mode, ("", 0))
    if not feature:
        raise ValueError(
            f"no signature feature registered for mode {capsule.failure_mode!r}"
        )
    reference = NominalReference.from_frames(
        _reference_frames(capsule, reference_fraction)
    )
    if feature not in reference.means:
        raise ValueError(
            f"capsule does not record the channels needed for {feature}"
        )

    departed = []
    for i in range(capsule.pre_failure_start_index, onset + 1):
        z = reference.departure(capsule.frames[i]).get(feature)
        departed.append(z is not None and expected_sign * z >= min_departure_z)
    # Walk back from onset while the departure holds, so a transient early
    # excursion cannot be mistaken for the start of this failure.
    start = onset
    for offset, ok in enumerate(reversed(departed)):
        if not ok:
            break
        start = onset - offset
    if start >= onset:
        raise ValueError(
            f"no pre-onset frame of this capsule departs from its own nominal window "
            f"by {min_departure_z} sigma on {feature} in the direction of "
            f"{capsule.failure_mode!r}. The recording holds no deliverable "
            "pre-failure state: widen the stable prefix or lengthen development."
        )
    return start + int(round(fraction * (onset - start)))


def assert_delivers(
    capsule: "FailureCapsule",
    seed_row: int,
    *,
    min_departure_z: float = 3.0,
    reference_fraction: float = 0.4,
) -> DeliveryEvidence:
    """Raise unless the seed row provably delivers the capsule's failure mode.

    This is the gate Phase I lacked. Phase I reset half its environments to a
    lightly jittered stand and measured the result as though a failure
    curriculum had been applied, and the same would have been true of the v2
    path, whose pre-onset guard compared an offset against a window computed
    from the same offset.
    """
    assert_deliverable(capsule.failure_mode)
    frame = capsule.frames[seed_row]
    if frame.missing_reset_fields:
        raise ValueError(f"seed frame lacks reset state: {frame.missing_reset_fields}")
    implausible = implausible_reset_fields(frame)
    if implausible:
        raise ValueError(
            f"seed frame has physically impossible values in {', '.join(implausible)}; "
            "a base height of exactly zero is an unrecorded signal, not a measurement"
        )
    evidence = measure_delivery(
        capsule,
        seed_row,
        min_departure_z=min_departure_z,
        reference_fraction=reference_fraction,
    )
    if not evidence.distinguishable:
        raise ValueError(
            f"seed row {seed_row} is indistinguishable from this capsule's own nominal "
            f"window: {evidence.feature} departs by {evidence.seed_departure_z:+.2f} "
            f"sigma against a {min_departure_z} sigma requirement. This is the Phase-I "
            "defect: the treatment is not delivered and the run still looks valid."
        )
    if not evidence.directional:
        raise ValueError(
            f"seed row {seed_row} departs from nominal but not toward "
            f"{capsule.failure_mode!r}: {evidence.feature} moved "
            f"{evidence.seed_departure_z:+.2f} sigma where the mode develops "
            f"{'upward' if evidence.expected_sign > 0 else 'downward'}"
        )
    if not evidence.sustained:
        raise ValueError(
            f"seed row {seed_row} does not hold its departure through onset "
            f"(row {evidence.onset_row}) on {evidence.feature}: either the seed is at "
            "or past onset, or the departure lapses back to nominal in between, which "
            "makes it a transient excursion rather than this failure developing"
        )
    return evidence
