"""Versioned failure records. Unknown physical state stays unknown, never imputed.

Quaternion order is xyzw; logged root velocities are in the body frame. Indices
address ``frames`` (zero based), and the post-failure end index is inclusive.
A capsule is recorded evidence, not proof of simulator reproduction.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ashfall.delivery import implausible_reset_fields, resolve_developing_seed_row

SCHEMA_VERSION = "1.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _vector(value: Sequence[float] | None, size: int | None, name: str):
    if value is None:
        return None
    result = tuple(float(x) for x in value)
    if not result or (size is not None and len(result) != size):
        raise ValueError(f"{name}: invalid vector length")
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name}: non-finite value")
    return result


@dataclass(frozen=True)
class CapsuleFrame:
    timestamp_s: float
    base_pos: tuple[float, ...] | None = None
    base_quat: tuple[float, ...] | None = None
    base_lin_vel_body: tuple[float, ...] | None = None
    base_ang_vel_body: tuple[float, ...] | None = None
    joint_pos: tuple[float, ...] | None = None
    joint_vel: tuple[float, ...] | None = None
    command_vel: tuple[float, ...] | None = None
    action: tuple[float, ...] | None = None
    contact_forces: tuple[float, ...] | None = None
    joint_torque: tuple[float, ...] | None = None
    motor_current: tuple[float, ...] | None = None
    imu_acceleration: tuple[float, ...] | None = None
    imu_angular_velocity: tuple[float, ...] | None = None

    def __post_init__(self):
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and nonnegative")
        for name, size in {
            "base_pos": 3,
            "base_quat": 4,
            "base_lin_vel_body": 3,
            "base_ang_vel_body": 3,
            "command_vel": 3,
            "imu_acceleration": 3,
            "imu_angular_velocity": 3,
            "joint_pos": None,
            "joint_vel": None,
            "action": None,
            "contact_forces": None,
            "joint_torque": None,
            "motor_current": None,
        }.items():
            object.__setattr__(self, name, _vector(getattr(self, name), size, name))
        if self.base_quat is not None:
            norm = math.sqrt(sum(x * x for x in self.base_quat))
            if not math.isclose(norm, 1.0, abs_tol=1e-4):
                raise ValueError("base_quat must be a unit xyzw quaternion")
        sizes = {
            len(getattr(self, name))
            for name in ("joint_pos", "joint_vel", "joint_torque", "motor_current")
            if getattr(self, name) is not None
        }
        if len(sizes) > 1:
            raise ValueError("joint telemetry dimensions disagree")

    @property
    def missing_reset_fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "base_pos",
                "base_quat",
                "base_lin_vel_body",
                "base_ang_vel_body",
                "joint_pos",
                "joint_vel",
                "command_vel",
            )
            if getattr(self, name) is None
        )


@dataclass(frozen=True)
class FailureCapsule:
    source: str
    robot: str
    policy_id: str | None
    timestamp: str | None
    control_dt: float
    failure_mode: str
    failure_onset_index: int
    pre_failure_start_index: int
    post_failure_end_index: int
    frames: tuple[CapsuleFrame, ...]
    detector_version: str | None = None
    threshold_config_version: str | None = None
    checkpoint_sha256: str | None = None
    trajectory_sha256: str | None = None
    event_descriptor: Mapping[str, Any] = field(default_factory=dict)
    environment_metadata: Mapping[str, Any] | None = None
    surface_terrain_metadata: Mapping[str, Any] | None = None
    disturbances: Mapping[str, Any] | None = None
    payload: Mapping[str, Any] | None = None
    actuator_settings: Mapping[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION
    capsule_id: str = ""

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported capsule schema: {self.schema_version}")
        for name in ("source", "robot", "failure_mode"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not math.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt must be positive and finite")
        if self.timestamp is not None:
            parsed = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("timestamp requires timezone")
        object.__setattr__(self, "frames", tuple(self.frames))
        if not self.frames or not all(isinstance(f, CapsuleFrame) for f in self.frames):
            raise ValueError("frames must contain CapsuleFrame records")
        if any(
            type(i) is not int
            for i in (
                self.pre_failure_start_index,
                self.failure_onset_index,
                self.post_failure_end_index,
            )
        ):
            raise ValueError("window indices must be integers")
        if not (
            0
            <= self.pre_failure_start_index
            <= self.failure_onset_index
            <= self.post_failure_end_index
            < len(self.frames)
        ):
            raise ValueError("invalid pre/onset/post window")
        for a, b in zip(self.frames, self.frames[1:]):
            if b.timestamp_s <= a.timestamp_s:
                raise ValueError("frame timestamps must strictly increase")
        for name in ("checkpoint_sha256", "trajectory_sha256"):
            value = getattr(self, name)
            if value is not None and (
                len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA256")
        for name in (
            "event_descriptor",
            "environment_metadata",
            "surface_terrain_metadata",
            "disturbances",
            "payload",
            "actuator_settings",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Mapping):
                raise ValueError(f"{name} must be an object or null")
        expected = self.content_id()
        if self.capsule_id and self.capsule_id != expected:
            raise ValueError("capsule_id does not match capsule content")
        object.__setattr__(self, "capsule_id", expected)

    def content_id(self) -> str:
        data = asdict(self)
        data.pop("capsule_id")
        return "cap_" + hashlib.sha256(_canonical(data).encode()).hexdigest()

    @property
    def pre_failure_frames(self) -> tuple[CapsuleFrame, ...]:
        """Window before onset; excludes the failure-onset frame."""
        return self.frames[self.pre_failure_start_index : self.failure_onset_index]

    def resolve_seed_index(
        self,
        strategy: str = "failure_onset_minus_seconds",
        *,
        offset_steps: int = 0,
        offset_seconds: float = 0.5,
        fraction: float = 0.5,
        min_departure_z: float = 3.0,
    ) -> int:
        """Resolve the row a reset should seed from.

        ``failure_onset_minus_fraction`` is the strategy to prefer. The
        seconds and steps strategies apply one global offset to every
        trajectory, and failure development time varies by a factor of six
        across this repo's modes, so a single offset lands inside the nominal
        prefix for most of them. That is the Phase-I delivery defect, and
        ``ashfall.delivery`` measures it rather than assuming it away.
        """
        onset = self.failure_onset_index
        if strategy == "first":
            return 0  # Explicit historical behavior only.
        if strategy == "failure_onset":
            return onset
        if strategy == "failure_onset_minus_fraction":
            return resolve_developing_seed_row(
                self, fraction=fraction, min_departure_z=min_departure_z
            )
        if strategy == "failure_onset_minus_steps":
            if type(offset_steps) is not int or offset_steps < 0:
                raise ValueError("offset_steps must be a nonnegative integer")
            index = onset - offset_steps
        elif strategy == "failure_onset_minus_seconds":
            if not math.isfinite(offset_seconds) or offset_seconds < 0:
                raise ValueError("offset_seconds must be finite and nonnegative")
            target = self.frames[onset].timestamp_s - offset_seconds
            # Last frame at or before requested time; timestamps handle dropped rows.
            eligible = [i for i in range(onset + 1) if self.frames[i].timestamp_s <= target + 1e-10]
            index = eligible[-1] if eligible else -1
        else:
            raise ValueError(f"unknown reset strategy: {strategy}")
        if index < self.pre_failure_start_index:
            raise ValueError("requested pre-onset seed is outside reviewed capsule window")
        return index

    def reset_frame(self, **kwargs) -> CapsuleFrame:
        """The frame a reset would seed from, or a refusal.

        Presence is not plausibility. The GO2 capture path writes hardcoded
        zeros where a signal was never measured, so a present-and-zero channel
        used to pass the null check and become a reset state: a base at the
        terrain origin at zero height, holding a zero command.
        """
        frame = self.frames[self.resolve_seed_index(**kwargs)]
        if frame.missing_reset_fields:
            raise ValueError(f"cannot reset from missing state: {frame.missing_reset_fields}")
        implausible = implausible_reset_fields(frame)
        if implausible:
            raise ValueError(
                f"cannot reset from physically impossible state: {implausible}; a base "
                "height of exactly zero is an unrecorded signal, not a measurement"
            )
        return frame

    def to_dict(self) -> dict:
        if self.content_id() != self.capsule_id:
            raise ValueError("capsule metadata changed after identity was assigned")
        return json.loads(_canonical(asdict(self)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FailureCapsule:
        values = dict(data)
        if "schema_version" not in values or not values.get("capsule_id"):
            raise ValueError("serialized capsule requires schema_version and capsule_id")
        values["frames"] = tuple(CapsuleFrame(**f) for f in values["frames"])
        return cls(**values)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Revalidate identity in case caller mutated a metadata mapping.
        if self.content_id() != self.capsule_id:
            raise ValueError("capsule metadata changed after identity was assigned")
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> FailureCapsule:
        return cls.from_dict(json.loads(Path(path).read_text()))


def capsules_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
    robot: str,
    policy_id: str | None = None,
    timestamp: str | None = None,
    control_dt: float | None = None,
    pre_failure_seconds: float = 0.5,
    post_failure_seconds: float = 1.0,
    onset_indices: Sequence[int] | None = None,
    **metadata,
) -> list[FailureCapsule]:
    """Migrate Phoenix rows, one capsule per labeled episode/mode transition.

    Existing flags are legacy labels, not independent detector ground truth.
    Explicit onset indices allow hand review to override those labels. Original
    row numbering and all frames are retained. Missing physical fields stay null.
    """
    if not rows:
        raise ValueError("trajectory is empty")
    if any(not math.isfinite(x) or x < 0 for x in (pre_failure_seconds, post_failure_seconds)):
        raise ValueError("window durations must be finite and nonnegative")
    names = CapsuleFrame.__dataclass_fields__
    frames = tuple(CapsuleFrame(**{k: row[k] for k in names if k in row}) for row in rows)
    if control_dt is None:
        if len(frames) < 2:
            raise ValueError("control_dt required for one-frame trajectory")
        import statistics

        control_dt = statistics.median(
            b.timestamp_s - a.timestamp_s for a, b in zip(frames, frames[1:])
        )
    if onset_indices is None:
        onsets = []
        previous = None
        for i, row in enumerate(rows):
            mode = row.get("failure_mode") if row.get("failure_flag", False) else None
            if mode and mode != previous:
                onsets.append(i)
            previous = mode
    else:
        onsets = list(onset_indices)
    if len(onsets) != len(set(onsets)):
        raise ValueError("duplicate onset indices")
    result = []
    for onset in onsets:
        if type(onset) is not int or not 0 <= onset < len(rows):
            raise ValueError("onset index is outside trajectory")
        onset_time = frames[onset].timestamp_s
        before = [
            i
            for i in range(onset + 1)
            if frames[i].timestamp_s <= onset_time - pre_failure_seconds + 1e-10
        ]
        start = before[-1] if before else 0
        after = [
            i
            for i in range(onset, len(frames))
            if frames[i].timestamp_s <= onset_time + post_failure_seconds + 1e-10
        ]
        mode = rows[onset].get("failure_mode") or "unclassified"
        descriptor = {
            "onset_label_source": "explicit_review"
            if onset_indices is not None
            else "legacy_failure_flag"
        }
        descriptor.update(metadata.get("event_descriptor", {}))
        extra = {k: v for k, v in metadata.items() if k != "event_descriptor"}
        result.append(
            FailureCapsule(
                source=source,
                robot=robot,
                policy_id=policy_id,
                timestamp=timestamp,
                control_dt=control_dt,
                failure_mode=mode,
                failure_onset_index=onset,
                pre_failure_start_index=start,
                post_failure_end_index=after[-1],
                frames=frames,
                event_descriptor=descriptor,
                **extra,
            )
        )
    return result


def capsules_from_parquet(path: str | Path, **kwargs) -> list[FailureCapsule]:
    import pyarrow.parquet as pq

    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return capsules_from_rows(pq.read_table(path).to_pylist(), trajectory_sha256=digest, **kwargs)
