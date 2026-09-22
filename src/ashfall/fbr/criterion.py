"""The one predeclared failure criterion of the first FBR study, and onset extraction.

The criterion is written against channels a stock GO2 actually records, so the
same definition labels a simulator rollout and a hardware trial:

* **attitude**: combined roll/pitch tilt from the base quaternion (IMU on the
  robot) exceeds ``max_tilt_rad`` for ``debounce_frames`` consecutive frames;
* **terminal flag**: the rollout carries an explicit terminal event, a
  simulator termination (base contact) or, on hardware, an operator catch or
  emergency stop. A terminal flag counts on the frame it is raised;
* **stall** (optional): planar forward speed stays below
  ``stall_speed_fraction`` of the commanded speed for ``stall_seconds``. It
  needs a base velocity estimate, so it is off by default and a criterion that
  enables it refuses a trajectory without that channel.

Onset is the first frame of the first debounced run of the criterion, or the
first terminal flag, whichever is earlier. The criterion is a value object
with a content hash: a capsule records which criterion labelled it, and a
replay must be judged by the same one.

Foot slip itself is not observed on the stock GO2 (no calibrated per-foot
contact), so the study's endpoint is traverse failure on the slip-inducing
patch, and "slip" names the family of the physical condition, not a detected
phenotype.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from ashfall.provenance import content_hash

TERMINAL_FLAGS: tuple[str, ...] = ("terminated", "operator_stop")


def tilt_from_quat_xyzw(quat: Sequence[float]) -> float:
    """Angle between the body z axis and world z, radians (roll and pitch combined)."""
    x, y, z, w = (float(v) for v in quat)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("base_quat must be a finite nonzero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    # z component of the body z axis expressed in world: 1 - 2(x^2 + y^2).
    cos_tilt = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    return math.acos(cos_tilt)


@dataclass(frozen=True)
class FailureCriterion:
    name: str = "traverse_failure_v1"
    max_tilt_rad: float = 0.8
    debounce_frames: int = 3
    use_terminal_flags: bool = True
    stall_speed_fraction: float | None = None
    stall_seconds: float = 1.0

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("criterion needs a name")
        if not (math.isfinite(self.max_tilt_rad) and 0.0 < self.max_tilt_rad < math.pi):
            raise ValueError("max_tilt_rad must lie in (0, pi)")
        if type(self.debounce_frames) is not int or self.debounce_frames < 1:
            raise ValueError("debounce_frames must be a positive integer")
        if self.stall_speed_fraction is not None and not 0.0 < self.stall_speed_fraction < 1.0:
            raise ValueError("stall_speed_fraction must lie in (0, 1)")
        if not (math.isfinite(self.stall_seconds) and self.stall_seconds > 0):
            raise ValueError("stall_seconds must be positive")

    @property
    def criterion_id(self) -> str:
        return content_hash(asdict(self))

    def to_dict(self) -> dict:
        return {**asdict(self), "criterion_id": self.criterion_id}


@dataclass(frozen=True)
class Onset:
    index: int
    timestamp_s: float
    reason: str  # "attitude", "terminated", "operator_stop" or "stall"
    criterion_id: str


def _field(row: Any, name: str):
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name, None)


def frame_fails_attitude(row: Any, criterion: FailureCriterion) -> bool:
    quat = _field(row, "base_quat")
    if quat is None:
        raise ValueError("attitude criterion needs base_quat on every frame")
    return tilt_from_quat_xyzw(quat) > criterion.max_tilt_rad


def detect_onset(rows: Sequence[Any], criterion: FailureCriterion) -> Onset | None:
    """First failure onset under ``criterion``, or None when the rollout never fails.

    ``rows`` are mappings (Phoenix parquet rows, hardware log rows) or
    ``CapsuleFrame`` objects; each needs ``timestamp_s`` and ``base_quat``.
    """
    if not rows:
        raise ValueError("empty trajectory")
    times = [float(_field(r, "timestamp_s")) for r in rows]
    if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("timestamps must be finite and strictly increasing")
    candidates: list[tuple[int, str]] = []

    if criterion.use_terminal_flags:
        for i, row in enumerate(rows):
            raised = [flag for flag in TERMINAL_FLAGS if bool(_field(row, flag))]
            if raised:
                candidates.append((i, raised[0]))
                break

    run = 0
    for i, row in enumerate(rows):
        run = run + 1 if frame_fails_attitude(row, criterion) else 0
        if run == criterion.debounce_frames:
            candidates.append((i - criterion.debounce_frames + 1, "attitude"))
            break

    if criterion.stall_speed_fraction is not None:
        start = None
        for i, row in enumerate(rows):
            vel, cmd = _field(row, "base_lin_vel_body"), _field(row, "command_vel")
            if vel is None or cmd is None:
                raise ValueError("stall criterion needs base_lin_vel_body and command_vel")
            commanded = math.hypot(float(cmd[0]), float(cmd[1]))
            stalled = commanded > 0 and float(vel[0]) < criterion.stall_speed_fraction * commanded
            if not stalled:
                start = None
                continue
            start = i if start is None else start
            if times[i] - times[start] >= criterion.stall_seconds - 1e-9:
                candidates.append((start, "stall"))
                break

    if not candidates:
        return None
    index, reason = min(candidates)
    return Onset(index, times[index], reason, criterion.criterion_id)
