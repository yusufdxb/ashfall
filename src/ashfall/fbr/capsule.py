"""Failure capsules for FBR: extraction from a rollout and the seed states it offers.

The capsule object is :class:`ashfall.capsule.FailureCapsule`; this module does
not define a second one. What FBR adds is a narrow contract on top of it:

1. the capsule was labelled by a named :class:`FailureCriterion` (its id is in
   ``event_descriptor``) and names the policy checkpoint that failed;
2. seed states are taken strictly before onset, at predeclared lead times, and
   each one must be (a) inside the reviewed pre-failure window, (b) within
   ``max_lead_s`` of onset, (c) resettable (no missing or impossible channel),
   and (d) not already failing under the criterion.

Rule (b) is the regression guard for the Phase-I defect. That curriculum
seeded row 0 of each trajectory, a nominal gait state 1.4 s before the failure
of the synthetic slip file, and nothing checked the distance to onset. Here a
seed that is far from onset is refused, not clamped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ashfall.capsule import CapsuleFrame, FailureCapsule, capsules_from_rows
from ashfall.delivery import implausible_reset_fields
from ashfall.fbr.criterion import FailureCriterion, detect_onset, frame_fails_attitude

#: Lead times before onset at which FBR seeds replay, seconds. Predeclared.
DEFAULT_LEAD_TIMES_S: tuple[float, ...] = (0.2, 0.4, 0.6)
#: A seed further than this from onset is not "near the failure boundary".
DEFAULT_MAX_LEAD_S: float = 1.0


@dataclass(frozen=True)
class SeedPoint:
    capsule_id: str
    row: int
    lead_s: float
    frame: CapsuleFrame

    @property
    def seed_id(self) -> str:
        return f"{self.capsule_id}:{self.row}"


def extract_capsule(
    rows: Sequence[Mapping[str, Any]],
    *,
    criterion: FailureCriterion,
    source: str,
    robot: str,
    policy_sha256: str,
    history_s: float = 1.0,
    post_s: float = 0.5,
    control_dt: float | None = None,
    environment_metadata: Mapping[str, Any] | None = None,
    surface_terrain_metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> FailureCapsule:
    """One capsule around the first criterion onset, or a refusal when none occurs."""
    if not isinstance(policy_sha256, str) or len(policy_sha256) != 64:
        raise ValueError("FBR capsules must name the failing policy by SHA256")
    onset = detect_onset(rows, criterion)
    if onset is None:
        raise ValueError(f"no failure under criterion {criterion.name!r}; nothing to capture")
    extra: dict[str, Any] = {"schema_version": "1.0"}
    if provenance is not None:
        extra = {"schema_version": "1.1", "provenance": dict(provenance)}
    (capsule,) = capsules_from_rows(
        rows,
        source=source,
        robot=robot,
        policy_id=policy_sha256,
        control_dt=control_dt,
        pre_failure_seconds=history_s,
        post_failure_seconds=post_s,
        onset_indices=[onset.index],
        checkpoint_sha256=policy_sha256,
        event_descriptor={
            "criterion": criterion.to_dict(),
            "criterion_id": criterion.criterion_id,
            "onset_reason": onset.reason,
            "onset_timestamp_s": onset.timestamp_s,
        },
        environment_metadata=None if environment_metadata is None else dict(environment_metadata),
        surface_terrain_metadata=None
        if surface_terrain_metadata is None
        else dict(surface_terrain_metadata),
        **extra,
    )
    return capsule


def capsule_criterion(capsule: FailureCapsule) -> FailureCriterion:
    """The criterion that labelled this capsule; refuses a capsule FBR did not label."""
    data = dict(capsule.event_descriptor.get("criterion") or {})
    if not data:
        raise ValueError("capsule carries no FBR failure criterion")
    recorded = data.pop("criterion_id", None)
    criterion = FailureCriterion(**data)
    if recorded != criterion.criterion_id or recorded != capsule.event_descriptor.get(
        "criterion_id"
    ):
        raise ValueError("capsule criterion id does not match its recorded definition")
    return criterion


def assert_near_onset(capsule: FailureCapsule, row: int, *, max_lead_s: float) -> float:
    """Lead time of ``row`` before onset, or a refusal. Row 0 gets no exemption."""
    if type(row) is not int:
        raise ValueError("seed row must be an integer")
    onset = capsule.failure_onset_index
    if not capsule.pre_failure_start_index <= row < onset:
        raise ValueError(
            f"seed row {row} is outside the pre-onset window "
            f"[{capsule.pre_failure_start_index}, {onset})"
        )
    lead = capsule.frames[onset].timestamp_s - capsule.frames[row].timestamp_s
    if lead > max_lead_s + 1e-9:
        raise ValueError(
            f"seed row {row} is {lead:.3f} s before onset, beyond the {max_lead_s} s "
            "boundary window: a state this early is a nominal state, not a failure boundary"
        )
    return lead


def seed_points(
    capsule: FailureCapsule,
    *,
    lead_times_s: Sequence[float] = DEFAULT_LEAD_TIMES_S,
    max_lead_s: float = DEFAULT_MAX_LEAD_S,
) -> tuple[SeedPoint, ...]:
    """The resettable pre-onset states FBR replays from, one per lead time."""
    criterion = capsule_criterion(capsule)
    if capsule.checkpoint_sha256 is None:
        raise ValueError("FBR capsule must name the failing policy checkpoint")
    if not lead_times_s or any(not math.isfinite(t) or t <= 0 for t in lead_times_s):
        raise ValueError("lead times must be positive and finite")
    onset_time = capsule.frames[capsule.failure_onset_index].timestamp_s
    result = []
    for lead in sorted(set(float(t) for t in lead_times_s)):
        target = onset_time - lead
        eligible = [
            i
            for i in range(capsule.failure_onset_index)
            if capsule.frames[i].timestamp_s <= target + 1e-9
        ]
        if not eligible:
            raise ValueError(f"no frame {lead} s before onset")
        row = eligible[-1]
        actual = assert_near_onset(capsule, row, max_lead_s=max_lead_s)
        frame = capsule.frames[row]
        if frame.missing_reset_fields:
            raise ValueError(f"seed row {row} lacks reset state {frame.missing_reset_fields}")
        implausible = implausible_reset_fields(frame)
        if implausible:
            raise ValueError(f"seed row {row} has physically impossible state {implausible}")
        if frame_fails_attitude(frame, criterion):
            raise ValueError(
                f"seed row {row} already meets the failure criterion; seeding at the failure "
                "makes reproducing it tautological"
            )
        result.append(SeedPoint(capsule.capsule_id, row, actual, frame))
    rows = [p.row for p in result]
    if len(set(rows)) != len(rows):
        raise ValueError("two lead times resolved to the same frame; the log is too sparse")
    return tuple(result)


def seed_before_position(
    capsule: FailureCapsule, x_threshold: float, *, max_lead_s: float
) -> SeedPoint:
    """The last pre-onset frame whose base x is still short of ``x_threshold``.

    For a patch failure this is the state at patch entry: the approach the robot
    was in when it met the surface, rather than a state already slipping on it.
    The patch position comes from the lab layout (tape marks, video), never from
    the policy's own estimate. The same refusals as :func:`seed_points` apply.
    """
    criterion = capsule_criterion(capsule)
    if not math.isfinite(x_threshold):
        raise ValueError("x_threshold must be finite")
    rows = []
    for i in range(capsule.failure_onset_index):
        pos = capsule.frames[i].base_pos
        if pos is not None and pos[0] < x_threshold:
            rows.append(i)
    if not rows:
        raise ValueError("no pre-onset frame lies before the position")
    row = rows[-1]
    lead = assert_near_onset(capsule, row, max_lead_s=max_lead_s)
    frame = capsule.frames[row]
    if frame.missing_reset_fields:
        raise ValueError(f"seed row {row} lacks reset state {frame.missing_reset_fields}")
    implausible = implausible_reset_fields(frame)
    if implausible:
        raise ValueError(f"seed row {row} has physically impossible state {implausible}")
    if frame_fails_attitude(frame, criterion):
        raise ValueError(f"seed row {row} already meets the failure criterion")
    return SeedPoint(capsule.capsule_id, row, lead, frame)
