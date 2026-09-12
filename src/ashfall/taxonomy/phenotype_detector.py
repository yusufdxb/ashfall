"""Phenotype-emitting wrapper over the threshold detector, with mutation hooks.

:class:`ashfall.taxonomy.detector.FailureDetector` emits ``FailureEvent``
records named after the threshold that fired. This wrapper turns them into
:class:`ashfall.ontology.PhenotypeObservation` records with an
:class:`ashfall.ontology.OnsetWindow`, and it is honest about what a threshold
detector knows: the frame at which its criterion was first met. So
``transition_start == established_start``, no precursor is claimed, and the
label source is ``"detector"``.

Two things live here that the base detector deliberately does not carry.

* **Support.** A phenotype is scored on an episode only when the channels its
  signature reads are present and the platform can observe it
  (:data:`ashfall.ontology.PHENOTYPES`). A missing channel makes the phenotype
  UNSUPPORTED for that episode, never silently negative.
* **Mutations.** :class:`DetectorMutation` builds deliberately broken
  detectors (always-collapse, always-stumble, always-command-mismatch,
  slip-priority removed, two labels swapped) without editing ``detector.py``.
  The detector evaluation in :mod:`ashfall.detector_eval` must catch every one
  of them before any accuracy figure it produces may be quoted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ashfall.ontology import (
    PHENOTYPE_NAMES,
    PHENOTYPES,
    SUPPORTED,
    OnsetWindow,
    PhenotypeObservation,
)
from ashfall.taxonomy.detector import FailureDetector, FailureThresholds

#: Telemetry channel names an episode may carry, per step.
TELEMETRY_CHANNELS: tuple[str, ...] = (
    "pitch_rad",
    "roll_rad",
    "base_height_m",
    "cmd_lin_vel",
    "actual_lin_vel",
    "joint_vel",
    "contact_forces",
)

#: The base detector cannot step without these.
REQUIRED_TELEMETRY_CHANNELS: tuple[str, ...] = (
    "pitch_rad",
    "roll_rad",
    "base_height_m",
    "cmd_lin_vel",
    "actual_lin_vel",
)

#: Which telemetry channels realise each ontology signature channel.
SIGNATURE_TO_TELEMETRY: Mapping[str, tuple[str, ...]] = {
    "base_quat": ("pitch_rad", "roll_rad"),
    "base_pos": ("base_height_m",),
    "base_lin_vel_body": ("actual_lin_vel",),
    "command_vel": ("cmd_lin_vel",),
    "joint_vel": ("joint_vel",),
    "contact_forces": ("contact_forces",),
}

#: Platforms an evaluation may declare. ``synthetic_fixture`` is decided by
#: channel presence alone; the other two also consult the ontology's
#: observability, which records why a phenotype cannot be observed there.
PLATFORMS: tuple[str, ...] = ("synthetic_fixture", "simulation", "go2_hardware")

MUTATION_KINDS: tuple[str, ...] = (
    "none",
    "always_collapse",
    "always_stumble",
    "always_command_mismatch",
    "no_slip_priority",
    "label_swap",
)


@dataclass(frozen=True)
class DetectorMutation:
    """A deliberately introduced detector defect, named so a report can quote it."""

    kind: str = "none"
    swap: tuple[str, str] | None = None

    def __post_init__(self):
        if self.kind not in MUTATION_KINDS:
            raise ValueError(f"unknown mutation {self.kind!r}; expected {MUTATION_KINDS}")
        if self.kind == "label_swap":
            if self.swap is None or len(self.swap) != 2 or self.swap[0] == self.swap[1]:
                raise ValueError("label_swap needs two distinct phenotype names")
            for name in self.swap:
                if name not in PHENOTYPE_NAMES:
                    raise ValueError(f"label_swap: unknown phenotype {name!r}")
            object.__setattr__(self, "swap", (str(self.swap[0]), str(self.swap[1])))
        elif self.swap is not None:
            raise ValueError("swap applies to label_swap only")

    @property
    def is_identity(self) -> bool:
        return self.kind == "none"

    @property
    def affected_phenotypes(self) -> tuple[str, ...]:
        """Phenotypes whose scores this defect is expected to move."""
        if self.kind == "none":
            return ()
        if self.kind == "label_swap":
            return tuple(self.swap)  # type: ignore[arg-type]
        if self.kind == "no_slip_priority":
            return ("command_mismatch",)
        return (self.kind.removeprefix("always_"),)

    @property
    def label(self) -> str:
        return self.kind if self.swap is None else f"label_swap({self.swap[0]},{self.swap[1]})"


def supported_phenotypes(channels: Sequence[str], platform: str) -> dict[str, str | None]:
    """Per phenotype: ``None`` when scorable on these channels and platform, else the reason."""
    if platform not in PLATFORMS:
        raise ValueError(f"unknown platform {platform!r}; expected {PLATFORMS}")
    present = set(channels)
    result: dict[str, str | None] = {}
    for name, spec in PHENOTYPES.items():
        missing = sorted(
            telemetry
            for signature in spec.signature_channels
            for telemetry in SIGNATURE_TO_TELEMETRY[signature]
            if telemetry not in present
        )
        if missing:
            result[name] = f"telemetry lacks {', '.join(missing)}"
            continue
        if platform != "synthetic_fixture" and spec.observability[platform] != SUPPORTED:
            result[name] = f"{platform}: {spec.observability[platform]}"
            continue
        result[name] = None
    return result


@dataclass(frozen=True)
class DetectedEvent:
    phenotype: str
    frame: int
    detail: Mapping[str, Any] = field(default_factory=dict)


class PhenotypeDetector:
    """Run the threshold detector over one episode and emit windowed phenotype observations.

    One observation is emitted per detector event, not one per phenotype, so a
    failure that recurs after a genuine recovery is reported twice, which is
    what a recovery-episode evaluation needs.
    """

    detector_id = "ashfall.taxonomy.detector.FailureDetector"

    def __init__(
        self,
        thresholds: FailureThresholds | None = None,
        mutation: DetectorMutation | None = None,
    ) -> None:
        self.thresholds = thresholds or FailureThresholds()
        self.mutation = mutation or DetectorMutation()

    @property
    def label(self) -> str:
        return self.detector_id + ("" if self.mutation.is_identity else f"+{self.mutation.label}")

    # -- support ---------------------------------------------------------------

    @staticmethod
    def supported(channels: Sequence[str], platform: str) -> dict[str, str | None]:
        return supported_phenotypes(channels, platform)

    # -- detection -------------------------------------------------------------

    def events(self, telemetry: Mapping[str, Any], dt_s: float) -> list[DetectedEvent]:
        """Raw per-event output after mutation. ``telemetry`` maps channel -> array."""
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be positive and finite")
        missing = [name for name in REQUIRED_TELEMETRY_CHANNELS if telemetry.get(name) is None]
        if missing:
            raise ValueError(f"telemetry lacks required channels {missing}")
        pitch = np.asarray(telemetry["pitch_rad"], dtype=float).ravel()
        roll = np.asarray(telemetry["roll_rad"], dtype=float).ravel()
        height = np.asarray(telemetry["base_height_m"], dtype=float).ravel()
        cmd = np.atleast_2d(np.asarray(telemetry["cmd_lin_vel"], dtype=float))
        act = np.atleast_2d(np.asarray(telemetry["actual_lin_vel"], dtype=float))
        joint_vel = telemetry.get("joint_vel")
        contacts = telemetry.get("contact_forces")
        joint_vel = None if joint_vel is None else np.atleast_2d(np.asarray(joint_vel, dtype=float))
        contacts = None if contacts is None else np.atleast_2d(np.asarray(contacts, dtype=float))
        n = len(pitch)
        if not (len(roll) == len(height) == cmd.shape[0] == act.shape[0] == n) or n == 0:
            raise ValueError("telemetry channels must be nonempty and equal in length")
        detector = FailureDetector(self.thresholds)
        events: list[DetectedEvent] = []
        mismatch = _UnprioritisedMismatch(self.thresholds)
        for i in range(n):
            timestamp = i * dt_s
            emitted = detector.step(
                timestamp_s=timestamp,
                pitch_rad=float(pitch[i]),
                roll_rad=float(roll[i]),
                base_height_m=float(height[i]),
                cmd_lin_vel=cmd[i],
                actual_lin_vel=act[i],
                joint_vel=None if joint_vel is None else joint_vel[i],
                contact_forces=None if contacts is None else contacts[i],
            )
            for event in emitted:
                if (
                    self.mutation.kind == "no_slip_priority"
                    and event.mode.value == "command_mismatch"
                ):
                    continue  # replaced by the unprioritised tracker below
                events.append(DetectedEvent(event.mode.value, i, dict(event.detail)))
            if self.mutation.kind == "no_slip_priority":
                fired = mismatch.step(timestamp, cmd[i], act[i])
                if fired is not None:
                    events.append(DetectedEvent("command_mismatch", i, fired))
        return self._mutate(events)

    def _mutate(self, events: list[DetectedEvent]) -> list[DetectedEvent]:
        kind = self.mutation.kind
        if kind in ("always_collapse", "always_stumble", "always_command_mismatch"):
            forced = kind.removeprefix("always_")
            events = [DetectedEvent(forced, 0, {"mutation": kind})] + events
        elif kind == "label_swap":
            a, b = self.mutation.swap  # type: ignore[misc]
            swapped = {a: b, b: a}
            events = [
                DetectedEvent(swapped.get(e.phenotype, e.phenotype), e.frame, e.detail)
                for e in events
            ]
        return sorted(events, key=lambda e: (e.frame, e.phenotype))

    def run(self, telemetry: Mapping[str, Any], dt_s: float) -> tuple[PhenotypeObservation, ...]:
        """Windowed observations. A threshold detector claims no precursor and no development."""
        return tuple(
            PhenotypeObservation(
                event.phenotype,
                OnsetWindow(event.frame, event.frame, None, None, "detector"),
                {"detector_event_frame": event.frame, "detail": _jsonable(event.detail)},
            )
            for event in self.events(telemetry, dt_s)
        )


class _UnprioritisedMismatch:
    """The base detector's command-mismatch rule with the slip exclusion removed.

    Exists only to build the ``no_slip_priority`` mutant. It is the same
    sustained-error rule, the same debounce, and the same thresholds; the one
    difference is that it does not stand down while the slip rule is active.
    """

    def __init__(self, thresholds: FailureThresholds) -> None:
        self.t = thresholds
        self.start: float | None = None
        self.last_event = -math.inf

    def step(self, timestamp: float, cmd, actual) -> dict | None:
        cmd_2d = np.asarray(cmd, dtype=float).ravel()[:2]
        act_2d = np.asarray(actual, dtype=float).ravel()[:2]
        cmd_speed = float(np.linalg.norm(cmd_2d))
        error = float(np.linalg.norm(cmd_2d - act_2d))
        if cmd_speed > self.t.cmd_mismatch_min_cmd_speed and error > self.t.cmd_mismatch_vel_error:
            if self.start is None:
                self.start = timestamp
            elif timestamp - self.start >= self.t.cmd_mismatch_min_duration_s:
                self.start = None
                if timestamp - self.last_event >= self.t.min_event_gap_s:
                    self.last_event = timestamp
                    return {
                        "cmd_speed": cmd_speed,
                        "vel_error": error,
                        "mutation": "no_slip_priority",
                    }
        else:
            self.start = None
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


__all__ = [
    "MUTATION_KINDS",
    "PLATFORMS",
    "REQUIRED_TELEMETRY_CHANNELS",
    "SIGNATURE_TO_TELEMETRY",
    "TELEMETRY_CHANNELS",
    "DetectedEvent",
    "DetectorMutation",
    "PhenotypeDetector",
    "supported_phenotypes",
]
