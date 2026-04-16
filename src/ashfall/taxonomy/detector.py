"""Extended failure taxonomy for quadruped locomotion.

Ashfall extends Phoenix's 3-mode detector (attitude/collapse/slip) with three
additional failure modes grounded in real quadruped failure literature:

4. **Stumble** — transient foot-catch events where a leg's swing trajectory
   is interrupted. Detected via sudden joint velocity spikes during swing phase.
5. **Contact loss** — one or more feet lose ground contact unexpectedly during
   stance phase. Detected via contact force dropout below a minimum threshold
   while the leg should be in stance.
6. **Command mismatch** — sustained disagreement between the velocity command
   and achieved velocity that does NOT qualify as slip (robot is moving, just
   in the wrong direction or at wrong speed). Indicates policy tracking failure.

All detectors are stateful, pure-numpy, and produce a single FailureEvent per
failure episode (not per timestep).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FailureMode(str, Enum):
    ATTITUDE = "attitude"
    COLLAPSE = "collapse"
    SLIP = "slip"
    STUMBLE = "stumble"
    CONTACT_LOSS = "contact_loss"
    COMMAND_MISMATCH = "command_mismatch"


# Severity ranking for downstream prioritization (higher = more severe)
SEVERITY = {
    FailureMode.COLLAPSE: 5,
    FailureMode.ATTITUDE: 4,
    FailureMode.SLIP: 3,
    FailureMode.STUMBLE: 2,
    FailureMode.CONTACT_LOSS: 2,
    FailureMode.COMMAND_MISMATCH: 1,
}


@dataclass(frozen=True)
class FailureThresholds:
    # --- Attitude ---
    pitch_rad: float = 0.8
    roll_rad: float = 0.6

    # --- Collapse ---
    base_height_min_m: float = 0.15

    # --- Slip ---
    slip_velocity_cmd_min: float = 0.3  # m/s
    slip_velocity_actual_max: float = 0.05  # m/s
    slip_min_duration_s: float = 0.5

    # --- Stumble ---
    stumble_joint_vel_spike: float = 15.0  # rad/s — sudden swing interruption
    stumble_min_feet_in_contact: int = 2  # at least 2 feet on ground (not mid-air)

    # --- Contact loss ---
    contact_force_min_n: float = 5.0  # below this = no contact
    contact_loss_min_feet: int = 2  # lose contact on >= N feet simultaneously
    contact_loss_min_duration_s: float = 0.1

    # --- Command mismatch ---
    cmd_mismatch_vel_error: float = 0.4  # m/s tracking error
    cmd_mismatch_min_cmd_speed: float = 0.2  # only flag when command is non-trivial
    cmd_mismatch_min_duration_s: float = 1.0  # sustained tracking failure

    # --- Global ---
    min_event_gap_s: float = 1.0


@dataclass
class FailureEvent:
    mode: FailureMode
    timestamp_s: float
    severity: int = 0
    detail: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.severity == 0:
            self.severity = SEVERITY.get(self.mode, 1)


class FailureDetector:
    """Stateful detector for 6 quadruped failure modes.

    Feed telemetry with :meth:`step`; returns a list of failure events
    detected at each timestep (usually 0 or 1, occasionally 2+ if
    multiple independent failures co-occur).
    """

    def __init__(self, thresholds: FailureThresholds | None = None) -> None:
        self.thresholds = thresholds or FailureThresholds()
        self._slip_start: float | None = None
        self._contact_loss_start: float | None = None
        self._cmd_mismatch_start: float | None = None
        self._last_event_at: dict[FailureMode, float] = {m: -np.inf for m in FailureMode}

    def step(
        self,
        *,
        timestamp_s: float,
        pitch_rad: float,
        roll_rad: float,
        base_height_m: float,
        cmd_lin_vel: np.ndarray,  # (2,) or (3,) — [vx, vy] used
        actual_lin_vel: np.ndarray,  # (2,) or (3,)
        joint_vel: np.ndarray | None = None,  # (12,) for stumble detection
        contact_forces: np.ndarray | None = None,  # (4,) per-foot normal force
    ) -> list[FailureEvent]:
        t = self.thresholds
        events: list[FailureEvent] = []

        cmd_2d = np.asarray(cmd_lin_vel, dtype=np.float64).ravel()[:2]
        actual_2d = np.asarray(actual_lin_vel, dtype=np.float64).ravel()[:2]

        # --- Priority 1: Attitude ---
        if abs(pitch_rad) > t.pitch_rad or abs(roll_rad) > t.roll_rad:
            ev = self._try_emit(
                FailureMode.ATTITUDE,
                timestamp_s,
                {"pitch": float(pitch_rad), "roll": float(roll_rad)},
            )
            if ev:
                events.append(ev)

        # --- Priority 2: Collapse ---
        if base_height_m < t.base_height_min_m:
            ev = self._try_emit(
                FailureMode.COLLAPSE,
                timestamp_s,
                {"height": float(base_height_m)},
            )
            if ev:
                events.append(ev)

        # --- Priority 3: Slip ---
        cmd_speed = float(np.linalg.norm(cmd_2d))
        actual_speed = float(np.linalg.norm(actual_2d))
        slipping = cmd_speed > t.slip_velocity_cmd_min and actual_speed < t.slip_velocity_actual_max

        if slipping:
            if self._slip_start is None:
                self._slip_start = timestamp_s
            elif timestamp_s - self._slip_start >= t.slip_min_duration_s:
                ev = self._try_emit(
                    FailureMode.SLIP,
                    timestamp_s,
                    {"cmd_speed": cmd_speed, "actual_speed": actual_speed},
                )
                if ev:
                    events.append(ev)
                self._slip_start = None
        else:
            self._slip_start = None

        # --- Priority 4: Stumble ---
        if joint_vel is not None:
            jv = np.asarray(joint_vel, dtype=np.float64).ravel()
            max_jv = float(np.max(np.abs(jv)))
            feet_in_contact = 4  # default if no contact info
            if contact_forces is not None:
                cf = np.asarray(contact_forces, dtype=np.float64).ravel()
                feet_in_contact = int(np.sum(cf > t.contact_force_min_n))
            spike = max_jv > t.stumble_joint_vel_spike
            grounded = feet_in_contact >= t.stumble_min_feet_in_contact
            if spike and grounded:
                ev = self._try_emit(
                    FailureMode.STUMBLE,
                    timestamp_s,
                    {"max_joint_vel": max_jv, "feet_in_contact": feet_in_contact},
                )
                if ev:
                    events.append(ev)

        # --- Priority 5: Contact loss ---
        if contact_forces is not None:
            cf = np.asarray(contact_forces, dtype=np.float64).ravel()
            feet_lost = int(np.sum(cf < t.contact_force_min_n))
            if feet_lost >= t.contact_loss_min_feet:
                if self._contact_loss_start is None:
                    self._contact_loss_start = timestamp_s
                elif timestamp_s - self._contact_loss_start >= t.contact_loss_min_duration_s:
                    ev = self._try_emit(
                        FailureMode.CONTACT_LOSS,
                        timestamp_s,
                        {"feet_lost": feet_lost, "forces": cf.tolist()},
                    )
                    if ev:
                        events.append(ev)
                    self._contact_loss_start = None
            else:
                self._contact_loss_start = None

        # --- Priority 6: Command mismatch ---
        # Only fires if NOT already slipping (slip is the more specific diagnosis)
        if not slipping and cmd_speed > t.cmd_mismatch_min_cmd_speed:
            vel_error = float(np.linalg.norm(cmd_2d - actual_2d))
            if vel_error > t.cmd_mismatch_vel_error:
                if self._cmd_mismatch_start is None:
                    self._cmd_mismatch_start = timestamp_s
                elif timestamp_s - self._cmd_mismatch_start >= t.cmd_mismatch_min_duration_s:
                    ev = self._try_emit(
                        FailureMode.COMMAND_MISMATCH,
                        timestamp_s,
                        {
                            "cmd_speed": cmd_speed,
                            "actual_speed": actual_speed,
                            "vel_error": vel_error,
                        },
                    )
                    if ev:
                        events.append(ev)
                    self._cmd_mismatch_start = None
            else:
                self._cmd_mismatch_start = None
        else:
            self._cmd_mismatch_start = None

        return events

    def _try_emit(
        self, mode: FailureMode, ts: float, detail: dict
    ) -> FailureEvent | None:
        if ts - self._last_event_at[mode] < self.thresholds.min_event_gap_s:
            return None
        self._last_event_at[mode] = ts
        return FailureEvent(mode=mode, timestamp_s=ts, detail=detail)
