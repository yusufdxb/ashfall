"""Adversarial fixtures, not independently labeled physical trajectories.

These are intended successful or non-target motions. Current detector false
positives are recorded in tests to expose signal ambiguity, not hidden by tuning
thresholds against the same generators. The wall fixture is a slip negative,
although blockage can legitimately be a command-tracking event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NegativeFixture:
    name: str
    target_negative_modes: tuple[str, ...]
    telemetry: tuple[dict, ...]


def adversarial_negatives(n_steps: int = 100, dt: float = 0.02) -> list[NegativeFixture]:
    if n_steps < 1 or dt <= 0:
        raise ValueError("positive duration and timestep required")
    specs = {
        "wall_blockage": ("slip",),
        "normal_trot_contact_loss": ("contact_loss",),
        "commanded_stop": ("slip", "command_mismatch"),
        "successful_turn": ("attitude", "command_mismatch"),
        "successful_bound_flight": ("contact_loss",),
        "high_joint_speed": ("stumble",),
        "temporary_tracking_lag": ("slip", "command_mismatch"),
        "intended_low_posture": ("collapse",),
    }
    result = []
    for name, modes in specs.items():
        trajectory = []
        for i in range(n_steps):
            row = dict(
                timestamp_s=i * dt,
                pitch_rad=0.0,
                roll_rad=0.0,
                base_height_m=0.3,
                cmd_lin_vel=np.array([0.5, 0.0]),
                actual_lin_vel=np.array([0.5, 0.0]),
                joint_vel=np.zeros(12),
                contact_forces=np.full(4, 50.0),
            )
            if name == "wall_blockage":
                row["actual_lin_vel"] = np.zeros(2)
            elif name == "normal_trot_contact_loss":
                row["contact_forces"] = np.array(
                    [0.0, 50.0, 50.0, 0.0] if (i // 10) % 2 else [50.0, 0.0, 0.0, 50.0]
                )
            elif name == "commanded_stop":
                row["cmd_lin_vel"] = row["actual_lin_vel"] = np.zeros(2)
            elif name == "successful_turn":
                row["roll_rad"] = 0.4
                row["cmd_lin_vel"] = row["actual_lin_vel"] = np.array([0.6, 0.3])
            elif name == "successful_bound_flight" and i % 30 < 8:
                row["contact_forces"] = np.zeros(4)
            elif name == "high_joint_speed":
                row["joint_vel"] = np.full(12, 18.0)
            elif name == "temporary_tracking_lag" and i < 10:
                row["actual_lin_vel"] = np.zeros(2)
            elif name == "intended_low_posture":
                row["base_height_m"] = 0.12
            trajectory.append(row)
        result.append(NegativeFixture(name, modes, tuple(trajectory)))
    return result
