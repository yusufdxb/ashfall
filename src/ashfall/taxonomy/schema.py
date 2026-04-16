"""Failure taxonomy schema and metadata.

Provides structured metadata about each failure mode: description, detection
method, sim-reproducibility, and replay strategy. This powers both the
README taxonomy table and the experiment runner's mode-aware logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from ashfall.taxonomy.detector import FailureMode


@dataclass(frozen=True)
class FailureModeSpec:
    mode: FailureMode
    label: str
    description: str
    detection: str
    sim_reproducible: bool
    replay_strategy: str
    severity: int


TAXONOMY: dict[FailureMode, FailureModeSpec] = {
    FailureMode.ATTITUDE: FailureModeSpec(
        mode=FailureMode.ATTITUDE,
        label="Attitude Loss",
        description="Pitch or roll exceeds safety threshold — robot approaching tip-over.",
        detection="Instantaneous: |pitch| > 0.8 rad or |roll| > 0.6 rad.",
        sim_reproducible=True,
        replay_strategy="Reconstruct initial pose and velocity, sweep friction + push forces.",
        severity=4,
    ),
    FailureMode.COLLAPSE: FailureModeSpec(
        mode=FailureMode.COLLAPSE,
        label="Body Collapse",
        description="Base height drops below floor threshold — leg buckle or belly contact.",
        detection="Instantaneous: base_height < 0.15 m.",
        sim_reproducible=True,
        replay_strategy="Spawn from pre-failure state, vary terrain + joint stiffness.",
        severity=5,
    ),
    FailureMode.SLIP: FailureModeSpec(
        mode=FailureMode.SLIP,
        label="Foot Slip",
        description="High command velocity but near-zero actual velocity — loss of traction.",
        detection="Sustained: cmd_speed > 0.3 m/s and actual_speed < 0.05 m/s for 0.5 s.",
        sim_reproducible=True,
        replay_strategy="Reconstruct with low-friction terrain, sweep friction coefficient.",
        severity=3,
    ),
    FailureMode.STUMBLE: FailureModeSpec(
        mode=FailureMode.STUMBLE,
        label="Stumble",
        description="Transient foot-catch during swing phase — sudden joint velocity spike.",
        detection="Instantaneous: max |joint_vel| > 15 rad/s with >= 2 feet in contact.",
        sim_reproducible=True,
        replay_strategy="Add terrain obstacles at swing-foot trajectory height.",
        severity=2,
    ),
    FailureMode.CONTACT_LOSS: FailureModeSpec(
        mode=FailureMode.CONTACT_LOSS,
        label="Contact Loss",
        description="Multiple feet lose ground contact during expected stance phase.",
        detection="Sustained: >= 2 feet below 5N force for >= 0.1 s.",
        sim_reproducible=True,
        replay_strategy="Vary terrain slope and surface irregularity.",
        severity=2,
    ),
    FailureMode.COMMAND_MISMATCH: FailureModeSpec(
        mode=FailureMode.COMMAND_MISMATCH,
        label="Command Mismatch",
        description="Sustained velocity tracking error — robot moves but not as commanded.",
        detection="Sustained: |cmd - actual| > 0.4 m/s for > 1.0 s (excludes slip).",
        sim_reproducible=True,
        replay_strategy="Replay with same commands, sweep mass + actuator strength.",
        severity=1,
    ),
}


def taxonomy_table_rows() -> list[dict[str, str]]:
    """Return taxonomy as a list of dicts for tabulate/markdown rendering."""
    rows = []
    for spec in TAXONOMY.values():
        rows.append(
            {
                "Mode": spec.label,
                "Severity": str(spec.severity),
                "Detection": spec.detection,
                "Sim Replay": spec.replay_strategy,
            }
        )
    return sorted(rows, key=lambda r: -int(r["Severity"]))
