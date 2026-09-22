"""When a simulator replay counts as a reproduction of the observed failure.

Three checks, each named for what it tests. They are the main-paper form of
the D1/D2/D3 gates in :mod:`ashfall.gates`, restated against the observed
failure rather than against nominal replicates:

1. **delivered**: the friction the replay asked for was read back from the
   simulator (an :class:`ashfall.gates.InterventionReceipt` that verified);
2. **trajectory match**: from the restored pre-failure state, the treated
   replay stays closer to the recorded trajectory than its matched untreated
   control does, by a predeclared relative margin;
3. **failure match**: the treated replay meets the same failure criterion as
   the recording, with onset within a predeclared tolerance of the recorded
   onset, and the matched control does not meet it inside that window.

Control and treatment share the restored state, command, policy and simulator
seed and differ only in friction. A capsule is REPRODUCED at a friction value
when at least ``min_pass_fraction`` of its replicate pairs pass all three.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ashfall.fbr.criterion import FailureCriterion, detect_onset, tilt_from_quat_xyzw
from ashfall.gates import InterventionReceipt

#: Per-channel scales that make trajectory distances unitless. Predeclared.
DEFAULT_SCALES: Mapping[str, float] = {
    "tilt_rad": 0.1,
    "base_ang_vel_body": 0.5,
    "joint_pos": 0.1,
    "joint_vel": 2.0,
}


@dataclass(frozen=True)
class AcceptanceConfig:
    trajectory_margin: float = 0.2
    onset_tolerance_s: float = 0.3
    min_pass_fraction: float = 2 / 3
    scales: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_SCALES))

    def __post_init__(self):
        if not 0.0 <= self.trajectory_margin < 1.0:
            raise ValueError("trajectory_margin must lie in [0, 1)")
        if not (math.isfinite(self.onset_tolerance_s) and self.onset_tolerance_s > 0):
            raise ValueError("onset_tolerance_s must be positive")
        if not 0.0 < self.min_pass_fraction <= 1.0:
            raise ValueError("min_pass_fraction must lie in (0, 1]")
        if not self.scales or any(not v > 0 for v in self.scales.values()):
            raise ValueError("scales must be positive")


def _field(row: Any, name: str):
    return row.get(name) if isinstance(row, Mapping) else getattr(row, name, None)


def _features(rows: Sequence[Any], scales: Mapping[str, float]) -> np.ndarray:
    columns = []
    for name, scale in scales.items():
        if name == "tilt_rad":
            values = [[tilt_from_quat_xyzw(_field(r, "base_quat"))] for r in rows]
        else:
            values = [list(_field(r, name)) for r in rows]
        columns.append(np.asarray(values, dtype=float) / scale)
    return np.concatenate(columns, axis=1)


def trajectory_distance(
    observed: Sequence[Any], replay: Sequence[Any], scales: Mapping[str, float]
) -> float:
    """RMS scaled distance over the common prefix of two rollouts from one seed state.

    Both sequences start at the seed state and run at the same control period;
    the comparison stops at the shorter one, which is the observed failure
    window for the recording.
    """
    n = min(len(observed), len(replay))
    if n < 2:
        raise ValueError("need at least two aligned frames to compare trajectories")
    a, b = _features(observed[:n], scales), _features(replay[:n], scales)
    return float(np.sqrt(np.mean((a - b) ** 2)))


@dataclass(frozen=True)
class PairCheck:
    delivered: bool
    trajectory_match: bool
    failure_match: bool
    distance_treated: float
    distance_control: float
    treated_onset_s: float | None
    control_onset_s: float | None

    @property
    def passed(self) -> bool:
        return self.delivered and self.trajectory_match and self.failure_match


@dataclass(frozen=True)
class ReconstructionVerdict:
    mu: float
    status: str  # REPRODUCED or UNREPRODUCED
    pass_fraction: float
    pairs: tuple[PairCheck, ...]
    config: dict

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "pairs": [{**asdict(p), "passed": p.passed} for p in self.pairs],
        }


def check_pair(
    observed: Sequence[Any],
    observed_onset_s: float,
    control: Sequence[Any],
    treated: Sequence[Any],
    receipt: InterventionReceipt,
    criterion: FailureCriterion,
    config: AcceptanceConfig,
) -> PairCheck:
    """One matched pair. ``observed_onset_s`` is measured from the seed frame."""
    d_t = trajectory_distance(observed, treated, config.scales)
    d_c = trajectory_distance(observed, control, config.scales)

    def onset_s(rows):
        onset = detect_onset(rows, criterion)
        return None if onset is None else onset.timestamp_s - float(_field(rows[0], "timestamp_s"))

    treated_onset, control_onset = onset_s(treated), onset_s(control)
    window_end = observed_onset_s + config.onset_tolerance_s
    failure_match = (
        treated_onset is not None
        and abs(treated_onset - observed_onset_s) <= config.onset_tolerance_s
        and (control_onset is None or control_onset > window_end)
    )
    return PairCheck(
        delivered=bool(receipt.verified),
        trajectory_match=d_t <= (1.0 - config.trajectory_margin) * d_c,
        failure_match=bool(failure_match),
        distance_treated=d_t,
        distance_control=d_c,
        treated_onset_s=treated_onset,
        control_onset_s=control_onset,
    )


def accept_reconstruction(
    mu: float,
    observed: Sequence[Any],
    observed_onset_s: float,
    pairs: Sequence[tuple[Sequence[Any], Sequence[Any], InterventionReceipt]],
    criterion: FailureCriterion,
    config: AcceptanceConfig | None = None,
) -> ReconstructionVerdict:
    """REPRODUCED when enough replicate (control, treated, receipt) pairs pass all checks."""
    config = config or AcceptanceConfig()
    if not pairs:
        raise ValueError("reconstruction needs at least one matched pair")
    checks = tuple(
        check_pair(observed, observed_onset_s, c, t, r, criterion, config) for c, t, r in pairs
    )
    fraction = sum(c.passed for c in checks) / len(checks)
    status = "REPRODUCED" if fraction >= config.min_pass_fraction - 1e-12 else "UNREPRODUCED"
    return ReconstructionVerdict(float(mu), status, fraction, checks, asdict(config))
