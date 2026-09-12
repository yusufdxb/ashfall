"""Matched counterfactual pairs and the multivariate departure that gate D2 measures.

A matched pair is two rollouts that share the restored initial state, the
policy, the command, the simulator seed, the environment configuration and the
horizon, and differ only in whether an intervention was applied. The control
arm is the no-intervention counterfactual; the treatment arm carries the
intervention.

Gate D2 asks whether the treatment trajectory departs from the matched nominal
trajectory by more than nominal trajectories depart from one another. The
comparison is multivariate over the physically restorable channels (root pose,
root velocity, joint state, command), standardised by a covariance estimated
from independent nominal rollouts of the same initial state under different
simulator seeds. The covariance is shrunk toward a scaled identity with the
Ledoit and Wolf (2004) closed-form intensity, and a small diagonal floor keeps
the inverse finite when a channel is constant across replicates. That floor is
numerical regularisation. It is not a claim about sensor resolution, and its
influence is reported by :func:`sensitivity_analysis` rather than assumed away.

Nothing here decides whether a phenotype occurred (gate D3) or whether the
intervention was applied (gate D1); see :mod:`ashfall.gates`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ashfall.delivery import quat_xyzw_to_roll_pitch
from ashfall.provenance import content_hash

#: Scalar features computed from restorable channels only. Contact force is
#: deliberately absent: it is an output of the physics engine, not a channel a
#: reset writes, so a departure measured on it could not be re-delivered.
DEPARTURE_FEATURES: tuple[str, ...] = (
    "base_height_m",
    "tilt_rad",
    "forward_speed_error_mps",
    "lateral_speed_error_mps",
    "vertical_speed_mps",
    "yaw_rate_error_radps",
    "pitch_rate_radps",
    "joint_speed_rms_radps",
)

#: Which feature carries each phenotype's interpretable directional signature,
#: and the sign in which it moves as the phenotype develops. Kept from the
#: earlier one-feature gate as a REPORTED metric; it is no longer the
#: criterion.
DIRECTIONAL_SIGNATURE: Mapping[str, tuple[str, int]] = {
    "attitude": ("tilt_rad", +1),
    "collapse": ("base_height_m", -1),
    "slip": ("forward_speed_error_mps", +1),
    "stumble": ("joint_speed_rms_radps", +1),
    "command_mismatch": ("forward_speed_error_mps", +1),
    "contact_loss": ("vertical_speed_mps", +1),
}


def _array(value, name: str, width: int | None = None, dtype=np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a (T, k) array, got shape {array.shape}")
    if width is not None and array.shape[1] != width:
        raise ValueError(f"{name} must have {width} columns, got {array.shape[1]}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} has non-finite values")
    return array


@dataclass(frozen=True)
class RestorableState:
    """The seven channels a reset writes, as one identifiable initial state."""

    base_pos: tuple[float, float, float]
    base_quat: tuple[float, float, float, float]
    base_lin_vel_body: tuple[float, float, float]
    base_ang_vel_body: tuple[float, float, float]
    joint_pos: tuple[float, ...]
    joint_vel: tuple[float, ...]
    command_vel: tuple[float, float, float]

    def __post_init__(self):
        for name, width in (
            ("base_pos", 3),
            ("base_quat", 4),
            ("base_lin_vel_body", 3),
            ("base_ang_vel_body", 3),
            ("command_vel", 3),
        ):
            values = tuple(float(v) for v in getattr(self, name))
            if len(values) != width or not all(math.isfinite(v) for v in values):
                raise ValueError(f"{name} must be {width} finite values")
            object.__setattr__(self, name, values)
        for name in ("joint_pos", "joint_vel"):
            values = tuple(float(v) for v in getattr(self, name))
            if not values or not all(math.isfinite(v) for v in values):
                raise ValueError(f"{name} must be nonempty and finite")
            object.__setattr__(self, name, values)
        if len(self.joint_pos) != len(self.joint_vel):
            raise ValueError("joint_pos and joint_vel disagree on joint count")
        norm = math.sqrt(sum(v * v for v in self.base_quat))
        if not math.isclose(norm, 1.0, abs_tol=1e-4):
            raise ValueError("base_quat must be a unit xyzw quaternion")
        if self.base_pos[2] == 0.0:
            raise ValueError("a base height of exactly zero is an unrecorded signal, not a state")

    @property
    def state_id(self) -> str:
        return "st_" + content_hash(asdict(self))

    @classmethod
    def from_frame(cls, frame) -> "RestorableState":
        missing = getattr(frame, "missing_reset_fields", ())
        if missing:
            raise ValueError(f"frame lacks restorable channels: {missing}")
        return cls(
            frame.base_pos,
            frame.base_quat,
            frame.base_lin_vel_body,
            frame.base_ang_vel_body,
            frame.joint_pos,
            frame.joint_vel,
            frame.command_vel,
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "state_id": self.state_id}


@dataclass(frozen=True)
class RolloutTrace:
    """One rollout's restorable channels per control step, plus diagnostics.

    Arrays are ``(T, k)``. ``contact_forces`` is diagnostic only and may be
    None. ``terminated_step`` is the step at which the simulator ended the
    episode for a non-timeout reason, or None.
    """

    control_dt: float
    base_pos: np.ndarray
    base_quat: np.ndarray
    base_lin_vel_body: np.ndarray
    base_ang_vel_body: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    command_vel: np.ndarray
    termination_reason: str
    terminated_step: int | None = None
    contact_forces: np.ndarray | None = None
    actions: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt must be positive and finite")
        arrays = {
            "base_pos": _array(self.base_pos, "base_pos", 3),
            "base_quat": _array(self.base_quat, "base_quat", 4),
            "base_lin_vel_body": _array(self.base_lin_vel_body, "base_lin_vel_body", 3),
            "base_ang_vel_body": _array(self.base_ang_vel_body, "base_ang_vel_body", 3),
            "joint_pos": _array(self.joint_pos, "joint_pos"),
            "joint_vel": _array(self.joint_vel, "joint_vel"),
            "command_vel": _array(self.command_vel, "command_vel", 3),
        }
        lengths = {a.shape[0] for a in arrays.values()}
        if len(lengths) != 1 or next(iter(lengths)) < 2:
            raise ValueError("all channels need the same length of at least two steps")
        if arrays["joint_pos"].shape != arrays["joint_vel"].shape:
            raise ValueError("joint_pos and joint_vel shapes differ")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        for name in ("contact_forces", "actions"):
            value = getattr(self, name)
            if value is not None:
                array = _array(value, name)
                if array.shape[0] != self.n_steps:
                    raise ValueError(f"{name} length differs from the trace")
                object.__setattr__(self, name, array)
        if not self.termination_reason:
            raise ValueError("termination_reason is required")
        if self.terminated_step is not None and not 0 <= self.terminated_step < self.n_steps:
            raise ValueError("terminated_step outside the trace")

    @property
    def n_steps(self) -> int:
        return int(self.base_pos.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_steps * self.control_dt

    def state_at(self, step: int) -> RestorableState:
        return RestorableState(
            tuple(self.base_pos[step]),
            tuple(self.base_quat[step]),
            tuple(self.base_lin_vel_body[step]),
            tuple(self.base_ang_vel_body[step]),
            tuple(self.joint_pos[step]),
            tuple(self.joint_vel[step]),
            tuple(self.command_vel[step]),
        )

    def features(self) -> np.ndarray:
        """``(T, len(DEPARTURE_FEATURES))`` feature matrix from restorable channels."""
        rolls_pitches = np.array([quat_xyzw_to_roll_pitch(q) for q in self.base_quat])
        tilt = np.hypot(rolls_pitches[:, 0], rolls_pitches[:, 1])
        forward_error = self.command_vel[:, 0] - self.base_lin_vel_body[:, 0]
        lateral_error = self.command_vel[:, 1] - self.base_lin_vel_body[:, 1]
        yaw_rate_error = self.command_vel[:, 2] - self.base_ang_vel_body[:, 2]
        joint_rms = np.sqrt(np.mean(np.square(self.joint_vel), axis=1))
        columns = [
            self.base_pos[:, 2],
            tilt,
            forward_error,
            lateral_error,
            self.base_lin_vel_body[:, 2],
            yaw_rate_error,
            self.base_ang_vel_body[:, 1],
            joint_rms,
        ]
        return np.stack(columns, axis=1)

    def content_hash(self) -> str:
        payload = {
            "control_dt": self.control_dt,
            "termination_reason": self.termination_reason,
            "terminated_step": self.terminated_step,
            "channels": {
                name: np.round(getattr(self, name), 7).tolist()
                for name in (
                    "base_pos",
                    "base_quat",
                    "base_lin_vel_body",
                    "base_ang_vel_body",
                    "joint_pos",
                    "joint_vel",
                    "command_vel",
                )
            },
        }
        return content_hash(payload)


# --------------------------------------------------------------------------- #
# Covariance estimation
# --------------------------------------------------------------------------- #


def ledoit_wolf_shrinkage(residuals: np.ndarray) -> float:
    """Closed-form shrinkage intensity toward a scaled identity (Ledoit and Wolf 2004).

    ``residuals`` is ``(n, p)`` and already centred. Returns the intensity in
    [0, 1]. Follows the estimator scikit-learn implements; written out here so
    Ashfall carries no dependency on it and the formula is auditable.
    """
    x = np.asarray(residuals, dtype=np.float64)
    n, p = x.shape
    if n < 2:
        return 1.0
    x2 = x**2
    emp_cov = x.T @ x / n
    mu = float(np.trace(emp_cov)) / p
    delta = float(np.sum((emp_cov - mu * np.eye(p)) ** 2)) / p
    beta = float(np.sum(x2.T @ x2 / n - emp_cov**2)) / (n * p)
    if delta <= 0:
        return 1.0
    return float(min(1.0, max(0.0, min(beta, delta) / delta)))


def shrunk_covariance(
    residuals: np.ndarray, *, shrinkage: float | str = "ledoit_wolf", floor: float = 1e-6
) -> tuple[np.ndarray, float]:
    """Shrunk covariance plus a diagonal floor. Returns ``(sigma, intensity_used)``.

    The floor is numerical regularisation only: it keeps the inverse defined
    when a channel is constant across the nominal replicates. Its effect on the
    verdict is reported by :func:`sensitivity_analysis`.
    """
    x = np.asarray(residuals, dtype=np.float64)
    n, p = x.shape
    if n < 2:
        raise ValueError("covariance needs at least two residual rows")
    if not math.isfinite(floor) or floor < 0:
        raise ValueError("floor must be finite and nonnegative")
    emp_cov = x.T @ x / n
    if shrinkage == "ledoit_wolf":
        intensity = ledoit_wolf_shrinkage(x)
    else:
        intensity = float(shrinkage)
        if not 0.0 <= intensity <= 1.0:
            raise ValueError("shrinkage intensity must lie in [0, 1]")
    mu = float(np.trace(emp_cov)) / p
    sigma = (1.0 - intensity) * emp_cov + intensity * mu * np.eye(p)
    sigma = sigma + floor * np.eye(p)
    return sigma, intensity


# --------------------------------------------------------------------------- #
# Departure
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DepartureConfig:
    """Preregistered D2 parameters. Defaults are protocol constants, not fits."""

    window_steps: int | None = None
    shrinkage: float | str = "ledoit_wolf"
    regularization_floor: float = 1e-6
    statistic: str = "q95"
    margin_ratio: float = 1.5
    min_replicates: int = 3

    def __post_init__(self):
        if self.statistic not in ("max", "q95", "mean"):
            raise ValueError("statistic must be max, q95 or mean")
        if not math.isfinite(self.margin_ratio) or self.margin_ratio < 1.0:
            raise ValueError("margin_ratio must be at least 1")
        if self.min_replicates < 2:
            raise ValueError("at least two nominal replicates are needed for a null")
        if self.window_steps is not None and self.window_steps < 2:
            raise ValueError("window_steps must be at least 2")
        if not math.isfinite(self.regularization_floor) or self.regularization_floor < 0:
            raise ValueError("regularization_floor must be finite and nonnegative")


def _summarise(distances: np.ndarray, statistic: str) -> float:
    if statistic == "max":
        return float(np.max(distances))
    if statistic == "q95":
        return float(np.quantile(distances, 0.95))
    return float(np.mean(distances))


def _common_window(traces: Sequence[RolloutTrace], window_steps: int | None) -> int:
    shortest = min(t.n_steps for t in traces)
    length = shortest if window_steps is None else min(window_steps, shortest)
    if length < 2:
        raise ValueError("traces share fewer than two steps")
    return length


@dataclass(frozen=True)
class DepartureResult:
    """What D2 measured. ``passed`` is derived, never asserted."""

    statistic: str
    treatment_vs_control: float
    null_pairwise: tuple[float, ...]
    null_max: float
    margin_ratio: float
    ratio_to_null_max: float
    empirical_p: float
    p_floor: float
    shrinkage_used: float
    regularization_floor: float
    replicates: int
    window_steps: int
    per_step_distance: tuple[float, ...]
    directional: Mapping[str, Any]
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _feature_residuals(
    nominal: Sequence[RolloutTrace], length: int
) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack([t.features()[:length] for t in nominal])  # (K, T, p)
    mean = stack.mean(axis=0)
    residuals = (stack - mean).reshape(-1, stack.shape[-1])
    return mean, residuals


def _mahalanobis(diff: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    solve = np.linalg.solve(sigma, diff.T)
    return np.sqrt(np.einsum("tp,pt->t", diff, solve))


def directional_metrics(
    treatment: RolloutTrace,
    control: RolloutTrace,
    nominal: Sequence[RolloutTrace],
    *,
    phenotype: str | None,
    length: int,
) -> dict:
    """Interpretable per-phenotype direction: reported alongside D2, not the criterion."""
    if phenotype is None or phenotype not in DIRECTIONAL_SIGNATURE:
        return {"phenotype": phenotype, "available": False}
    name, sign = DIRECTIONAL_SIGNATURE[phenotype]
    column = DEPARTURE_FEATURES.index(name)
    diff = treatment.features()[:length, column] - control.features()[:length, column]
    _, residuals = _feature_residuals(nominal, length)
    spread = float(np.std(residuals[:, column], ddof=1)) if residuals.shape[0] > 1 else 0.0
    signed = sign * diff
    return {
        "phenotype": phenotype,
        "available": True,
        "feature": name,
        "expected_sign": sign,
        "nominal_spread": spread,
        "final_signed_difference": float(signed[-1]),
        "max_signed_difference": float(np.max(signed)),
        "final_signed_z": float(signed[-1] / spread) if spread > 0 else None,
        "moves_in_expected_direction": bool(signed[-1] > 0),
    }


def measure_departure(
    treatment: RolloutTrace,
    control: RolloutTrace,
    nominal: Sequence[RolloutTrace],
    *,
    config: DepartureConfig | None = None,
    phenotype: str | None = None,
) -> DepartureResult:
    """Gate D2: multivariate departure of treatment from its matched control.

    ``nominal`` are independent rollouts of the same initial state and command
    under different simulator seeds with no intervention; the matched control
    is included in that ensemble. The null distribution is the same statistic
    computed for every pair of nominal replicates, so the treatment is judged
    against how much nominal rollouts differ from one another, not against a
    fixed sigma.

    Passes when the treatment statistic exceeds every pairwise null statistic
    by ``margin_ratio`` and the empirical exceedance p sits at its floor. With
    K replicates there are K(K-1)/2 null pairs, so the p floor is
    1/(pairs+1); the ratio, not the p, is the discriminating quantity at the
    small K a simulator budget allows, and both are reported.
    """
    config = config or DepartureConfig()
    replicates = list(nominal)
    if not any(t is control for t in replicates):
        replicates = [control, *replicates]
    if len(replicates) < config.min_replicates:
        raise ValueError(
            f"D2 needs at least {config.min_replicates} nominal replicates, got {len(replicates)}"
        )
    length = _common_window([treatment, *replicates], config.window_steps)
    mean, residuals = _feature_residuals(replicates, length)
    sigma, intensity = shrunk_covariance(
        residuals, shrinkage=config.shrinkage, floor=config.regularization_floor
    )
    features = [t.features()[:length] for t in replicates]
    control_index = next(i for i, t in enumerate(replicates) if t is control)
    treatment_distance = _mahalanobis(
        treatment.features()[:length] - features[control_index], sigma
    )
    treatment_stat = _summarise(treatment_distance, config.statistic)
    null = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            null.append(
                _summarise(_mahalanobis(features[i] - features[j], sigma), config.statistic)
            )
    null_arr = np.asarray(null)
    null_max = float(null_arr.max())
    exceed = int(np.sum(null_arr >= treatment_stat))
    empirical_p = (1 + exceed) / (len(null) + 1)
    ratio = treatment_stat / null_max if null_max > 0 else math.inf
    passed = bool(ratio >= config.margin_ratio and exceed == 0)
    return DepartureResult(
        statistic=config.statistic,
        treatment_vs_control=treatment_stat,
        null_pairwise=tuple(float(v) for v in null),
        null_max=null_max,
        margin_ratio=config.margin_ratio,
        ratio_to_null_max=float(ratio),
        empirical_p=float(empirical_p),
        p_floor=1.0 / (len(null) + 1),
        shrinkage_used=intensity,
        regularization_floor=config.regularization_floor,
        replicates=len(replicates),
        window_steps=length,
        per_step_distance=tuple(float(v) for v in treatment_distance),
        directional=directional_metrics(
            treatment, control, replicates, phenotype=phenotype, length=length
        ),
        passed=passed,
    )


def departure_onset(result: DepartureResult, *, sustain_steps: int = 5) -> int | None:
    """First step from which the per-step distance stays above the null maximum.

    This is the precursor boundary of an onset window: the earliest point at
    which the treatment is already distinguishable from nominal variation,
    which is usually before any threshold detector fires.
    """
    if sustain_steps < 1:
        raise ValueError("sustain_steps must be positive")
    above = np.asarray(result.per_step_distance) > result.null_max
    for start in range(len(above) - sustain_steps + 1):
        if above[start : start + sustain_steps].all():
            return start
    return None


# --------------------------------------------------------------------------- #
# Sensitivity to regularisation
# --------------------------------------------------------------------------- #

DEFAULT_SHRINKAGE_GRID: tuple[float | str, ...] = ("ledoit_wolf", 0.0, 0.1, 0.3, 0.6, 1.0)
DEFAULT_FLOOR_GRID: tuple[float, ...] = (1e-9, 1e-6, 1e-4, 1e-2)


@dataclass(frozen=True)
class SensitivityReport:
    primary_passed: bool
    cells: tuple[Mapping[str, Any], ...]
    agreement_fraction: float
    stable: bool

    def to_dict(self) -> dict:
        return asdict(self)


def sensitivity_analysis(
    treatment: RolloutTrace,
    control: RolloutTrace,
    nominal: Sequence[RolloutTrace],
    *,
    config: DepartureConfig | None = None,
    phenotype: str | None = None,
    shrinkage_grid: Sequence[float | str] = DEFAULT_SHRINKAGE_GRID,
    floor_grid: Sequence[float] = DEFAULT_FLOOR_GRID,
    stability_threshold: float = 0.9,
) -> SensitivityReport:
    """Re-run D2 over a grid of shrinkage intensities and diagonal floors.

    A verdict that flips under a different regulariser is a verdict about the
    regulariser. ``stable`` is True when at least ``stability_threshold`` of
    the grid agrees with the primary configuration.
    """
    config = config or DepartureConfig()
    primary = measure_departure(treatment, control, nominal, config=config, phenotype=phenotype)
    cells = []
    for shrinkage in shrinkage_grid:
        for floor in floor_grid:
            cell_config = DepartureConfig(
                window_steps=config.window_steps,
                shrinkage=shrinkage,
                regularization_floor=floor,
                statistic=config.statistic,
                margin_ratio=config.margin_ratio,
                min_replicates=config.min_replicates,
            )
            try:
                result = measure_departure(
                    treatment, control, nominal, config=cell_config, phenotype=phenotype
                )
                cells.append(
                    {
                        "shrinkage": shrinkage,
                        "floor": floor,
                        "passed": result.passed,
                        "ratio": result.ratio_to_null_max,
                        "shrinkage_used": result.shrinkage_used,
                    }
                )
            except np.linalg.LinAlgError as exc:
                cells.append(
                    {"shrinkage": shrinkage, "floor": floor, "passed": None, "error": str(exc)}
                )
    decided = [c for c in cells if c["passed"] is not None]
    agreement = (
        sum(1 for c in decided if c["passed"] == primary.passed) / len(decided) if decided else 0.0
    )
    return SensitivityReport(
        primary_passed=primary.passed,
        cells=tuple(cells),
        agreement_fraction=float(agreement),
        stable=bool(agreement >= stability_threshold),
    )


__all__ = [
    "DEFAULT_FLOOR_GRID",
    "DEFAULT_SHRINKAGE_GRID",
    "DEPARTURE_FEATURES",
    "DIRECTIONAL_SIGNATURE",
    "DepartureConfig",
    "DepartureResult",
    "RestorableState",
    "RolloutTrace",
    "SensitivityReport",
    "departure_onset",
    "directional_metrics",
    "ledoit_wolf_shrinkage",
    "measure_departure",
    "sensitivity_analysis",
    "shrunk_covariance",
]
