"""Cause, response and phenotype: the vocabulary every Ashfall artifact uses.

The previous design carried one string, ``failure_mode``, which named the
detector threshold that fired, the thing the curriculum was supposed to teach,
and the physical cause that was supposed to be reproduced. Those are three
different objects and the conflation is what let a friction sweep be reported
as a "slip" treatment with nothing checking that any slip occurred.

The ontology here is

    Intervention (cause)  ->  Dynamical response  ->  Failure phenotype

* An :class:`Intervention` is a physical causal manipulation of the simulator
  or the robot: friction reduction, actuator weakening, terrain geometry, an
  external perturbation, command corruption, payload mass. It is applied, and
  its application is verified by readback (gate D1 in :mod:`ashfall.gates`).
* The dynamical response is the trajectory itself. Whether it departs from the
  matched no-intervention counterfactual is measured, not assumed (gate D2).
* A failure phenotype is an observed behavioural failure: slip, collapse,
  stumble or blockage, command mismatch, contact loss, attitude loss. It is
  detected, and the detector's accuracy is a separate empirical question
  (:mod:`ashfall.detector_eval`).

No function in this module maps an intervention to a single phenotype. An
intervention can produce several phenotypes or none, a phenotype can arise
from several interventions, and :data:`PLAUSIBLE_PATHWAYS` records the
hypotheses being tested, not a lookup table.

Every dataclass here is frozen, validated on construction, JSON round-trippable
and content-addressed through :func:`ashfall.provenance.content_hash`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from ashfall.provenance import content_hash

# --------------------------------------------------------------------------- #
# Interventions (causes)
# --------------------------------------------------------------------------- #

#: Kinds of physical causal manipulation. ``none`` is the matched control.
INTERVENTION_KINDS: tuple[str, ...] = (
    "none",
    "friction_reduction",
    "actuator_weakening",
    "terrain_geometry",
    "external_perturbation",
    "command_corruption",
    "payload_mass",
)

#: What each intervention kind manipulates. Recorded on the intervention so a
#: backend can refuse a target it cannot write.
INTERVENTION_TARGETS: Mapping[str, str] = {
    "none": "none",
    "friction_reduction": "robot_shape_material",
    "actuator_weakening": "actuator_gains",
    "terrain_geometry": "terrain",
    "external_perturbation": "root_velocity",
    "command_corruption": "velocity_command",
    "payload_mass": "base_mass",
}

#: Parameter names each kind is allowed to carry. A parameter outside this set
#: is rejected, so an intervention cannot smuggle in an undeclared manipulation.
INTERVENTION_PARAMETERS: Mapping[str, tuple[str, ...]] = {
    "none": (),
    "friction_reduction": ("static_friction", "dynamic_friction"),
    "actuator_weakening": ("stiffness_scale", "damping_scale", "effort_scale"),
    "terrain_geometry": ("step_height_m", "slope_rad", "obstacle_height_m"),
    "external_perturbation": ("push_vx_mps", "push_vy_mps", "push_time_s"),
    "command_corruption": ("command_vx_mps", "command_vy_mps", "command_yaw_radps"),
    "payload_mass": ("mass_offset_kg",),
}


@dataclass(frozen=True)
class Intervention:
    """A physical causal manipulation, or the explicit absence of one.

    ``intensity`` is a scalar position on a declared severity axis (for
    example ``1 - dynamic_friction``) used for H3 generalization over withheld
    intensities. It is descriptive: the parameters are what a backend applies.
    """

    kind: str
    parameters: tuple[tuple[str, float], ...] = ()
    intensity: float | None = None
    intensity_axis: str | None = None

    def __post_init__(self):
        if self.kind not in INTERVENTION_KINDS:
            if self.kind in PHENOTYPE_NAMES:
                raise ValueError(
                    f"{self.kind!r} is a failure phenotype, not an intervention. A phenotype "
                    "is observed; an intervention is applied. Name the physical manipulation."
                )
            raise ValueError(
                f"unknown intervention kind {self.kind!r}; expected {INTERVENTION_KINDS}"
            )
        params = tuple(sorted((str(k), float(v)) for k, v in self.parameters))
        if len({k for k, _ in params}) != len(params):
            raise ValueError("duplicate intervention parameter")
        allowed = set(INTERVENTION_PARAMETERS[self.kind])
        unknown = sorted({k for k, _ in params} - allowed)
        if unknown:
            raise ValueError(
                f"intervention {self.kind!r} does not accept parameters {unknown}; "
                f"allowed: {sorted(allowed)}"
            )
        if any(not math.isfinite(v) for _, v in params):
            raise ValueError("intervention parameters must be finite")
        if self.kind == "none" and params:
            raise ValueError("the control intervention carries no parameters")
        if self.kind != "none" and not params:
            raise ValueError(f"intervention {self.kind!r} needs at least one parameter")
        if (self.intensity is None) != (self.intensity_axis is None):
            raise ValueError("intensity and intensity_axis are declared together or not at all")
        if self.intensity is not None and not math.isfinite(self.intensity):
            raise ValueError("intensity must be finite")
        object.__setattr__(self, "parameters", params)

    @classmethod
    def none(cls) -> "Intervention":
        return cls("none")

    @classmethod
    def friction_reduction(cls, static_friction: float, dynamic_friction: float) -> "Intervention":
        if dynamic_friction > static_friction:
            raise ValueError("dynamic friction must not exceed static friction")
        return cls(
            "friction_reduction",
            (("static_friction", static_friction), ("dynamic_friction", dynamic_friction)),
            intensity=1.0 - dynamic_friction,
            intensity_axis="1-dynamic_friction",
        )

    @property
    def is_control(self) -> bool:
        return self.kind == "none"

    @property
    def target(self) -> str:
        return INTERVENTION_TARGETS[self.kind]

    @property
    def intervention_id(self) -> str:
        return "int_" + content_hash(asdict(self))

    def to_dict(self) -> dict:
        return {**asdict(self), "intervention_id": self.intervention_id, "target": self.target}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Intervention":
        values = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        values["parameters"] = tuple(tuple(p) for p in values.get("parameters", ()))
        result = cls(**values)
        if "intervention_id" in data and data["intervention_id"] != result.intervention_id:
            raise ValueError("intervention content identity mismatch")
        return result


# --------------------------------------------------------------------------- #
# Phenotypes (observed behavioural failures)
# --------------------------------------------------------------------------- #

#: Observability of a phenotype on a platform. ``supported`` means the
#: channels its definition needs are measured there; anything else names why not.
SUPPORTED = "supported"


@dataclass(frozen=True)
class PhenotypeSpec:
    """A behavioural failure phenotype and what it takes to observe it.

    ``signature_channels`` are what the threshold detector reads.
    ``ground_truth_channels`` are what an independent, non-detector labelling
    needs; when a platform lacks them the phenotype is *unsupported* there and
    no detector accuracy claim may be made for it on that platform.
    """

    name: str
    label: str
    definition: str
    signature_channels: tuple[str, ...]
    ground_truth_channels: tuple[str, ...]
    observability: Mapping[str, str]
    first_confirmatory_study: str
    exclusion_reason: str | None = None

    def supported_on(self, platform: str) -> bool:
        try:
            return self.observability[platform] == SUPPORTED
        except KeyError:
            raise ValueError(f"unknown platform {platform!r}") from None

    def unsupported_reason(self, platform: str) -> str | None:
        status = self.observability[platform]
        return None if status == SUPPORTED else status


PHENOTYPES: Mapping[str, PhenotypeSpec] = {
    "slip": PhenotypeSpec(
        name="slip",
        label="Foot slip / traction loss",
        definition=(
            "Feet in contact translate relative to the support surface while the body "
            "fails to track the commanded planar velocity."
        ),
        signature_channels=("base_lin_vel_body", "command_vel"),
        ground_truth_channels=("foot_contact", "foot_velocity_world"),
        observability={
            "simulation": SUPPORTED,
            "go2_hardware": (
                "no calibrated per-foot contact or foot velocity; the hardware detector "
                "sees a tracking stall, which is a superset of slip"
            ),
        },
        first_confirmatory_study="included",
    ),
    "collapse": PhenotypeSpec(
        name="collapse",
        label="Body collapse",
        definition="The trunk descends to ground contact or below a standing-height floor.",
        signature_channels=("base_pos",),
        ground_truth_channels=("base_height_ground_relative", "trunk_contact"),
        observability={
            "simulation": SUPPORTED,
            "go2_hardware": (
                "no validated ground-relative height source; odometry z is boot-relative"
            ),
        },
        first_confirmatory_study="included",
    ),
    "stumble": PhenotypeSpec(
        name="stumble",
        label="Stumble / blockage",
        definition=(
            "A swing foot is arrested by an obstacle or step and the gait is interrupted; "
            "joint velocity spikes while the body is supported."
        ),
        signature_channels=("joint_vel", "contact_forces"),
        ground_truth_channels=("foot_contact", "terrain_geometry"),
        observability={
            "simulation": "no terrain-geometry intervention is wired in the backend",
            "go2_hardware": "no per-foot contact signal",
        },
        first_confirmatory_study="excluded",
        exclusion_reason="requires a terrain geometry intervention the backend cannot apply",
    ),
    "command_mismatch": PhenotypeSpec(
        name="command_mismatch",
        label="Command mismatch",
        definition=(
            "The body moves but its planar velocity deviates from the applied command by a "
            "sustained margin, without a traction stall."
        ),
        signature_channels=("base_lin_vel_body", "command_vel"),
        ground_truth_channels=("applied_command", "base_lin_vel_body"),
        observability={
            "simulation": SUPPORTED,
            "go2_hardware": SUPPORTED,
        },
        first_confirmatory_study="included",
    ),
    "contact_loss": PhenotypeSpec(
        name="contact_loss",
        label="Contact loss",
        definition=(
            "Two or more feet lose support contact outside a gait flight phase, induced "
            "through support, terrain or environment changes rather than written as state."
        ),
        signature_channels=("contact_forces",),
        ground_truth_channels=("foot_contact", "gait_phase", "support_geometry"),
        observability={
            "simulation": "no support/terrain intervention is wired and gait phase is not labelled",
            "go2_hardware": "foot_force is uncalibrated counts with undocumented leg ordering",
        },
        first_confirmatory_study="excluded",
        exclusion_reason=(
            "no supported inducing intervention and no gait-phase-conditioned ground truth; "
            "contact force is not a restorable state and is never written by a reset"
        ),
    ),
    "attitude": PhenotypeSpec(
        name="attitude",
        label="Attitude loss",
        definition="Roll or pitch leaves the recoverable envelope; the body is tipping.",
        signature_channels=("base_quat",),
        ground_truth_channels=("base_quat",),
        observability={
            "simulation": SUPPORTED,
            "go2_hardware": SUPPORTED,
        },
        first_confirmatory_study="included",
    ),
}

PHENOTYPE_NAMES: tuple[str, ...] = tuple(PHENOTYPES)

#: The phenotypes the first confirmatory study aggregates for its primary
#: endpoint. Derived from the specs, never hand-listed elsewhere.
FIRST_STUDY_PHENOTYPES: tuple[str, ...] = tuple(
    name for name, spec in PHENOTYPES.items() if spec.first_confirmatory_study == "included"
)

#: Hypothesised cause -> phenotype pathways. Many-to-many by design. Nothing
#: in Ashfall treats membership here as established; the H0 gate tests it.
PLAUSIBLE_PATHWAYS: Mapping[str, tuple[str, ...]] = {
    "friction_reduction": ("slip", "attitude", "collapse"),
    "actuator_weakening": ("collapse", "attitude", "command_mismatch"),
    "terrain_geometry": ("stumble", "contact_loss", "attitude"),
    "external_perturbation": ("attitude", "collapse", "contact_loss"),
    "command_corruption": ("command_mismatch", "slip", "attitude"),
    "payload_mass": ("collapse", "command_mismatch"),
}


def plausible_phenotypes(intervention_kind: str) -> tuple[str, ...]:
    """Phenotypes an intervention is hypothesised to produce. Never a single answer."""
    if intervention_kind == "none":
        return ()
    if intervention_kind not in INTERVENTION_KINDS:
        raise ValueError(f"unknown intervention kind {intervention_kind!r}")
    return PLAUSIBLE_PATHWAYS[intervention_kind]


def assert_not_conflated(intervention_kind: str, phenotype: str) -> None:
    """Refuse the historical conflation of cause and effect in one name."""
    if intervention_kind in PHENOTYPE_NAMES:
        raise ValueError(f"{intervention_kind!r} names a phenotype; an intervention is a cause")
    if phenotype in INTERVENTION_KINDS:
        raise ValueError(f"{phenotype!r} names an intervention; a phenotype is an observed effect")
    if intervention_kind not in INTERVENTION_KINDS:
        raise ValueError(f"unknown intervention kind {intervention_kind!r}")
    if phenotype not in PHENOTYPE_NAMES:
        raise ValueError(f"unknown phenotype {phenotype!r}")
    if intervention_kind == "none":
        raise ValueError("the control intervention has no intended phenotype")


# --------------------------------------------------------------------------- #
# Observations with windows
# --------------------------------------------------------------------------- #

LABEL_SOURCES: tuple[str, ...] = (
    "detector",
    "simulator_ground_truth",
    "human_review",
    "synthetic_fixture",
)


@dataclass(frozen=True)
class OnsetWindow:
    """Where a phenotype develops, as three indices rather than one timestamp.

    ``precursor_start``  first frame that departs from the matched nominal
                         trajectory (may be None when nothing precedes
                         transition, e.g. an impulsive failure);
    ``transition_start`` first frame at which the phenotype is developing;
    ``established_start`` first frame at which the phenotype's definition is
                         met (a detector's onset would be at or after this);
    ``end``              last frame (inclusive) of the phenotype, or None when
                         it persists to the end of the recording.
    """

    transition_start: int
    established_start: int
    precursor_start: int | None = None
    end: int | None = None
    label_source: str = "detector"

    def __post_init__(self):
        for name in ("transition_start", "established_start"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.precursor_start is not None and (
            type(self.precursor_start) is not int or self.precursor_start < 0
        ):
            raise ValueError("precursor_start must be a nonnegative integer or None")
        if self.end is not None and (type(self.end) is not int or self.end < 0):
            raise ValueError("end must be a nonnegative integer or None")
        order = [
            v
            for v in (self.precursor_start, self.transition_start, self.established_start, self.end)
            if v is not None
        ]
        if order != sorted(order):
            raise ValueError(
                "window indices must satisfy precursor <= transition <= established <= end"
            )
        if self.label_source not in LABEL_SOURCES:
            raise ValueError(f"unknown label source {self.label_source!r}")

    @property
    def development_frames(self) -> int:
        return self.established_start - self.transition_start

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PhenotypeObservation:
    phenotype: str
    window: OnsetWindow
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.phenotype in INTERVENTION_KINDS:
            raise ValueError(
                f"{self.phenotype!r} is an intervention kind, not an observable phenotype"
            )
        if self.phenotype not in PHENOTYPE_NAMES:
            raise ValueError(f"unknown phenotype {self.phenotype!r}")
        if not isinstance(self.window, OnsetWindow):
            raise ValueError("window must be an OnsetWindow")
        if not isinstance(self.evidence, Mapping):
            raise ValueError("evidence must be a mapping")

    def to_dict(self) -> dict:
        return {
            "phenotype": self.phenotype,
            "window": self.window.to_dict(),
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhenotypeObservation":
        return cls(data["phenotype"], OnsetWindow(**data["window"]), data.get("evidence", {}))


# --------------------------------------------------------------------------- #
# Episodes, attempts and verdicts
# --------------------------------------------------------------------------- #

EPISODE_SOURCES: tuple[str, ...] = ("simulation", "hardware", "synthetic_fixture")


def _hex64(value: str | None, name: str) -> None:
    if value is not None and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be lowercase SHA256")


@dataclass(frozen=True)
class FailureEpisode:
    """One physics rollout and everything needed to identify it.

    ``provenance`` must name the simulator version, seed, environment
    configuration hash, backend identity and repository revisions; the
    validator refuses an episode that cannot be traced.
    """

    source: str
    policy_id: str
    intervention: Intervention
    command: tuple[float, float, float]
    initial_state_id: str
    control_dt: float
    n_frames: int
    termination_reason: str
    phenotypes: tuple[PhenotypeObservation, ...]
    provenance: Mapping[str, Any]
    trajectory_sha256: str | None = None
    seed: int | None = None
    episode_id: str = ""

    REQUIRED_PROVENANCE = ("simulator_version", "seed", "env_config_hash", "backend_id")

    def __post_init__(self):
        if self.source not in EPISODE_SOURCES:
            raise ValueError(f"unknown episode source {self.source!r}")
        if not self.policy_id or not self.initial_state_id or not self.termination_reason:
            raise ValueError("policy, initial state and termination identity are required")
        if not isinstance(self.intervention, Intervention):
            raise ValueError("intervention must be an Intervention")
        command = tuple(float(v) for v in self.command)
        if len(command) != 3 or not all(math.isfinite(v) for v in command):
            raise ValueError("command must be three finite values")
        object.__setattr__(self, "command", command)
        if not math.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt must be positive and finite")
        if type(self.n_frames) is not int or self.n_frames < 1:
            raise ValueError("n_frames must be a positive integer")
        object.__setattr__(self, "phenotypes", tuple(self.phenotypes))
        if not all(isinstance(p, PhenotypeObservation) for p in self.phenotypes):
            raise ValueError("phenotypes must be PhenotypeObservation records")
        for observation in self.phenotypes:
            last = observation.window.end
            if observation.window.established_start >= self.n_frames or (
                last is not None and last >= self.n_frames
            ):
                raise ValueError("phenotype window lies outside the episode")
        _hex64(self.trajectory_sha256, "trajectory_sha256")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        missing = [k for k in self.REQUIRED_PROVENANCE if self.provenance.get(k) is None]
        if missing:
            raise ValueError(f"episode provenance is missing {missing}")
        expected = self.content_id()
        if self.episode_id and self.episode_id != expected:
            raise ValueError("episode_id does not match content")
        object.__setattr__(self, "episode_id", expected)

    def content_id(self) -> str:
        data = self.to_dict(include_id=False)
        return "ep_" + content_hash(data)

    @property
    def phenotype_names(self) -> tuple[str, ...]:
        return tuple(sorted({p.phenotype for p in self.phenotypes}))

    def to_dict(self, include_id: bool = True) -> dict:
        data = {
            "source": self.source,
            "policy_id": self.policy_id,
            "intervention": self.intervention.to_dict(),
            "command": list(self.command),
            "initial_state_id": self.initial_state_id,
            "control_dt": self.control_dt,
            "n_frames": self.n_frames,
            "termination_reason": self.termination_reason,
            "phenotypes": [p.to_dict() for p in self.phenotypes],
            "provenance": dict(self.provenance),
            "trajectory_sha256": self.trajectory_sha256,
            "seed": self.seed,
        }
        if include_id:
            data["episode_id"] = self.episode_id
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FailureEpisode":
        return cls(
            source=data["source"],
            policy_id=data["policy_id"],
            intervention=Intervention.from_dict(data["intervention"]),
            command=tuple(data["command"]),
            initial_state_id=data["initial_state_id"],
            control_dt=data["control_dt"],
            n_frames=data["n_frames"],
            termination_reason=data["termination_reason"],
            phenotypes=tuple(PhenotypeObservation.from_dict(p) for p in data["phenotypes"]),
            provenance=data["provenance"],
            trajectory_sha256=data.get("trajectory_sha256"),
            seed=data.get("seed"),
            episode_id=data.get("episode_id", ""),
        )


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.name not in ("D1_intervention", "D2_departure", "D3_phenotype"):
            raise ValueError(f"unknown gate {self.name!r}")
        if type(self.passed) is not bool:
            raise ValueError("gate result must be a boolean")

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": dict(self.detail)}


DELIVERY_STATUSES: tuple[str, ...] = (
    "DELIVERED",
    "NOT_APPLIED",
    "NO_DEPARTURE",
    "PHENOTYPE_ABSENT",
    "WRONG_PHENOTYPE",
    "UNSUPPORTED",
)


@dataclass(frozen=True)
class DeliveryVerdict:
    """Only D1 and D2 and D3 is DELIVERED; every other outcome names its gate."""

    intended_phenotype: str
    intervention: Intervention
    d1_intervention: GateResult
    d2_departure: GateResult
    d3_phenotype: GateResult
    status: str = ""

    def __post_init__(self):
        assert_not_conflated(self.intervention.kind, self.intended_phenotype)
        if (
            self.d1_intervention.name != "D1_intervention"
            or self.d2_departure.name != "D2_departure"
        ):
            raise ValueError("gate results are in the wrong slots")
        if self.d3_phenotype.name != "D3_phenotype":
            raise ValueError("gate results are in the wrong slots")
        expected = self._derive_status()
        if self.status and self.status != expected:
            raise ValueError(f"status {self.status!r} contradicts the gate results ({expected})")
        object.__setattr__(self, "status", expected)

    def _derive_status(self) -> str:
        if self.d1_intervention.detail.get("unsupported"):
            return "UNSUPPORTED"
        if not self.d1_intervention.passed:
            return "NOT_APPLIED"
        if not self.d2_departure.passed:
            return "NO_DEPARTURE"
        if not self.d3_phenotype.passed:
            observed = tuple(self.d3_phenotype.detail.get("observed_phenotypes", ()))
            return "WRONG_PHENOTYPE" if observed else "PHENOTYPE_ABSENT"
        return "DELIVERED"

    @property
    def delivered(self) -> bool:
        return self.d1_intervention.passed and self.d2_departure.passed and self.d3_phenotype.passed

    def to_dict(self) -> dict:
        return {
            "intended_phenotype": self.intended_phenotype,
            "intervention": self.intervention.to_dict(),
            "d1_intervention": self.d1_intervention.to_dict(),
            "d2_departure": self.d2_departure.to_dict(),
            "d3_phenotype": self.d3_phenotype.to_dict(),
            "status": self.status,
            "delivered": self.delivered,
        }


@dataclass(frozen=True)
class ReproductionAttempt:
    """One matched counterfactual pair and the verdict it earned.

    Both episodes share the initial state, policy, command, simulator seed,
    environment configuration and horizon; they differ only in the
    intervention. The attempt is retained whatever the verdict.
    """

    initial_state_id: str
    policy_id: str
    intervention: Intervention
    intended_phenotype: str
    simulator_seed: int
    control_episode_id: str
    treatment_episode_id: str
    backend_id: str
    evidence_kind: str
    verdict: DeliveryVerdict
    capsule_id: str | None = None
    attempt_id: str = ""

    def __post_init__(self):
        if self.evidence_kind not in ("simulation", "mock"):
            raise ValueError("evidence_kind must be simulation or mock")
        if type(self.simulator_seed) is not int or self.simulator_seed < 0:
            raise ValueError("simulator_seed must be a nonnegative integer")
        if not all(
            (
                self.initial_state_id,
                self.policy_id,
                self.control_episode_id,
                self.treatment_episode_id,
                self.backend_id,
            )
        ):
            raise ValueError("attempt identities are required")
        if self.control_episode_id == self.treatment_episode_id:
            raise ValueError("control and treatment episodes must be distinct")
        if self.verdict.intervention != self.intervention:
            raise ValueError("verdict intervention differs from attempt intervention")
        if self.verdict.intended_phenotype != self.intended_phenotype:
            raise ValueError("verdict phenotype differs from attempt phenotype")
        expected = self.content_id()
        if self.attempt_id and self.attempt_id != expected:
            raise ValueError("attempt_id does not match content")
        object.__setattr__(self, "attempt_id", expected)

    def content_id(self) -> str:
        return "att_" + content_hash(self.to_dict(include_id=False))

    def to_dict(self, include_id: bool = True) -> dict:
        data = {
            "initial_state_id": self.initial_state_id,
            "policy_id": self.policy_id,
            "intervention": self.intervention.to_dict(),
            "intended_phenotype": self.intended_phenotype,
            "simulator_seed": self.simulator_seed,
            "control_episode_id": self.control_episode_id,
            "treatment_episode_id": self.treatment_episode_id,
            "backend_id": self.backend_id,
            "evidence_kind": self.evidence_kind,
            "verdict": self.verdict.to_dict(),
            "capsule_id": self.capsule_id,
        }
        if include_id:
            data["attempt_id"] = self.attempt_id
        return data


def phenotype_windows_from_events(
    events: Sequence[Mapping[str, Any]], *, control_dt: float, label_source: str = "detector"
) -> tuple[PhenotypeObservation, ...]:
    """Turn detector events (``mode``, ``timestamp_s``) into windowed observations.

    A threshold detector only knows the frame at which its criterion was
    first met, so ``transition_start == established_start`` and no precursor
    is claimed. The window machinery exists so that ground-truth and reviewed
    labels can say more; a detector label never pretends to.
    """
    if control_dt <= 0:
        raise ValueError("control_dt must be positive")
    first: dict[str, int] = {}
    for event in events:
        frame = int(round(float(event["timestamp_s"]) / control_dt))
        mode = str(event["mode"])
        if mode not in PHENOTYPE_NAMES:
            continue
        first[mode] = min(frame, first.get(mode, frame))
    return tuple(
        PhenotypeObservation(
            mode,
            OnsetWindow(frame, frame, None, None, label_source),
            {"detector_event_frame": frame},
        )
        for mode, frame in sorted(first.items())
    )


__all__ = [
    "DELIVERY_STATUSES",
    "EPISODE_SOURCES",
    "FIRST_STUDY_PHENOTYPES",
    "INTERVENTION_KINDS",
    "INTERVENTION_PARAMETERS",
    "INTERVENTION_TARGETS",
    "LABEL_SOURCES",
    "PHENOTYPES",
    "PHENOTYPE_NAMES",
    "PLAUSIBLE_PATHWAYS",
    "SUPPORTED",
    "DeliveryVerdict",
    "FailureEpisode",
    "GateResult",
    "Intervention",
    "OnsetWindow",
    "PhenotypeObservation",
    "PhenotypeSpec",
    "ReproductionAttempt",
    "assert_not_conflated",
    "phenotype_windows_from_events",
    "plausible_phenotypes",
]
