"""The three delivery gates: D1 intervention applied, D2 departure, D3 phenotype.

    DELIVERED  iff  D1 and D2 and D3

D1 asks the backend, not the request: an :class:`InterventionReceipt` records
what was requested, what the simulator read back after the write, and whether
they agree within tolerance. A pair without a verified receipt is NOT_APPLIED
whatever the trajectories did, because a departure that cannot be attributed
to an applied intervention is not evidence of causation.

D2 is :func:`ashfall.counterfactual.measure_departure`: multivariate departure
of the treatment from its matched control, judged against how much independent
nominal rollouts of the same state differ from one another.

D3 asks whether the intended phenotype occurred in the treatment arm and did
NOT occur in the matched control arm. A phenotype present in both arms is not
caused by the intervention. Phenotype presence comes from the threshold
detector today; the detector's accuracy is a separate empirical question and
the verdict records the detector version it used.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ashfall.counterfactual import (
    DepartureConfig,
    DepartureResult,
    RolloutTrace,
    departure_onset,
    measure_departure,
)
from ashfall.delivery import quat_xyzw_to_roll_pitch
from ashfall.ontology import (
    PHENOTYPES,
    DeliveryVerdict,
    GateResult,
    Intervention,
    OnsetWindow,
    PhenotypeObservation,
    assert_not_conflated,
)
from ashfall.taxonomy.detector import FailureDetector, FailureThresholds

DETECTOR_VERSION = "ashfall.threshold.v1"


# --------------------------------------------------------------------------- #
# D1
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InterventionReceipt:
    """Evidence that a backend applied (or, for control, did not apply) an intervention.

    ``readback`` is what the simulator reported after the write; ``verified``
    is derived from it and the tolerance, never asserted by the caller.
    """

    intervention: Intervention
    requested: Mapping[str, float]
    readback: Mapping[str, float] | None
    method: str
    tolerance: float = 1e-6
    unsupported_reason: str | None = None
    env_id: int | None = None
    applied_at_step: int | None = None
    baseline: Mapping[str, float] = field(default_factory=dict)
    verified: bool = False

    def __post_init__(self):
        if not self.method:
            raise ValueError("a receipt must name the method it verified by")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        requested = {str(k): float(v) for k, v in self.requested.items()}
        if set(requested) != {k for k, _ in self.intervention.parameters}:
            raise ValueError("requested parameters must equal the intervention's parameters")
        object.__setattr__(self, "requested", requested)
        verified = False
        if self.unsupported_reason is None and self.readback is not None:
            readback = {str(k): float(v) for k, v in self.readback.items()}
            object.__setattr__(self, "readback", readback)
            verified = set(readback) >= set(requested) and all(
                math.isfinite(readback[k]) and abs(readback[k] - v) <= self.tolerance
                for k, v in requested.items()
            )
        object.__setattr__(self, "verified", bool(verified))

    @classmethod
    def unsupported(cls, intervention: Intervention, reason: str) -> "InterventionReceipt":
        return cls(
            intervention,
            dict(intervention.parameters),
            None,
            "unsupported",
            unsupported_reason=reason,
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["intervention"] = self.intervention.to_dict()
        return data


def gate_d1(
    treatment_receipt: InterventionReceipt,
    control_receipt: InterventionReceipt | None,
    *,
    expected: Intervention,
) -> GateResult:
    """D1: the requested intervention was applied to the treatment arm and not to the control."""
    detail: dict[str, Any] = {
        "treatment_receipt": treatment_receipt.to_dict(),
        "control_receipt": None if control_receipt is None else control_receipt.to_dict(),
    }
    if treatment_receipt.unsupported_reason:
        detail["unsupported"] = True
        detail["reason"] = treatment_receipt.unsupported_reason
        return GateResult("D1_intervention", False, detail)
    if treatment_receipt.intervention != expected:
        detail["reason"] = "receipt is for a different intervention than the one requested"
        return GateResult("D1_intervention", False, detail)
    if expected.is_control:
        detail["reason"] = "a control intervention cannot be the treatment"
        return GateResult("D1_intervention", False, detail)
    if not treatment_receipt.verified:
        detail["reason"] = "readback did not confirm the requested parameters"
        return GateResult("D1_intervention", False, detail)
    if control_receipt is None:
        detail["reason"] = "no control receipt; the counterfactual arm was not verified untouched"
        return GateResult("D1_intervention", False, detail)
    if not control_receipt.intervention.is_control or not control_receipt.verified:
        detail["reason"] = "control arm readback did not confirm the nominal parameters"
        return GateResult("D1_intervention", False, detail)
    return GateResult("D1_intervention", True, detail)


# --------------------------------------------------------------------------- #
# D2
# --------------------------------------------------------------------------- #


def gate_d2(
    treatment: RolloutTrace,
    control: RolloutTrace,
    nominal: Sequence[RolloutTrace],
    *,
    config: DepartureConfig | None = None,
    phenotype: str | None = None,
) -> tuple[GateResult, DepartureResult]:
    result = measure_departure(treatment, control, nominal, config=config, phenotype=phenotype)
    detail = result.to_dict()
    detail["precursor_step"] = departure_onset(result)
    return GateResult("D2_departure", result.passed, detail), result


# --------------------------------------------------------------------------- #
# D3
# --------------------------------------------------------------------------- #


def detect_phenotypes(
    trace: RolloutTrace, *, thresholds: FailureThresholds | None = None
) -> tuple[PhenotypeObservation, ...]:
    """Run the threshold detector over a trace and return windowed observations.

    A threshold detector knows only the frame at which its criterion was first
    met, so ``transition_start == established_start`` and no precursor is
    claimed. Contact forces are passed when the trace carries them; a trace
    without them cannot support stumble or contact-loss detection, and those
    phenotypes are simply not emitted rather than reported negative.
    """
    detector = FailureDetector(thresholds)
    first: dict[str, int] = {}
    for step in range(trace.n_steps):
        roll, pitch = quat_xyzw_to_roll_pitch(trace.base_quat[step])
        contact = None if trace.contact_forces is None else trace.contact_forces[step]
        events = detector.step(
            timestamp_s=step * trace.control_dt,
            pitch_rad=pitch,
            roll_rad=roll,
            base_height_m=float(trace.base_pos[step, 2]),
            cmd_lin_vel=trace.command_vel[step, :2],
            actual_lin_vel=trace.base_lin_vel_body[step, :2],
            joint_vel=trace.joint_vel[step],
            contact_forces=contact,
        )
        for event in events:
            first.setdefault(event.mode.value, step)
    return tuple(
        PhenotypeObservation(
            name,
            OnsetWindow(step, step, None, None, "detector"),
            {"detector": DETECTOR_VERSION, "frame": step},
        )
        for name, step in sorted(first.items())
    )


def gate_d3(
    treatment: RolloutTrace,
    control: RolloutTrace,
    *,
    intended_phenotype: str,
    platform: str = "simulation",
    thresholds: FailureThresholds | None = None,
) -> GateResult:
    spec = PHENOTYPES[intended_phenotype]
    if not spec.supported_on(platform):
        return GateResult(
            "D3_phenotype",
            False,
            {
                "unsupported": True,
                "reason": spec.unsupported_reason(platform),
                "phenotype": intended_phenotype,
            },
        )
    observed_t = detect_phenotypes(treatment, thresholds=thresholds)
    observed_c = detect_phenotypes(control, thresholds=thresholds)
    names_t = tuple(o.phenotype for o in observed_t)
    names_c = tuple(o.phenotype for o in observed_c)
    detail: dict[str, Any] = {
        "phenotype": intended_phenotype,
        "detector": DETECTOR_VERSION,
        "observed_phenotypes": list(names_t),
        "control_phenotypes": list(names_c),
        "treatment_observations": [o.to_dict() for o in observed_t],
        "control_observations": [o.to_dict() for o in observed_c],
    }
    in_treatment = intended_phenotype in names_t
    in_control = intended_phenotype in names_c
    detail["phenotype_in_treatment"] = in_treatment
    detail["phenotype_in_control"] = in_control
    if not in_treatment:
        detail["reason"] = "intended phenotype not detected in the treatment arm"
        return GateResult("D3_phenotype", False, detail)
    if in_control:
        detail["reason"] = "intended phenotype also occurs in the matched control; not attributable"
        return GateResult("D3_phenotype", False, detail)
    return GateResult("D3_phenotype", True, detail)


# --------------------------------------------------------------------------- #
# Combined
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MatchedPairEvidence:
    """Everything a backend returns for one matched pair."""

    treatment: RolloutTrace
    control: RolloutTrace
    nominal_replicates: tuple[RolloutTrace, ...]
    treatment_receipt: InterventionReceipt
    control_receipt: InterventionReceipt | None
    simulator_seed: int
    replicate_seeds: tuple[int, ...]

    def __post_init__(self):
        if type(self.simulator_seed) is not int or self.simulator_seed < 0:
            raise ValueError("simulator_seed must be a nonnegative integer")
        object.__setattr__(self, "nominal_replicates", tuple(self.nominal_replicates))
        object.__setattr__(self, "replicate_seeds", tuple(int(s) for s in self.replicate_seeds))
        if len(self.replicate_seeds) != len(self.nominal_replicates):
            raise ValueError("one seed per nominal replicate")
        if len(set(self.replicate_seeds)) != len(self.replicate_seeds):
            raise ValueError("replicate seeds must be distinct")
        if self.simulator_seed in self.replicate_seeds:
            raise ValueError("a nominal replicate must not reuse the matched pair's seed")
        if not math.isclose(self.treatment.control_dt, self.control.control_dt):
            raise ValueError("treatment and control control periods differ")
        # Matched pairs share the command. The one exception is by construction:
        # a command-corruption intervention changes the applied command, and the
        # control keeps the reference command.
        if self.treatment_receipt.intervention.kind != "command_corruption" and not np.array_equal(
            self.treatment.command_vel[0], self.control.command_vel[0]
        ):
            raise ValueError("treatment and control start from different commands")


def evaluate_delivery(
    evidence: MatchedPairEvidence,
    *,
    intervention: Intervention,
    intended_phenotype: str,
    departure: DepartureConfig | None = None,
    platform: str = "simulation",
    thresholds: FailureThresholds | None = None,
) -> DeliveryVerdict:
    """Run D1, D2 and D3 on one matched pair and return the derived verdict.

    All three gates are evaluated even when an earlier one fails, so the
    verdict carries the complete picture; the status names the first gate
    that failed.
    """
    assert_not_conflated(intervention.kind, intended_phenotype)
    d1 = gate_d1(evidence.treatment_receipt, evidence.control_receipt, expected=intervention)
    try:
        d2, _ = gate_d2(
            evidence.treatment,
            evidence.control,
            evidence.nominal_replicates,
            config=departure,
            phenotype=intended_phenotype,
        )
    except ValueError as exc:
        d2 = GateResult("D2_departure", False, {"reason": str(exc)})
    d3 = gate_d3(
        evidence.treatment,
        evidence.control,
        intended_phenotype=intended_phenotype,
        platform=platform,
        thresholds=thresholds,
    )
    return DeliveryVerdict(intended_phenotype, intervention, d1, d2, d3)


__all__ = [
    "DETECTOR_VERSION",
    "InterventionReceipt",
    "MatchedPairEvidence",
    "detect_phenotypes",
    "evaluate_delivery",
    "gate_d1",
    "gate_d2",
    "gate_d3",
]
