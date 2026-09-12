"""Preregistered experimental protocol: arms, compute budgets and hypothesis gates.

The legacy v2 five-arm declaration lives in :mod:`ashfall.evaluation.protocol`
and is left as it was. This package is the Phase-II protocol: four research
arms under one enforced compute budget (:mod:`ashfall.protocol.arms`,
:mod:`ashfall.protocol.budget`) and the gated hypothesis hierarchy
H0 to H4 (:mod:`ashfall.protocol.hypotheses`).
"""

from ashfall.protocol.arms import ARM_KINDS, ArmSet, ArmSpec, SamplerConfig, default_arm_set
from ashfall.protocol.budget import (
    BUDGET_FIELDS,
    ComplianceReport,
    ComputeBudget,
    ProtocolViolation,
    Tolerance,
    TrainingArtifact,
    Violation,
    artifact_from_rsl_rl,
    assert_equal_compute,
    compare_arms,
)
from ashfall.protocol.hypotheses import (
    HYPOTHESES,
    VERDICTS,
    HypothesisLedger,
    HypothesisSpec,
    VerdictRecord,
)

__all__ = [
    "ARM_KINDS",
    "BUDGET_FIELDS",
    "HYPOTHESES",
    "VERDICTS",
    "ArmSet",
    "ArmSpec",
    "ComplianceReport",
    "ComputeBudget",
    "HypothesisLedger",
    "HypothesisSpec",
    "ProtocolViolation",
    "SamplerConfig",
    "Tolerance",
    "TrainingArtifact",
    "VerdictRecord",
    "Violation",
    "artifact_from_rsl_rl",
    "assert_equal_compute",
    "compare_arms",
    "default_arm_set",
]
