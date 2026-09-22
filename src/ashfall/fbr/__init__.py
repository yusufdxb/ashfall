"""Failure-Boundary Replay: the one method of the first Ashfall study.

REAL FAILURE -> capsule (``capsule``, ``criterion``) -> reconstruction accepted
(``acceptance``) -> boundary per seed state (``boundary``) -> local curriculum
(``BoundarySampler``) -> equal-budget fine-tune -> held-out evaluation.

``toy_slip`` and ``toy_study`` are the CPU mechanism study; nothing they
produce is a GO2 result.
"""

from ashfall.fbr.acceptance import AcceptanceConfig, ReconstructionVerdict, accept_reconstruction
from ashfall.fbr.boundary import (
    BoundaryEstimate,
    BoundaryNotIdentified,
    BoundarySampler,
    BroadSampler,
    LocalNeighborhood,
    ReplayDraw,
    boundary_mass,
    estimate_boundary,
)
from ashfall.fbr.capsule import SeedPoint, assert_near_onset, extract_capsule, seed_points
from ashfall.fbr.criterion import FailureCriterion, Onset, detect_onset

__all__ = [
    "AcceptanceConfig",
    "BoundaryEstimate",
    "BoundaryNotIdentified",
    "BoundarySampler",
    "BroadSampler",
    "FailureCriterion",
    "LocalNeighborhood",
    "Onset",
    "ReconstructionVerdict",
    "ReplayDraw",
    "SeedPoint",
    "accept_reconstruction",
    "assert_near_onset",
    "boundary_mass",
    "detect_onset",
    "estimate_boundary",
    "extract_capsule",
    "seed_points",
]
