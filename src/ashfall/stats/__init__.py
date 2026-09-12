"""Confirmatory statistics: one primary endpoint, declared secondaries, exact tests.

The experimental unit throughout is the independently trained policy, indexed
by its training seed. Evaluation episodes are measurements of one policy, not
replicates of the experiment, and no function here will count them as such.
"""

from ashfall.stats.primary import (
    PairedSeedEffect,
    PrimaryEndpointResult,
    SecondaryEndpointsResult,
    SeedIncidence,
    exact_sign_flip_p,
    incidence_by_training_seed,
    paired_seed_effect,
    primary_endpoint,
    secondary_endpoints,
    sign_flip_p_floor,
)
from ashfall.stats.reference import reference_two_sample_bca_acceleration

__all__ = [
    "PairedSeedEffect",
    "PrimaryEndpointResult",
    "SecondaryEndpointsResult",
    "SeedIncidence",
    "exact_sign_flip_p",
    "incidence_by_training_seed",
    "paired_seed_effect",
    "primary_endpoint",
    "reference_two_sample_bca_acceleration",
    "secondary_endpoints",
    "sign_flip_p_floor",
]
