"""One confirmatory primary endpoint, declared secondaries, and the paired-seed test.

Primary endpoint. Held-out failure incidence aggregated over the preregistered
validated phenotype set (:data:`ashfall.ontology.FIRST_STUDY_PHENOTYPES` unless
the preregistration names another). An episode counts as a failure when any
phenotype in the set was observed in it. Incidence is computed per arm and per
TRAINING SEED; the paired difference across seeds is the estimate, and the
exact sign-flip permutation test on those paired deltas is the confirmatory
test.

Experimental unit. The independently trained policy, indexed by training
seed. Evaluation episodes and evaluation seeds are measurements of one policy.
Asking this module to treat them as replicates raises.

Secondaries. Per-phenotype incidence deltas, one per phenotype in a declared
family, with Holm step-down correction over that family. They are descriptive
of where an effect lives; they do not confirm anything on their own.

Assumptions stated per function. The sign-flip test assumes exchangeability of
the sign of each paired delta under the null; the t interval assumes
approximately normal paired deltas, which at small n is a reporting
convention and is labelled so.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
from scipy.stats import t as student_t

from ashfall.evaluation.significance import holm_adjust
from ashfall.ontology import FIRST_STUDY_PHENOTYPES, PHENOTYPE_NAMES

REPLICATE_UNIT = "training_seed"
REJECTED_UNITS = ("evaluation_episode", "evaluation_seed", "episode", "scenario", "replicate")


def _require_unit(replicate_unit: str) -> None:
    if replicate_unit != REPLICATE_UNIT:
        raise ValueError(
            f"replicate_unit={replicate_unit!r} is not a valid experimental unit: evaluation "
            "episodes are measurements of one trained policy, not independent replicates of "
            f"training. Only {REPLICATE_UNIT!r} is accepted."
        )


@dataclass(frozen=True)
class SeedIncidence:
    """Incidence of the phenotype set for one arm and one training seed."""

    arm: str
    training_seed: int
    policy_id: str
    episodes: int
    failures: int

    @property
    def incidence(self) -> float:
        return self.failures / self.episodes

    def to_dict(self) -> dict:
        return {**asdict(self), "incidence": self.incidence}


def incidence_by_training_seed(
    records: Iterable[Mapping],
    phenotypes: Sequence[str] = FIRST_STUDY_PHENOTYPES,
    *,
    arm: str,
    replicate_unit: str = REPLICATE_UNIT,
) -> dict[int, SeedIncidence]:
    """Reduce one arm's episodes to one incidence per training seed.

    Every record must carry ``training_seed`` (an integer), ``policy_id``,
    ``scenario_id``, ``evaluation_seed`` and ``failure_modes`` (a list; None
    means the failure capture is unknown and is refused, because an unknown
    capture is not an absence of failure). One policy per seed is required.
    """
    _require_unit(replicate_unit)
    phenotype_set = _validated_phenotypes(phenotypes)
    grouped: dict[int, list[Mapping]] = {}
    for record in records:
        seed = record.get("training_seed")
        if type(seed) is not int:
            raise ValueError(f"{arm}: every record needs an integer training_seed")
        for key in ("policy_id", "scenario_id", "evaluation_seed"):
            if record.get(key) in (None, ""):
                raise ValueError(f"{arm}: record lacks {key}")
        if record.get("failure_modes") is None:
            raise ValueError(
                f"{arm}: failure_modes is None for a record; unknown failure capture cannot be "
                "counted as an episode without failure"
            )
        grouped.setdefault(seed, []).append(record)
    if not grouped:
        raise ValueError(f"{arm}: no records")
    out: dict[int, SeedIncidence] = {}
    for seed, rows in sorted(grouped.items()):
        policies = {r["policy_id"] for r in rows}
        if len(policies) != 1:
            raise ValueError(f"{arm}: training seed {seed} mixes policies {sorted(policies)}")
        keys = [(r["scenario_id"], r["evaluation_seed"]) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"{arm}: duplicate (scenario, evaluation seed) at training seed {seed}"
            )
        failures = sum(1 for r in rows if set(r["failure_modes"]) & phenotype_set)
        out[seed] = SeedIncidence(arm, seed, next(iter(policies)), len(rows), failures)
    return out


def _validated_phenotypes(phenotypes: Sequence[str]) -> set[str]:
    names = list(phenotypes)
    if not names or len(set(names)) != len(names):
        raise ValueError("a nonempty set of distinct phenotypes is required")
    unknown = sorted(set(names) - set(PHENOTYPE_NAMES))
    if unknown:
        raise ValueError(f"unknown phenotypes {unknown}")
    return set(names)


def exact_sign_flip_p(deltas: Sequence[float]) -> float:
    """Two-sided exact sign-flip permutation p-value on ``|mean(deltas)|``.

    Enumerates all ``2**n`` sign assignments. Under the null hypothesis of no
    effect the sign of each paired delta is exchangeable, which is the only
    assumption. Ties at the observed statistic count as at least as extreme.
    An exactly-zero delta is invariant under its sign, so it never changes the
    statistic and doubles every count: see :func:`sign_flip_p_floor`.
    """
    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("deltas must be a nonempty finite one-dimensional sequence")
    observed = abs(float(values.mean()))
    if observed == 0.0:
        return 1.0
    hits = 0
    total = 0
    for signs in product((-1.0, 1.0), repeat=len(values)):
        total += 1
        if abs(float(np.mean(values * np.asarray(signs)))) >= observed - 1e-12:
            hits += 1
    return hits / total


def sign_flip_p_floor(deltas: Sequence[float]) -> float:
    """Smallest p the exact sign-flip test can return for these deltas.

    ``2 / 2**k`` where ``k`` is the number of sign-carrying (nonzero) deltas:
    the observed assignment and its mirror image always tie for most extreme.
    With no nonzero delta the test is uninformative and the floor is 1.
    """
    values = np.asarray(deltas, dtype=float)
    carrying = int(np.count_nonzero(values))
    return 1.0 if carrying == 0 else 2.0 / (2.0**carrying)


@dataclass(frozen=True)
class PairedSeedEffect:
    """Treatment minus baseline, paired by training seed."""

    metric: str
    seeds: tuple[int, ...]
    deltas: tuple[float, ...]
    mean: float
    sd: float
    sem: float
    ci_low: float
    ci_high: float
    confidence: float
    p_exact_two_sided: float
    p_floor: float
    sign_carrying_seeds: int
    inference_unit: str = REPLICATE_UNIT
    interval_method: str = "student_t_paired"

    def alpha_reachable(self, alpha: float = 0.05) -> bool:
        """Whether the design could have rejected at ``alpha`` at all."""
        return self.p_floor <= alpha

    def to_dict(self) -> dict:
        return asdict(self)


def paired_seed_effect(
    baseline_by_seed: Mapping[int, float],
    treatment_by_seed: Mapping[int, float],
    *,
    metric: str = "incidence",
    confidence: float = 0.95,
) -> PairedSeedEffect:
    """Paired deltas across training seeds with a t interval and the exact test.

    Seeds must match exactly between arms; an unpaired seed is an error, not
    a dropped row. Needs at least two paired seeds for a sample standard
    deviation. The Student-t interval assumes approximately normal deltas and
    is a reporting convention at small n; the exact p-value does not share
    that assumption.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")
    base_seeds, treat_seeds = set(baseline_by_seed), set(treatment_by_seed)
    if base_seeds != treat_seeds:
        raise ValueError(
            "unpaired training seeds: baseline-only "
            f"{sorted(base_seeds - treat_seeds)}, treatment-only {sorted(treat_seeds - base_seeds)}"
        )
    if any(type(seed) is not int for seed in base_seeds):
        raise ValueError("training seeds must be integers")
    seeds = tuple(sorted(base_seeds))
    if len(seeds) < 2:
        raise ValueError("paired-seed inference needs at least two paired training seeds")
    deltas = np.array([float(treatment_by_seed[s]) - float(baseline_by_seed[s]) for s in seeds])
    if not np.isfinite(deltas).all():
        raise ValueError("per-seed values must be finite")
    n = len(deltas)
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1))
    sem = sd / math.sqrt(n)
    half = float(student_t.ppf((1 + confidence) / 2, n - 1)) * sem
    return PairedSeedEffect(
        metric=metric,
        seeds=seeds,
        deltas=tuple(float(d) for d in deltas),
        mean=mean,
        sd=sd,
        sem=sem,
        ci_low=mean - half,
        ci_high=mean + half,
        confidence=confidence,
        p_exact_two_sided=exact_sign_flip_p(deltas),
        p_floor=sign_flip_p_floor(deltas),
        sign_carrying_seeds=int(np.count_nonzero(deltas)),
    )


@dataclass(frozen=True)
class PrimaryEndpointResult:
    endpoint: str
    phenotypes: tuple[str, ...]
    baseline: tuple[SeedIncidence, ...]
    treatment: tuple[SeedIncidence, ...]
    effect: PairedSeedEffect
    alpha: float
    inference_unit: str = REPLICATE_UNIT

    @property
    def n_seeds(self) -> int:
        return len(self.effect.seeds)

    @property
    def rejected_at_alpha(self) -> bool:
        return self.effect.p_exact_two_sided <= self.alpha

    @property
    def statement(self) -> str:
        effect = self.effect
        reachable = "reachable" if effect.alpha_reachable(self.alpha) else "NOT reachable"
        return (
            f"{self.endpoint} over {list(self.phenotypes)}: mean delta {effect.mean:+.4f} "
            f"({effect.confidence:.0%} t interval [{effect.ci_low:+.4f}, {effect.ci_high:+.4f}]), "
            f"exact two-sided sign-flip p={effect.p_exact_two_sided:.4f} over {self.n_seeds} "
            f"paired training seeds; achievable p-floor {effect.p_floor:.4f} from "
            f"{effect.sign_carrying_seeds} sign-carrying seeds, alpha={self.alpha} {reachable}."
        )

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "phenotypes": list(self.phenotypes),
            "baseline": [s.to_dict() for s in self.baseline],
            "treatment": [s.to_dict() for s in self.treatment],
            "effect": self.effect.to_dict(),
            "alpha": self.alpha,
            "inference_unit": self.inference_unit,
            "n_seeds": self.n_seeds,
            "rejected_at_alpha": self.rejected_at_alpha,
            "statement": self.statement,
        }


def primary_endpoint(
    baseline_records: Iterable[Mapping],
    treatment_records: Iterable[Mapping],
    phenotypes: Sequence[str] = FIRST_STUDY_PHENOTYPES,
    *,
    alpha: float = 0.05,
    confidence: float = 0.95,
    replicate_unit: str = REPLICATE_UNIT,
) -> PrimaryEndpointResult:
    """Held-out failure incidence over the preregistered phenotype set, paired by seed."""
    _require_unit(replicate_unit)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    base = incidence_by_training_seed(baseline_records, phenotypes, arm="baseline")
    treat = incidence_by_training_seed(treatment_records, phenotypes, arm="treatment")
    effect = paired_seed_effect(
        {s: v.incidence for s, v in base.items()},
        {s: v.incidence for s, v in treat.items()},
        metric="held_out_failure_incidence",
        confidence=confidence,
    )
    return PrimaryEndpointResult(
        "held_out_failure_incidence",
        tuple(phenotypes),
        tuple(base[s] for s in effect.seeds),
        tuple(treat[s] for s in effect.seeds),
        effect,
        alpha,
    )


@dataclass(frozen=True)
class SecondaryEndpointsResult:
    family: tuple[str, ...]
    effects: Mapping[str, PairedSeedEffect]
    holm_adjusted_p: Mapping[str, float]
    alpha: float
    correction: str = "holm_step_down"

    def to_dict(self) -> dict:
        return {
            "family": list(self.family),
            "family_size": len(self.family),
            "effects": {k: v.to_dict() for k, v in self.effects.items()},
            "holm_adjusted_p": dict(self.holm_adjusted_p),
            "alpha": self.alpha,
            "correction": self.correction,
        }


def secondary_endpoints(
    baseline_records: Iterable[Mapping],
    treatment_records: Iterable[Mapping],
    phenotypes: Sequence[str] = FIRST_STUDY_PHENOTYPES,
    *,
    alpha: float = 0.05,
    confidence: float = 0.95,
    replicate_unit: str = REPLICATE_UNIT,
) -> SecondaryEndpointsResult:
    """Per-phenotype paired incidence effects with Holm correction over the declared family.

    The family is exactly ``phenotypes``; it is recorded in the result so a
    reader can see what was corrected for. Holm's step-down is valid under
    arbitrary dependence between the per-phenotype tests.
    """
    _require_unit(replicate_unit)
    family = tuple(phenotypes)
    _validated_phenotypes(family)
    baseline_records, treatment_records = list(baseline_records), list(treatment_records)
    effects: dict[str, PairedSeedEffect] = {}
    for name in family:
        base = incidence_by_training_seed(baseline_records, (name,), arm="baseline")
        treat = incidence_by_training_seed(treatment_records, (name,), arm="treatment")
        effects[name] = paired_seed_effect(
            {s: v.incidence for s, v in base.items()},
            {s: v.incidence for s, v in treat.items()},
            metric=f"incidence[{name}]",
            confidence=confidence,
        )
    adjusted = holm_adjust([effects[name].p_exact_two_sided for name in family])
    return SecondaryEndpointsResult(family, effects, dict(zip(family, adjusted)), alpha)


__all__ = [
    "REPLICATE_UNIT",
    "PairedSeedEffect",
    "PrimaryEndpointResult",
    "SecondaryEndpointsResult",
    "SeedIncidence",
    "exact_sign_flip_p",
    "incidence_by_training_seed",
    "paired_seed_effect",
    "primary_endpoint",
    "secondary_endpoints",
    "sign_flip_p_floor",
]
