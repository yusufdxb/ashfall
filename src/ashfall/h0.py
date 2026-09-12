"""H0, causal delivery: does an intervention induce its phenotype more often than its
matched no-intervention counterfactual?

The unit is a matched pair. Across ``n`` pairs the treatment arm and the
control arm each either show the intended phenotype or do not, so the data are
paired binary outcomes and the exact test is McNemar's on the discordant pairs
(binomial with p = 1/2 on the count of treatment-only pairs among all
discordant pairs). The gates D1 to D3 are evaluated on every pair and their
pass counts are reported; a pair whose intervention was not verifiably applied
makes the whole calibration UNINFORMATIVE, because causation cannot be tested
when the cause did not happen.

Everything about the acceptance rule is in :class:`H0Spec`, which is content
addressed and written into the evidence bundle before any pair is run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

from scipy.stats import binomtest

from ashfall.counterfactual import DepartureConfig, RestorableState, RolloutTrace
from ashfall.gates import MatchedPairEvidence, evaluate_delivery
from ashfall.ontology import (
    FailureEpisode,
    Intervention,
    ReproductionAttempt,
    assert_not_conflated,
    phenotype_windows_from_events,
)
from ashfall.provenance import content_hash

H0_VERDICTS = ("PASS", "FAIL", "UNINFORMATIVE")


class MatchedPairBackend(Protocol):
    evidence_kind: str
    backend_id: str
    policy_id: str

    def matched_pair(
        self,
        state: RestorableState,
        intervention: Intervention,
        *,
        simulator_seed: int,
        replicate_seeds: Sequence[int],
        horizon_steps: int,
    ) -> MatchedPairEvidence: ...

    def provenance(self) -> dict: ...


@dataclass(frozen=True)
class H0Spec:
    """Preregistered H0 acceptance rule for one intervention -> phenotype pathway."""

    intervention: Intervention
    intended_phenotype: str
    n_pairs: int
    replicates_per_pair: int
    horizon_steps: int
    alpha: float = 0.05
    min_delivered_fraction: float = 0.5
    departure: DepartureConfig = DepartureConfig()
    platform: str = "simulation"
    seed_base: int = 10_000
    state_selection: str = "nominal_rollout_rows"

    def __post_init__(self):
        assert_not_conflated(self.intervention.kind, self.intended_phenotype)
        if self.n_pairs < 1 or self.replicates_per_pair < 2 or self.horizon_steps < 2:
            raise ValueError("n_pairs, replicates_per_pair and horizon_steps must be positive")
        if not 0 < self.alpha < 1 or not 0 < self.min_delivered_fraction <= 1:
            raise ValueError("alpha in (0,1) and min_delivered_fraction in (0,1] required")
        if self.state_selection not in ("nominal_rollout_rows", "provided_states"):
            raise ValueError("unknown state_selection")

    @property
    def spec_id(self) -> str:
        return "h0_" + content_hash(self.to_dict(include_id=False))

    def to_dict(self, include_id: bool = True) -> dict:
        data = asdict(self)
        data["intervention"] = self.intervention.to_dict()
        if include_id:
            data["spec_id"] = self.spec_id
        return data

    def seeds_for_pair(self, index: int) -> tuple[int, tuple[int, ...]]:
        """Deterministic, non-overlapping seeds: pair seed then its replicate seeds."""
        block = self.seed_base + index * (self.replicates_per_pair + 1)
        return block, tuple(block + 1 + k for k in range(self.replicates_per_pair))


def exact_mcnemar_p(treatment_only: int, control_only: int) -> float:
    """Two-sided exact McNemar p on the discordant pairs; 1.0 when none are discordant."""
    if treatment_only < 0 or control_only < 0:
        raise ValueError("counts must be nonnegative")
    n = treatment_only + control_only
    if n == 0:
        return 1.0
    return float(binomtest(treatment_only, n, 0.5, alternative="two-sided").pvalue)


@dataclass(frozen=True)
class H0Result:
    spec_id: str
    n_pairs: int
    statuses: Mapping[str, int]
    d1_pass: int
    d2_pass: int
    d3_pass: int
    delivered: int
    delivered_fraction: float
    treatment_incidence: float
    control_incidence: float
    treatment_only: int
    control_only: int
    exact_p: float
    p_floor: float
    verdict: str
    reasons: tuple[str, ...]
    attempt_ids: tuple[str, ...]
    evidence_kind: str
    backend_id: str

    def __post_init__(self):
        if self.verdict not in H0_VERDICTS:
            raise ValueError(f"unknown verdict {self.verdict!r}")

    def to_dict(self) -> dict:
        return asdict(self)


def summarise_attempts(spec: H0Spec, attempts: Sequence[ReproductionAttempt]) -> H0Result:
    """Apply the preregistered rule to a set of attempts. Pure; no simulator."""
    if not attempts:
        raise ValueError("H0 needs at least one attempt")
    if any(a.intended_phenotype != spec.intended_phenotype for a in attempts):
        raise ValueError("attempt phenotype differs from the spec")
    if any(a.intervention != spec.intervention for a in attempts):
        raise ValueError("attempt intervention differs from the spec")
    kinds = {a.evidence_kind for a in attempts}
    backends = {a.backend_id for a in attempts}
    if len(kinds) != 1 or len(backends) != 1:
        raise ValueError("attempts mix evidence kinds or backends")
    statuses: dict[str, int] = {}
    d1 = d2 = d3 = delivered = 0
    treatment_hits = control_hits = treatment_only = control_only = 0
    for attempt in attempts:
        pair_verdict = attempt.verdict
        statuses[pair_verdict.status] = statuses.get(pair_verdict.status, 0) + 1
        d1 += pair_verdict.d1_intervention.passed
        d2 += pair_verdict.d2_departure.passed
        d3 += pair_verdict.d3_phenotype.passed
        delivered += pair_verdict.delivered
        detail = pair_verdict.d3_phenotype.detail
        in_t = bool(detail.get("phenotype_in_treatment", False))
        in_c = bool(detail.get("phenotype_in_control", False))
        treatment_hits += in_t
        control_hits += in_c
        treatment_only += in_t and not in_c
        control_only += in_c and not in_t
    n = len(attempts)
    p = exact_mcnemar_p(treatment_only, control_only)
    # With every pair discordant in the treatment direction the exact p is
    # 2 / 2**n; fewer than six pairs cannot reach alpha = 0.05 at all.
    p_floor = 2.0 / (2**n)
    fraction = delivered / n
    reasons = []
    if n < spec.n_pairs:
        reasons.append(f"only {n} of {spec.n_pairs} preregistered pairs were run")
    if statuses.get("UNSUPPORTED"):
        reasons.append("at least one pair reported an unsupported gate")
    if d1 < n:
        reasons.append(f"intervention application was not verified in {n - d1} of {n} pairs")
    outcome: str
    if reasons:
        outcome = "UNINFORMATIVE"
    else:
        if fraction < spec.min_delivered_fraction:
            reasons.append(
                f"delivered fraction {fraction:.3f} below preregistered "
                f"{spec.min_delivered_fraction}"
            )
        if p > spec.alpha:
            reasons.append(f"exact McNemar p {p:.4f} exceeds alpha {spec.alpha}")
        if p_floor > spec.alpha:
            reasons.append(
                f"{n} pairs cannot reach alpha {spec.alpha}: " f"the exact p floor is {p_floor:.4f}"
            )
        if treatment_only <= control_only:
            reasons.append("treatment did not show the phenotype more often than control")
        outcome = "FAIL" if reasons else "PASS"
    return H0Result(
        spec_id=spec.spec_id,
        n_pairs=n,
        statuses=dict(sorted(statuses.items())),
        d1_pass=d1,
        d2_pass=d2,
        d3_pass=d3,
        delivered=delivered,
        delivered_fraction=fraction,
        treatment_incidence=treatment_hits / n,
        control_incidence=control_hits / n,
        treatment_only=treatment_only,
        control_only=control_only,
        exact_p=p,
        p_floor=p_floor,
        verdict=outcome,
        reasons=tuple(reasons),
        attempt_ids=tuple(a.attempt_id for a in attempts),
        evidence_kind=next(iter(kinds)),
        backend_id=next(iter(backends)),
    )


def _episode(
    trace: RolloutTrace,
    *,
    source: str,
    policy_id: str,
    intervention: Intervention,
    state: RestorableState,
    seed: int,
    provenance: Mapping[str, Any],
    observations,
) -> FailureEpisode:
    return FailureEpisode(
        source=source,
        policy_id=policy_id,
        intervention=intervention,
        command=(
            float(trace.command_vel[0][0]),
            float(trace.command_vel[0][1]),
            float(trace.command_vel[0][2]),
        ),
        initial_state_id=state.state_id,
        control_dt=trace.control_dt,
        n_frames=trace.n_steps,
        termination_reason=trace.termination_reason,
        phenotypes=tuple(observations),
        provenance={**provenance, "seed": seed, "trace_hash": trace.content_hash()},
        seed=seed,
    )


@dataclass(frozen=True)
class PairRecord:
    """One matched pair's full evidence: attempt, both episodes, raw traces."""

    attempt: ReproductionAttempt
    control_episode: FailureEpisode
    treatment_episode: FailureEpisode
    evidence: MatchedPairEvidence
    state: RestorableState


def select_states(
    backend, spec: H0Spec, *, command: Sequence[float], rows: Sequence[int] | None = None
) -> list[RestorableState]:
    """Initial states from the policy's own nominal rollout, one per pair.

    Rows are spread across the middle of a nominal rollout so the states are
    ones the policy actually visits, not the spawn pose, and are recorded by
    content hash on every episode.
    """
    trace = backend.nominal_rollout(
        seed=spec.seed_base - 1, horizon_steps=spec.horizon_steps, command=command
    )
    if rows is None:
        first, last = trace.n_steps // 4, max(trace.n_steps // 4 + 1, (3 * trace.n_steps) // 4)
        rows = [first + (k * (last - first)) // max(spec.n_pairs, 1) for k in range(spec.n_pairs)]
    return [trace.state_at(int(r)) for r in rows]


def run_h0(
    backend: MatchedPairBackend,
    spec: H0Spec,
    *,
    states: Sequence[RestorableState],
    provenance: Mapping[str, Any] | None = None,
) -> tuple[H0Result, list[PairRecord]]:
    """Run every preregistered matched pair against ``backend`` and apply the rule."""
    if len(states) != spec.n_pairs:
        raise ValueError(f"spec asks for {spec.n_pairs} pairs; {len(states)} states were given")
    source = "simulation" if backend.evidence_kind == "simulation" else "synthetic_fixture"
    base_provenance = {**backend.provenance(), **(provenance or {})}
    records: list[PairRecord] = []
    for index, state in enumerate(states):
        pair_seed, replicate_seeds = spec.seeds_for_pair(index)
        evidence = backend.matched_pair(
            state,
            spec.intervention,
            simulator_seed=pair_seed,
            replicate_seeds=replicate_seeds,
            horizon_steps=spec.horizon_steps,
        )
        verdict = evaluate_delivery(
            evidence,
            intervention=spec.intervention,
            intended_phenotype=spec.intended_phenotype,
            departure=spec.departure,
            platform=spec.platform,
        )
        d3 = verdict.d3_phenotype.detail
        control_obs = _observations(d3.get("control_observations"), evidence.control)
        treatment_obs = _observations(d3.get("treatment_observations"), evidence.treatment)
        control_episode = _episode(
            evidence.control,
            source=source,
            policy_id=backend.policy_id,
            intervention=Intervention.none(),
            state=state,
            seed=pair_seed,
            provenance=base_provenance,
            observations=control_obs,
        )
        treatment_episode = _episode(
            evidence.treatment,
            source=source,
            policy_id=backend.policy_id,
            intervention=spec.intervention,
            state=state,
            seed=pair_seed,
            provenance=base_provenance,
            observations=treatment_obs,
        )
        attempt = ReproductionAttempt(
            initial_state_id=state.state_id,
            policy_id=backend.policy_id,
            intervention=spec.intervention,
            intended_phenotype=spec.intended_phenotype,
            simulator_seed=pair_seed,
            control_episode_id=control_episode.episode_id,
            treatment_episode_id=treatment_episode.episode_id,
            backend_id=backend.backend_id,
            evidence_kind=backend.evidence_kind,
            verdict=verdict,
        )
        records.append(PairRecord(attempt, control_episode, treatment_episode, evidence, state))
    result = summarise_attempts(spec, [r.attempt for r in records])
    return result, records


def _observations(serialised, trace: RolloutTrace):
    from ashfall.ontology import PhenotypeObservation

    if serialised is None:
        return ()
    return tuple(PhenotypeObservation.from_dict(o) for o in serialised)


def h0_seed_plan(spec: H0Spec) -> list[dict]:
    """The seeds every pair will use, so they can be written before the run."""
    return [
        {"pair": i, "simulator_seed": s, "replicate_seeds": list(r)}
        for i, (s, r) in ((i, spec.seeds_for_pair(i)) for i in range(spec.n_pairs))
    ]


__all__ = [
    "H0_VERDICTS",
    "H0Result",
    "H0Spec",
    "MatchedPairBackend",
    "PairRecord",
    "exact_mcnemar_p",
    "h0_seed_plan",
    "phenotype_windows_from_events",
    "run_h0",
    "select_states",
    "summarise_attempts",
]
