"""Mutation sensitivity: can the evaluation tell a broken detector from the real one?

An evaluation that reports the same numbers for a detector and for a detector
that always says "collapse" is not measuring anything. Before any accuracy
figure from :func:`ashfall.detector_eval.metrics.evaluate_detector` may be
quoted, the suite here must show that each of five deliberate defects moves
the report by at least the preregistered margin.

Preregistered catch criterion, per mutant, against the unmutated baseline on
the same dataset:

* ``always_collapse``, ``always_stumble``, ``always_command_mismatch``,
  ``no_slip_priority``: for the affected phenotype, F1 falls by at least
  :data:`F1_DROP` absolute, OR precision falls by at least :data:`F1_DROP`,
  OR false-positive events per scored episode rise by at least
  :data:`FP_RISE`.
* ``label_swap(a, b)``: the exclusive confusion matrix moves mass off the
  diagonal for the pair (the ``(a, b)`` or ``(b, a)`` cell increases), OR F1
  of either phenotype falls by at least :data:`F1_DROP`.

A mutant that is not caught is reported as such. The suite never hides it,
and the guard tests fail on it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

from ashfall.detector_eval.dataset import DetectorEvalDataset
from ashfall.detector_eval.metrics import DetectorEvaluationReport, evaluate_detector
from ashfall.provenance import content_hash
from ashfall.taxonomy.detector import FailureThresholds
from ashfall.taxonomy.phenotype_detector import DetectorMutation, PhenotypeDetector

#: Minimum absolute F1 (or precision) drop that counts as caught.
F1_DROP = 0.2
#: Minimum rise in false-positive events per scored episode that counts as caught.
FP_RISE = 0.25

#: The five defects every evaluation must be able to see.
STANDARD_MUTATIONS: tuple[DetectorMutation, ...] = (
    DetectorMutation("always_collapse"),
    DetectorMutation("always_stumble"),
    DetectorMutation("always_command_mismatch"),
    DetectorMutation("no_slip_priority"),
    DetectorMutation("label_swap", ("slip", "collapse")),
)


@dataclass(frozen=True)
class MutantOutcome:
    mutation: str
    affected_phenotypes: tuple[str, ...]
    caught: bool
    criteria_met: tuple[str, ...]
    baseline_f1: dict
    mutant_f1: dict
    baseline_fp_per_episode: dict
    mutant_fp_per_episode: dict
    confusion_delta: dict


@dataclass(frozen=True)
class MutationReport:
    dataset_id: str
    report_kind: str
    baseline_detector: str
    outcomes: tuple[MutantOutcome, ...]
    f1_drop: float = F1_DROP
    fp_rise: float = FP_RISE

    @property
    def all_caught(self) -> bool:
        return all(o.caught for o in self.outcomes)

    @property
    def missed(self) -> tuple[str, ...]:
        return tuple(o.mutation for o in self.outcomes if not o.caught)

    @property
    def report_id(self) -> str:
        return "mrep_" + content_hash(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "report_kind": self.report_kind,
            "baseline_detector": self.baseline_detector,
            "criterion": {"f1_drop": self.f1_drop, "fp_rise": self.fp_rise},
            "all_caught": self.all_caught,
            "missed": list(self.missed),
            "outcomes": [asdict(o) for o in self.outcomes],
        }


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else b - a


def judge(
    baseline: DetectorEvaluationReport, mutant: DetectorEvaluationReport, mutation: DetectorMutation
) -> MutantOutcome:
    affected = mutation.affected_phenotypes
    met: list[str] = []
    for name in affected:
        b, m = baseline.per_phenotype[name], mutant.per_phenotype[name]
        f1_change = _delta(b.f1, m.f1)
        if f1_change is not None and f1_change <= -F1_DROP + 1e-9:
            met.append(f"{name}: F1 fell by {-f1_change:.3f}")
        elif b.f1 is not None and m.f1 is None:
            met.append(f"{name}: F1 became undefined")
        precision_change = _delta(b.precision, m.precision)
        if precision_change is not None and precision_change <= -F1_DROP + 1e-9:
            met.append(f"{name}: precision fell by {-precision_change:.3f}")
        fp_change = _delta(b.false_positive_events_per_episode, m.false_positive_events_per_episode)
        if fp_change is not None and fp_change >= FP_RISE - 1e-9:
            met.append(f"{name}: false-positive events per episode rose by {fp_change:.3f}")
    confusion_delta: dict = {}
    if mutation.kind == "label_swap":
        a, b_name = mutation.swap  # type: ignore[misc]
        for row, column in ((a, b_name), (b_name, a)):
            before = baseline.confusion.cell(row, column)
            after = mutant.confusion.cell(row, column)
            confusion_delta[f"{row}->{column}"] = after - before
            if after > before:
                met.append(f"confusion {row}->{column} rose from {before} to {after}")
    return MutantOutcome(
        mutation=mutation.label,
        affected_phenotypes=affected,
        caught=bool(met),
        criteria_met=tuple(met),
        baseline_f1={n: baseline.per_phenotype[n].f1 for n in affected},
        mutant_f1={n: mutant.per_phenotype[n].f1 for n in affected},
        baseline_fp_per_episode={
            n: baseline.per_phenotype[n].false_positive_events_per_episode for n in affected
        },
        mutant_fp_per_episode={
            n: mutant.per_phenotype[n].false_positive_events_per_episode for n in affected
        },
        confusion_delta=confusion_delta,
    )


def run_mutation_suite(
    dataset: DetectorEvalDataset,
    base_factory: Callable[[], PhenotypeDetector] | None = None,
    mutations: Sequence[DetectorMutation] = STANDARD_MUTATIONS,
    thresholds: FailureThresholds | None = None,
) -> MutationReport:
    """Evaluate the baseline and every mutant; report which defects the evaluation caught."""
    factory = base_factory or (lambda: PhenotypeDetector(thresholds))
    baseline = evaluate_detector(dataset, factory)
    outcomes = []
    for mutation in mutations:
        if mutation.is_identity:
            raise ValueError("the identity mutation is the baseline, not a mutant")
        mutant_report = evaluate_detector(
            dataset, lambda m=mutation: PhenotypeDetector(thresholds, mutation=m)
        )
        outcomes.append(judge(baseline, mutant_report, mutation))
    return MutationReport(
        dataset.dataset_id, dataset.report_kind, baseline.detector, tuple(outcomes)
    )


__all__ = [
    "F1_DROP",
    "FP_RISE",
    "STANDARD_MUTATIONS",
    "MutantOutcome",
    "MutationReport",
    "judge",
    "run_mutation_suite",
]
