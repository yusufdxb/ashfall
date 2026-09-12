"""Detector accuracy against independently labelled episodes.

Scoring unit and rules, stated once:

* **Episode-level one-vs-rest.** For each phenotype, an episode is a true
  positive when a ground-truth window of that phenotype exists and the
  detector emitted that phenotype at least once inside one of its windows; a
  false positive when the detector emitted the phenotype and no window of it
  contains any of those emissions; a false negative when a window exists and
  no emission lands inside any window; a true negative otherwise. An episode
  on which the phenotype is UNSUPPORTED (channel or platform) is not scored
  for that phenotype and is counted under ``unsupported``.
* **Window-level.** Every ground-truth window is matched to the earliest
  emission of its phenotype inside ``[transition_start, end]``. Recovery
  episodes therefore need two detections for two windows, and a
  multi-failure episode contributes to every phenotype it carries. Emissions
  that land in no window of their phenotype are false-positive events.
* **Onset timing.** For each matched window, ``(matched frame - transition_start) * dt``
  in seconds. Reported as median, 90th percentile, mean absolute, and the
  fraction of windows detected inside their span.
* **False-positive load.** False-positive events per episode and per minute
  of telemetry, over the whole dataset and per phenotype.
* **Exclusive confusion matrix.** Rows are the primary ground-truth phenotype
  (earliest transition) or ``none``; columns are the earliest supported
  emission or ``none``, plus ``unsupported`` when the primary truth phenotype
  cannot be scored on that episode. The matrix asks WHICH phenotype was named
  and ignores WHEN; an emission of the right phenotype outside its window is
  a false negative in the window metrics and a diagonal entry here.

Undefined ratios are ``None``, never 1.0. A report from a fixture dataset is
a ``fixture_regression`` and says so in every serialisation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ashfall.detector_eval.dataset import DetectorEvalDataset, LabeledEpisode
from ashfall.ontology import PHENOTYPE_NAMES, PhenotypeObservation
from ashfall.provenance import content_hash
from ashfall.taxonomy.phenotype_detector import PhenotypeDetector, supported_phenotypes

DetectorFactory = Callable[[], PhenotypeDetector]


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class OnsetTiming:
    windows: int
    matched: int
    fraction_inside_window: float | None
    median_error_s: float | None
    p90_error_s: float | None
    mean_abs_error_s: float | None

    @classmethod
    def from_errors(cls, errors: Sequence[float], windows: int) -> "OnsetTiming":
        if windows and not 0 <= len(errors) <= windows:
            raise ValueError("more matches than windows")
        if not errors:
            return cls(windows, 0, _ratio(0, windows), None, None, None)
        arr = np.asarray(errors, dtype=float)
        return cls(
            windows,
            len(errors),
            _ratio(len(errors), windows),
            float(np.median(arr)),
            float(np.percentile(arr, 90)),
            float(np.mean(np.abs(arr))),
        )


@dataclass(frozen=True)
class PhenotypeMetrics:
    phenotype: str
    scored_episodes: int
    unsupported_episodes: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float | None
    recall: float | None
    f1: float | None
    false_positive_events: int
    false_positive_events_per_episode: float | None
    onset_timing: OnsetTiming

    def __post_init__(self):
        total = self.true_positive + self.false_positive + self.true_negative + self.false_negative
        if total != self.scored_episodes:
            raise ValueError("confusion counts do not sum to the scored episodes")
        for value in (self.precision, self.recall, self.f1):
            if value is not None and not 0 <= value <= 1:
                raise ValueError("ratios must lie in [0, 1] or be None")


def one_vs_rest_counts(truth: Sequence[bool], predicted: Sequence[bool]) -> dict[str, int]:
    """Episode-level counts. Pure, so a property test can pin it."""
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must align")
    tp = fp = tn = fn = 0
    for t, p in zip(truth, predicted):
        if t and p:
            tp += 1
        elif not t and p:
            fp += 1
        elif t and not p:
            fn += 1
        else:
            tn += 1
    return {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn}


def match_windows(
    truth: Sequence[PhenotypeObservation],
    predicted: Sequence[PhenotypeObservation],
    *,
    n_steps: int,
    phenotype: str,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match one phenotype's emissions to its ground-truth windows.

    Returns ``(matches, unmatched_windows, false_positive_frames)`` where a
    match is ``(window_index, frame)`` using the earliest emission inside the
    window. An emission can satisfy one window only; windows are visited in
    transition order. Emissions that land inside a window of their phenotype
    are never false positives, matched or not: a detector that re-fires while
    a slip persists is repeating a true detection, not inventing one. Only
    emissions outside every window of the phenotype count as false positives.
    """
    windows = [(i, o.window) for i, o in enumerate(truth) if o.phenotype == phenotype]
    windows.sort(key=lambda item: (item[1].transition_start, item[0]))
    spans = [(w.transition_start, n_steps - 1 if w.end is None else w.end) for _, w in windows]
    frames = sorted(o.window.established_start for o in predicted if o.phenotype == phenotype)
    used: set[int] = set()
    matches: list[tuple[int, int]] = []
    unmatched: list[int] = []
    for (index, _), (start, end) in zip(windows, spans):
        hit = next((f for f in frames if f not in used and start <= f <= end), None)
        if hit is None:
            unmatched.append(index)
        else:
            used.add(hit)
            matches.append((index, hit))
    false_positives = [f for f in frames if not any(start <= f <= end for start, end in spans)]
    return matches, unmatched, false_positives


@dataclass(frozen=True)
class ConfusionMatrix:
    rows: tuple[str, ...]
    columns: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]

    def cell(self, row: str, column: str) -> int:
        return self.counts[self.rows.index(row)][self.columns.index(column)]

    def to_dict(self) -> dict:
        return {
            "rows": list(self.rows),
            "columns": list(self.columns),
            "counts": [list(r) for r in self.counts],
        }


@dataclass(frozen=True)
class DetectorEvaluationReport:
    dataset_id: str
    report_kind: str
    validation_claim_allowed: bool
    label_source: str
    platform: str
    detector: str
    n_episodes: int
    telemetry_minutes: float
    categories: Mapping[str, int]
    per_phenotype: Mapping[str, PhenotypeMetrics]
    confusion: ConfusionMatrix
    false_positive_events: int
    false_positives_per_episode: float
    false_positives_per_minute: float | None
    unsupported_phenotypes: Mapping[str, Mapping[str, Any]]
    predictions_on_unsupported: int
    schema_version: str = "1.0"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.report_kind not in ("fixture_regression", "independent_evaluation"):
            raise ValueError("unknown report kind")
        if self.validation_claim_allowed and self.report_kind != "independent_evaluation":
            raise ValueError("only an independent evaluation may support a validation claim")

    @property
    def report_id(self) -> str:
        return "drep_" + content_hash(self.to_dict())

    def f1(self, phenotype: str) -> float | None:
        return self.per_phenotype[phenotype].f1

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "report_kind": self.report_kind,
            "validation_claim_allowed": self.validation_claim_allowed,
            "label_source": self.label_source,
            "platform": self.platform,
            "detector": self.detector,
            "n_episodes": self.n_episodes,
            "telemetry_minutes": self.telemetry_minutes,
            "categories": dict(self.categories),
            "per_phenotype": {k: asdict(v) for k, v in self.per_phenotype.items()},
            "confusion": self.confusion.to_dict(),
            "false_positive_events": self.false_positive_events,
            "false_positives_per_episode": self.false_positives_per_episode,
            "false_positives_per_minute": self.false_positives_per_minute,
            "unsupported_phenotypes": {k: dict(v) for k, v in self.unsupported_phenotypes.items()},
            "predictions_on_unsupported": self.predictions_on_unsupported,
            "notes": list(self.notes),
        }


def _primary_prediction(predicted: Sequence[PhenotypeObservation], supported: set[str]) -> str:
    usable = [o for o in predicted if o.phenotype in supported]
    if not usable:
        return "none"
    return min(usable, key=lambda o: (o.window.established_start, o.phenotype)).phenotype


def evaluate_detector(
    dataset: DetectorEvalDataset, detector_factory: DetectorFactory | None = None
) -> DetectorEvaluationReport:
    """Score a detector on an independently labelled dataset. Fresh detector per episode."""
    factory = detector_factory or PhenotypeDetector
    probe = factory()
    per_episode: list[
        tuple[LabeledEpisode, tuple[PhenotypeObservation, ...], dict[str, str | None]]
    ] = []
    for episode in dataset.episodes:
        detector = factory()
        support = supported_phenotypes(episode.telemetry.channels, dataset.platform)
        predicted = detector.run(episode.telemetry.as_mapping(), episode.telemetry.dt_s)
        per_episode.append((episode, predicted, support))

    per_phenotype: dict[str, PhenotypeMetrics] = {}
    unsupported: dict[str, dict[str, Any]] = {}
    total_fp_events = 0
    predictions_on_unsupported = 0
    for name in PHENOTYPE_NAMES:
        truth_flags: list[bool] = []
        pred_flags: list[bool] = []
        errors: list[float] = []
        windows = 0
        fp_events = 0
        skipped = 0
        reasons: dict[str, int] = {}
        for episode, predicted, support in per_episode:
            reason = support[name]
            if reason is not None:
                skipped += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                predictions_on_unsupported += sum(1 for o in predicted if o.phenotype == name)
                continue
            matches, unmatched, false_positive_frames = match_windows(
                episode.truth, predicted, n_steps=episode.telemetry.n_steps, phenotype=name
            )
            n_windows = len(matches) + len(unmatched)
            windows += n_windows
            fp_events += len(false_positive_frames)
            has_truth = n_windows > 0
            emitted = any(o.phenotype == name for o in predicted)
            truth_flags.append(has_truth)
            # Predicted positive counts for the episode only when some emission
            # landed inside a window (TP) or when there is no window at all (FP).
            pred_flags.append(bool(matches) if has_truth else emitted)
            dt = episode.telemetry.dt_s
            for index, frame in matches:
                errors.append((frame - episode.truth[index].window.transition_start) * dt)
        counts = one_vs_rest_counts(truth_flags, pred_flags)
        scored = len(truth_flags)
        precision = _ratio(
            counts["true_positive"], counts["true_positive"] + counts["false_positive"]
        )
        recall = _ratio(counts["true_positive"], counts["true_positive"] + counts["false_negative"])
        per_phenotype[name] = PhenotypeMetrics(
            phenotype=name,
            scored_episodes=scored,
            unsupported_episodes=skipped,
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
            false_positive_events=fp_events,
            false_positive_events_per_episode=_ratio(fp_events, scored),
            onset_timing=OnsetTiming.from_errors(errors, windows),
            **counts,
        )
        total_fp_events += fp_events
        if skipped:
            unsupported[name] = {"episodes": skipped, "reasons": reasons}

    rows = (*PHENOTYPE_NAMES, "none")
    columns = (*PHENOTYPE_NAMES, "none", "unsupported")
    matrix = [[0] * len(columns) for _ in rows]
    for episode, predicted, support in per_episode:
        supported = {name for name, reason in support.items() if reason is None}
        truth_primary = episode.primary_truth
        if truth_primary != "none" and truth_primary not in supported:
            column = "unsupported"
        else:
            column = _primary_prediction(predicted, supported)
        matrix[rows.index(truth_primary)][columns.index(column)] += 1

    n_episodes = len(dataset.episodes)
    minutes = sum(e.telemetry.duration_s for e in dataset.episodes) / 60.0
    notes = []
    if dataset.report_kind == "fixture_regression":
        notes.append(
            "Fixture regression: labels come from the generating recipe. This report is not "
            "detector validation and no accuracy figure in it may be quoted as such."
        )
    return DetectorEvaluationReport(
        dataset_id=dataset.dataset_id,
        report_kind=dataset.report_kind,
        validation_claim_allowed=dataset.validation_claim_allowed,
        label_source=dataset.label_source,
        platform=dataset.platform,
        detector=probe.label,
        n_episodes=n_episodes,
        telemetry_minutes=minutes,
        categories=dataset.categories,
        per_phenotype=per_phenotype,
        confusion=ConfusionMatrix(rows, columns, tuple(tuple(r) for r in matrix)),
        false_positive_events=total_fp_events,
        false_positives_per_episode=total_fp_events / n_episodes,
        false_positives_per_minute=(total_fp_events / minutes) if minutes > 0 else None,
        unsupported_phenotypes=unsupported,
        predictions_on_unsupported=predictions_on_unsupported,
        notes=tuple(notes),
    )


def render_markdown(report: DetectorEvaluationReport) -> str:
    lines = [
        f"## Detector evaluation ({report.report_kind})",
        "",
        f"Detector `{report.detector}` on `{report.dataset_id}` "
        f"({report.n_episodes} episodes, {report.telemetry_minutes:.2f} min, "
        f"labels: {report.label_source}, platform: {report.platform}).",
        "",
    ]
    if not report.validation_claim_allowed:
        lines += ["**Not a validation result.** " + " ".join(report.notes), ""]
    lines += [
        "| phenotype | scored | unsupported | TP | FP | TN | FN | precision | recall | F1 "
        "| FP events/episode | onset median s | inside window |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value):
        return (
            "n/a" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))
        )

    for name, m in report.per_phenotype.items():
        lines.append(
            f"| {name} | {m.scored_episodes} | {m.unsupported_episodes} | {m.true_positive} | "
            f"{m.false_positive} | {m.true_negative} | {m.false_negative} | {fmt(m.precision)} | "
            f"{fmt(m.recall)} | {fmt(m.f1)} | {fmt(m.false_positive_events_per_episode)} | "
            f"{fmt(m.onset_timing.median_error_s)} | {fmt(m.onset_timing.fraction_inside_window)} |"
        )
    lines += [
        "",
        f"False-positive events: {report.false_positive_events} "
        f"({report.false_positives_per_episode:.3f}/episode, "
        f"{fmt(report.false_positives_per_minute)}/min).",
        "",
        "Confusion (rows truth primary, columns predicted primary): "
        + ", ".join(report.confusion.columns),
    ]
    for row, counts in zip(report.confusion.rows, report.confusion.counts):
        lines.append(f"- {row}: {list(counts)}")
    return "\n".join(lines) + "\n"


__all__ = [
    "ConfusionMatrix",
    "DetectorEvaluationReport",
    "OnsetTiming",
    "PhenotypeMetrics",
    "evaluate_detector",
    "match_windows",
    "one_vs_rest_counts",
    "render_markdown",
]
