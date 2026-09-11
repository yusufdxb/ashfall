"""Independent event-window labels and detector accounting.

The evaluation unit is a reviewed trajectory window, not a correlated frame.
Each mode has a one-vs-rest confusion matrix; multiple events/modes are allowed.
Unreviewed windows cannot contribute to performance numbers. Synthetic fixture
reports are explicitly distinguished from hardware-label evaluations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ashfall.taxonomy.detector import FailureMode


@dataclass(frozen=True)
class WindowLabel:
    window_id: str
    trajectory_sha256: str
    start_index: int
    end_index: int  # inclusive
    modes: tuple[str, ...]
    label_source: str  # hardware_human, simulation_human, synthetic_fixture
    reviewer: str
    reviewed: bool
    notes: str = ""
    schema_version: str = "1.0"

    def __post_init__(self):
        if self.schema_version != "1.0":
            raise ValueError("unsupported label schema")
        if not self.window_id or not self.reviewer:
            raise ValueError("window identity and reviewer are required")
        if len(self.trajectory_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.trajectory_sha256
        ):
            raise ValueError("trajectory_sha256 must be lowercase SHA256")
        if type(self.start_index) is not int or type(self.end_index) is not int:
            raise ValueError("label indices must be integers")
        if not 0 <= self.start_index <= self.end_index:
            raise ValueError("invalid label window")
        if self.label_source not in {"hardware_human", "simulation_human", "synthetic_fixture"}:
            raise ValueError("unknown label source")
        object.__setattr__(self, "modes", tuple(self.modes))
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("duplicate label mode")
        for mode in self.modes:
            FailureMode(mode)
        if type(self.reviewed) is not bool:
            raise ValueError("reviewed must be boolean")


def save_labels(labels: Iterable[WindowLabel], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(asdict(x), sort_keys=True) + "\n" for x in labels))


def load_labels(path: str | Path) -> list[WindowLabel]:
    return [
        WindowLabel(**json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def evaluate_window_predictions(
    labels: Iterable[WindowLabel], predictions: dict[str, Iterable[str]]
) -> dict:
    labels = list(labels)
    ids = [label.window_id for label in labels]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evaluation window")
    if not labels or any(not label.reviewed for label in labels):
        raise ValueError("evaluation requires reviewed labels")
    if set(ids) != set(predictions):
        raise ValueError("predictions must match exactly the reviewed evaluation windows")
    # A window cannot be copied/overlapped and counted as independent evidence.
    by_trajectory: dict[str, list[WindowLabel]] = {}
    for label in labels:
        for prior in by_trajectory.setdefault(label.trajectory_sha256, []):
            if max(label.start_index, prior.start_index) <= min(label.end_index, prior.end_index):
                raise ValueError("overlapping windows would double-count evidence")
        by_trajectory[label.trajectory_sha256].append(label)
    predicted = {
        key: {FailureMode(mode).value for mode in values} for key, values in predictions.items()
    }
    sources = sorted({label.label_source for label in labels})
    if len(sources) != 1:
        raise ValueError("report fixture, simulation and hardware labels separately")
    confusion = {}
    for mode in FailureMode:
        tp = fp = tn = fn = 0
        for label in labels:
            truth = mode.value in label.modes
            pred = mode.value in predicted[label.window_id]
            tp += int(truth and pred)
            fp += int(not truth and pred)
            tn += int(not truth and not pred)
            fn += int(truth and not pred)
        confusion[mode.value] = {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "matrix": [[tn, fp], [fn, tp]],
        }
    return {
        "unit": "reviewed_trajectory_window",
        "label_source": sources[0],
        "window_count": len(labels),
        "trajectory_count": len(by_trajectory),
        "confusion_matrix_axes": "rows=true [absent,present]; columns=predicted [absent,present]",
        "per_mode": confusion,
    }
