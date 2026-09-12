"""Independent detector evaluation: labelled episodes, metrics, mutation sensitivity.

Nothing in this package validates the detector by itself. It provides the
path along which a validation could be produced: an independently labelled
dataset (never labelled by the detector's own thresholds), metrics that
refuse to invent a ratio, and a mutation suite that must catch five
deliberate defects before any figure may be quoted.
"""

from ashfall.detector_eval.dataset import (
    CATEGORIES,
    GROUND_TRUTH_SOURCES,
    DetectorEvalDataset,
    LabeledEpisode,
    LabelProvenanceError,
    Telemetry,
)
from ashfall.detector_eval.metrics import (
    DetectorEvaluationReport,
    PhenotypeMetrics,
    evaluate_detector,
    render_markdown,
)
from ashfall.detector_eval.mutation import (
    STANDARD_MUTATIONS,
    MutationReport,
    run_mutation_suite,
)

__all__ = [
    "CATEGORIES",
    "GROUND_TRUTH_SOURCES",
    "STANDARD_MUTATIONS",
    "DetectorEvalDataset",
    "DetectorEvaluationReport",
    "LabelProvenanceError",
    "LabeledEpisode",
    "MutationReport",
    "PhenotypeMetrics",
    "Telemetry",
    "evaluate_detector",
    "render_markdown",
    "run_mutation_suite",
]
