"""Independent detector evaluation: provenance refusals, metrics arithmetic, windows."""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ashfall.datasets import DatasetManifest
from ashfall.detector_eval.dataset import (
    DetectorEvalDataset,
    LabeledEpisode,
    LabelProvenanceError,
    Telemetry,
)
from ashfall.detector_eval.fixtures import DT, FIXTURE_FILE, fixture_dataset, fixture_episodes
from ashfall.detector_eval.metrics import (
    DetectorEvaluationReport,
    evaluate_detector,
    match_windows,
    one_vs_rest_counts,
    render_markdown,
)
from ashfall.ontology import OnsetWindow, PhenotypeObservation
from ashfall.taxonomy.phenotype_detector import (
    DetectorMutation,
    PhenotypeDetector,
    supported_phenotypes,
)

SHA = "a" * 64


def telemetry(n=100, cmd_vx=0.5, joints=True, contacts=True):
    cmd = np.tile([cmd_vx, 0.0], (n, 1))
    return Telemetry(
        DT,
        np.zeros(n),
        np.zeros(n),
        np.full(n, 0.3),
        cmd,
        cmd.copy(),
        np.zeros((n, 12)) if joints else None,
        np.full((n, 4), 50.0) if contacts else None,
    )


def window(transition, established, end=None, source="synthetic_fixture"):
    return OnsetWindow(transition, established, None, end, source)


def fixture_manifest():
    return DatasetManifest("fixture", "synthetic_generator", {"x.jsonl": SHA}, "fixture")


def scientific_manifest():
    return DatasetManifest(
        "scientific",
        "simulation",
        {"x.jsonl": SHA},
        "sim ground truth",
        dict(
            policy_id=SHA,
            env_config_hash=SHA,
            simulator_version="4.5.22",
            seeds=[1],
            ashfall_sha="1" * 40,
            phoenix_sha="2" * 40,
        ),
    )


class TestLabelProvenance:
    def test_detector_labelled_ground_truth_is_refused(self):
        with pytest.raises(LabelProvenanceError, match="cannot be labelled by the detector"):
            LabeledEpisode("e", telemetry(), (), "negative", "detector")

    def test_window_source_must_match_episode_source(self):
        obs = PhenotypeObservation("slip", window(10, 20, source="detector"))
        with pytest.raises(LabelProvenanceError, match="differs"):
            LabeledEpisode("e", telemetry(), (obs,), "positive", "synthetic_fixture")

    def test_threshold_evidence_is_refused(self):
        obs = PhenotypeObservation("slip", window(10, 20), {"detector_event_frame": 20})
        with pytest.raises(LabelProvenanceError, match="detector_event_frame"):
            LabeledEpisode("e", telemetry(), (obs,), "positive", "synthetic_fixture")
        with pytest.raises(LabelProvenanceError, match="threshold"):
            LabeledEpisode(
                "e", telemetry(), (), "negative", "synthetic_fixture", {"labelled_by": "threshold"}
            )

    def test_unknown_source_and_category(self):
        with pytest.raises(ValueError, match="label source"):
            LabeledEpisode("e", telemetry(), (), "negative", "oracle")
        with pytest.raises(ValueError, match="category"):
            LabeledEpisode("e", telemetry(), (), "maybe", "synthetic_fixture")


class TestCategories:
    def test_negative_and_near_miss_carry_no_phenotype(self):
        obs = PhenotypeObservation("slip", window(10, 20))
        for category in ("negative", "near_miss"):
            with pytest.raises(ValueError, match="no phenotype"):
                LabeledEpisode("e", telemetry(), (obs,), category, "synthetic_fixture")

    def test_positive_needs_a_window_and_windows_stay_inside(self):
        with pytest.raises(ValueError, match="needs a phenotype"):
            LabeledEpisode("e", telemetry(), (), "positive", "synthetic_fixture")
        with pytest.raises(ValueError, match="outside"):
            LabeledEpisode(
                "e",
                telemetry(50),
                (PhenotypeObservation("slip", window(10, 60)),),
                "positive",
                "synthetic_fixture",
            )

    def test_recovery_needs_the_same_phenotype_twice_with_a_gap(self):
        once = (PhenotypeObservation("slip", window(10, 20, end=40)),)
        with pytest.raises(ValueError, match="recovery"):
            LabeledEpisode("e", telemetry(), once, "recovery", "synthetic_fixture")
        overlapping = (
            PhenotypeObservation("slip", window(10, 20, end=60)),
            PhenotypeObservation("slip", window(50, 55)),
        )
        with pytest.raises(ValueError, match="overlap"):
            LabeledEpisode("e", telemetry(), overlapping, "recovery", "synthetic_fixture")
        fine = (
            PhenotypeObservation("slip", window(10, 20, end=40)),
            PhenotypeObservation("slip", window(60, 70)),
        )
        LabeledEpisode("e", telemetry(), fine, "recovery", "synthetic_fixture")

    def test_multi_failure_needs_two_phenotypes(self):
        one = (
            PhenotypeObservation("slip", window(10, 20, end=40)),
            PhenotypeObservation("slip", window(60, 70)),
        )
        with pytest.raises(ValueError, match="two distinct"):
            LabeledEpisode("e", telemetry(), one, "multi_failure", "synthetic_fixture")


class TestDataset:
    def test_fixture_manifest_means_fixture_regression(self):
        dataset = fixture_dataset()
        assert dataset.report_kind == "fixture_regression"
        assert not dataset.validation_claim_allowed
        report = evaluate_detector(dataset)
        assert report.report_kind == "fixture_regression"
        assert not report.validation_claim_allowed
        assert "not detector validation" in " ".join(report.notes)

    def test_fixture_manifest_hash_is_the_episode_lines(self):
        dataset = fixture_dataset()
        assert dataset.manifest.files[FIXTURE_FILE] == dataset.episodes_sha256()

    def test_scientific_manifest_means_independent_evaluation(self):
        episode = LabeledEpisode("e", telemetry(), (), "negative", "simulator_ground_truth")
        dataset = DetectorEvalDataset((episode,), scientific_manifest(), "simulation")
        assert dataset.report_kind == "independent_evaluation"
        assert evaluate_detector(dataset).validation_claim_allowed

    def test_fixture_labels_cannot_hide_under_a_scientific_manifest(self):
        episode = LabeledEpisode("e", telemetry(), (), "negative", "synthetic_fixture")
        with pytest.raises(ValueError, match="scientific manifest"):
            DetectorEvalDataset((episode,), scientific_manifest(), "simulation")
        sim = LabeledEpisode("s", telemetry(), (), "negative", "simulator_ground_truth")
        with pytest.raises(ValueError, match="fixture manifest"):
            DetectorEvalDataset((sim,), fixture_manifest(), "simulation")
        with pytest.raises(ValueError, match="separately"):
            DetectorEvalDataset((sim, episode), scientific_manifest(), "simulation")

    def test_jsonl_round_trip_preserves_identity(self, tmp_path):
        dataset = fixture_dataset()
        path = dataset.save(tmp_path / "eval.jsonl")
        loaded = DetectorEvalDataset.load(path)
        assert loaded.dataset_id == dataset.dataset_id
        assert loaded.categories == dataset.categories
        assert evaluate_detector(loaded).to_dict() == evaluate_detector(dataset).to_dict()
        dataset.save(path)
        path.write_text(path.read_text() + "\n")
        with pytest.raises(FileExistsError):
            DetectorEvalDataset(dataset.episodes[:1], dataset.manifest, dataset.platform).save(path)

    def test_fixture_covers_every_category(self):
        counts = fixture_dataset().categories
        assert all(counts[c] >= 1 for c in counts)
        assert len(fixture_episodes()) == sum(counts.values())


class TestSupport:
    def test_missing_channels_make_a_phenotype_unsupported(self):
        support = supported_phenotypes(
            ("pitch_rad", "roll_rad", "base_height_m", "cmd_lin_vel", "actual_lin_vel"),
            "synthetic_fixture",
        )
        assert support["stumble"] and "joint_vel" in support["stumble"]
        assert support["contact_loss"] and "contact_forces" in support["contact_loss"]
        assert support["collapse"] is None

    def test_platform_observability_is_consulted(self):
        channels = (
            "pitch_rad",
            "roll_rad",
            "base_height_m",
            "cmd_lin_vel",
            "actual_lin_vel",
            "joint_vel",
            "contact_forces",
        )
        sim = supported_phenotypes(channels, "simulation")
        assert sim["stumble"] and sim["contact_loss"] and sim["slip"] is None
        hardware = supported_phenotypes(channels, "go2_hardware")
        assert hardware["collapse"] and hardware["slip"] and hardware["attitude"] is None
        with pytest.raises(ValueError, match="platform"):
            supported_phenotypes(channels, "mars")

    def test_unsupported_phenotype_is_not_scored_and_predictions_are_counted(self):
        dataset = fixture_dataset()
        report = evaluate_detector(dataset)
        assert report.per_phenotype["stumble"].unsupported_episodes == 1
        assert report.per_phenotype["stumble"].scored_episodes == report.n_episodes - 1
        assert "stumble" in report.unsupported_phenotypes
        assert report.predictions_on_unsupported == 0
        forced = evaluate_detector(
            dataset, lambda: PhenotypeDetector(mutation=DetectorMutation("always_stumble"))
        )
        assert forced.predictions_on_unsupported == 1


class ScriptedDetector:
    """Emits scripted phenotypes keyed on the episode's command, for hand-computed metrics."""

    label = "scripted"

    def __init__(self, script):
        self.script = script

    def run(self, telemetry, dt_s):
        key = round(float(telemetry["cmd_lin_vel"][0][0]), 2)
        return tuple(
            PhenotypeObservation(name, OnsetWindow(frame, frame, None, None, "detector"))
            for name, frame in self.script.get(key, ())
        )


def scripted_dataset():
    episodes = (
        LabeledEpisode(
            "tp",
            telemetry(cmd_vx=0.11),
            (PhenotypeObservation("collapse", window(40, 50)),),
            "positive",
            "synthetic_fixture",
        ),
        LabeledEpisode(
            "fn",
            telemetry(cmd_vx=0.12),
            (PhenotypeObservation("collapse", window(40, 50)),),
            "positive",
            "synthetic_fixture",
        ),
        LabeledEpisode("fp", telemetry(cmd_vx=0.13), (), "negative", "synthetic_fixture"),
        LabeledEpisode("tn", telemetry(cmd_vx=0.14), (), "negative", "synthetic_fixture"),
        LabeledEpisode(
            "early",
            telemetry(cmd_vx=0.15),
            (PhenotypeObservation("collapse", window(40, 50)),),
            "positive",
            "synthetic_fixture",
        ),
    )
    return DetectorEvalDataset(episodes, fixture_manifest(), "synthetic_fixture")


class TestMetricsArithmetic:
    def test_hand_computed_counts_ratios_and_timing(self):
        script = {
            0.11: (("collapse", 50),),  # inside window: TP, error (50-40)*0.02 = 0.2 s
            0.13: (("collapse", 10), ("collapse", 70)),  # negative: FP episode, 2 FP events
            0.15: (("collapse", 30),),  # before transition: FP event and FN episode
        }
        report = evaluate_detector(scripted_dataset(), lambda: ScriptedDetector(script))
        m = report.per_phenotype["collapse"]
        assert (m.true_positive, m.false_positive, m.true_negative, m.false_negative) == (
            1,
            1,
            1,
            2,
        )
        assert m.precision == pytest.approx(0.5)
        assert m.recall == pytest.approx(1 / 3)
        assert m.f1 == pytest.approx(0.4)
        assert m.false_positive_events == 3
        assert m.false_positive_events_per_episode == pytest.approx(3 / 5)
        assert m.onset_timing.windows == 3 and m.onset_timing.matched == 1
        assert m.onset_timing.fraction_inside_window == pytest.approx(1 / 3)
        assert m.onset_timing.median_error_s == pytest.approx(0.2)
        assert m.onset_timing.p90_error_s == pytest.approx(0.2)
        assert report.false_positive_events == 3
        assert report.false_positives_per_episode == pytest.approx(0.6)
        assert report.false_positives_per_minute == pytest.approx(3 / (5 * 100 * DT / 60))
        # The confusion matrix names phenotypes and ignores timing: the early
        # emission still names collapse. Timing lives in the window metrics,
        # which is where that episode is a false negative.
        assert report.confusion.cell("collapse", "collapse") == 2
        assert report.confusion.cell("collapse", "none") == 1
        assert report.confusion.cell("none", "collapse") == 1
        assert report.confusion.cell("none", "none") == 1

    def test_undefined_ratios_are_none_not_one(self):
        report = evaluate_detector(scripted_dataset(), lambda: ScriptedDetector({}))
        m = report.per_phenotype["collapse"]
        assert m.precision is None and m.f1 is None
        assert m.recall == 0.0
        slip = report.per_phenotype["slip"]
        assert slip.precision is None and slip.recall is None and slip.f1 is None
        assert slip.onset_timing.fraction_inside_window is None

    def test_repeated_firing_inside_a_window_is_not_a_false_positive(self):
        matches, unmatched, fps = match_windows(
            (PhenotypeObservation("slip", window(10, 20, end=60)),),
            tuple(
                PhenotypeObservation("slip", OnsetWindow(f, f, None, None, "detector"))
                for f in (25, 45, 80)
            ),
            n_steps=100,
            phenotype="slip",
        )
        assert matches == [(0, 25)] and unmatched == [] and fps == [80]

    def test_recovery_needs_two_detections_for_two_windows(self):
        recovery = next(e for e in fixture_episodes() if e.category == "recovery")
        negative = next(e for e in fixture_episodes() if e.category == "negative")
        dataset = DetectorEvalDataset((recovery, negative), fixture_manifest(), "synthetic_fixture")
        real = evaluate_detector(dataset).per_phenotype["slip"]
        assert real.onset_timing.windows == 2 and real.onset_timing.matched == 2
        once = evaluate_detector(
            dataset, lambda: ScriptedDetector({0.6: (("slip", 70),)})
        ).per_phenotype["slip"]
        assert once.onset_timing.matched == 1
        assert once.onset_timing.fraction_inside_window == pytest.approx(0.5)
        assert once.true_positive == 1  # the episode still counts as detected

    def test_multi_failure_contributes_to_every_phenotype(self):
        multi = next(e for e in fixture_episodes() if e.category == "multi_failure")
        dataset = DetectorEvalDataset((multi,), fixture_manifest(), "synthetic_fixture")
        report = evaluate_detector(dataset)
        assert report.per_phenotype["attitude"].true_positive == 1
        assert report.per_phenotype["collapse"].true_positive == 1
        assert report.confusion.cell("attitude", "attitude") == 1
        assert report.per_phenotype["collapse"].onset_timing.median_error_s > 0

    def test_confusable_blockage_is_recorded_as_slip_confusion(self):
        report = evaluate_detector(fixture_dataset())
        assert report.confusion.cell("command_mismatch", "slip") == 1
        assert report.per_phenotype["command_mismatch"].false_negative == 1

    def test_report_is_content_addressed_and_renders(self):
        report = evaluate_detector(fixture_dataset())
        again = evaluate_detector(fixture_dataset())
        assert report.report_id == again.report_id
        json.dumps(report.to_dict())
        text = render_markdown(report)
        assert "Not a validation result" in text and "| slip |" in text
        with pytest.raises(ValueError, match="validation claim"):
            DetectorEvaluationReport(
                **{**{k: v for k, v in report.__dict__.items()}, "validation_claim_allowed": True}
            )


@given(
    st.lists(st.tuples(st.booleans(), st.booleans()), min_size=0, max_size=40),
)
@settings(max_examples=100, deadline=None)
def test_one_vs_rest_counts_partition_the_episodes(pairs):
    truth = [t for t, _ in pairs]
    predicted = [p for _, p in pairs]
    counts = one_vs_rest_counts(truth, predicted)
    assert sum(counts.values()) == len(pairs)
    assert counts["true_positive"] + counts["false_negative"] == sum(truth)
    assert counts["true_positive"] + counts["false_positive"] == sum(predicted)


@given(
    st.lists(st.integers(min_value=0, max_value=199), min_size=0, max_size=12),
    st.lists(
        st.tuples(st.integers(min_value=0, max_value=190), st.integers(min_value=0, max_value=9)),
        min_size=0,
        max_size=4,
    ),
)
@settings(max_examples=100, deadline=None)
def test_window_matching_never_double_counts(frames, spans):
    truth = []
    cursor = 0
    for start, length in sorted(spans):
        start = max(start, cursor)
        end = min(start + length, 199)
        if start > 199:
            break
        truth.append(
            PhenotypeObservation("slip", OnsetWindow(start, start, None, end, "synthetic_fixture"))
        )
        cursor = end + 1
    predicted = tuple(
        PhenotypeObservation("slip", OnsetWindow(f, f, None, None, "detector")) for f in frames
    )
    matches, unmatched, fps = match_windows(truth, predicted, n_steps=200, phenotype="slip")
    assert len(matches) + len(unmatched) == len(truth)
    assert len({frame for _, frame in matches}) == len(matches)
    inside = [
        f for f in frames if any(o.window.transition_start <= f <= o.window.end for o in truth)
    ]
    assert sorted(fps) == sorted(f for f in frames if f not in inside)
