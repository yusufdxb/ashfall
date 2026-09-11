from dataclasses import replace

import pytest

from ashfall.synth.negatives import adversarial_negatives
from ashfall.taxonomy.detector import FailureDetector
from ashfall.taxonomy.labeling import (
    WindowLabel,
    evaluate_window_predictions,
    load_labels,
    save_labels,
)


def label(identity, modes=(), **kwargs):
    return WindowLabel(
        identity,
        identity.ljust(64, "0"),
        0,
        99,
        modes,
        "synthetic_fixture",
        "fixture-author",
        True,
        **kwargs,
    )


def test_confusion_precision_recall_fpr():
    labels = [label("a", ("slip",)), label("b", ("slip",)), label("c"), label("d")]
    report = evaluate_window_predictions(labels, {"a": ["slip"], "b": [], "c": ["slip"], "d": []})
    slip = report["per_mode"]["slip"]
    assert slip["matrix"] == [[1, 1], [1, 1]]
    assert slip["precision"] == slip["recall"] == slip["false_positive_rate"] == 0.5
    assert report["per_mode"]["collapse"]["recall"] is None
    assert report["label_source"] == "synthetic_fixture"


def test_label_roundtrip(tmp_path):
    labels = [label("a")]
    path = tmp_path / "labels.jsonl"
    save_labels(labels, path)
    assert load_labels(path) == labels


def test_reject_unreviewed_mixed_sources_missing_predictions_and_overlap():
    with pytest.raises(ValueError, match="reviewed"):
        evaluate_window_predictions([replace(label("a"), reviewed=False)], {"a": []})
    with pytest.raises(ValueError, match="separately"):
        evaluate_window_predictions(
            [label("a"), replace(label("b"), label_source="hardware_human")], {"a": [], "b": []}
        )
    with pytest.raises(ValueError, match="exactly"):
        evaluate_window_predictions([label("a")], {})
    overlap = replace(label("b"), trajectory_sha256=label("a").trajectory_sha256)
    with pytest.raises(ValueError, match="overlapping"):
        evaluate_window_predictions([label("a"), overlap], {"a": [], "b": []})


def test_adversarial_negatives_expose_existing_detector_ambiguity():
    observations = {}
    for fixture in adversarial_negatives():
        detector = FailureDetector()
        modes = {event.mode.value for row in fixture.telemetry for event in detector.step(**row)}
        observations[fixture.name] = modes.intersection(fixture.target_negative_modes)
    # Expected limitations are visible test evidence, not a claimed accuracy score.
    assert observations["wall_blockage"] == {"slip"}
    assert observations["normal_trot_contact_loss"] == {"contact_loss"}
    assert observations["successful_bound_flight"] == {"contact_loss"}
    assert observations["high_joint_speed"] == {"stumble"}
    assert observations["intended_low_posture"] == {"collapse"}
    assert observations["commanded_stop"] == set()
    assert observations["successful_turn"] == set()
    assert observations["temporary_tracking_lag"] == set()
