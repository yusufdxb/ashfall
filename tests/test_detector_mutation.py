"""Mutation sensitivity of the detector evaluation.

These are scientific guards: each intentionally broken detector must move the
evaluation report by the preregistered margin. If a mutant is not caught the
evaluation is blind to that defect and no accuracy figure from it may be quoted.
"""

from __future__ import annotations

import pytest

from ashfall.datasets import DatasetManifest
from ashfall.detector_eval.dataset import DetectorEvalDataset, LabeledEpisode
from ashfall.detector_eval.fixtures import fixture_dataset, fixture_episodes
from ashfall.detector_eval.metrics import evaluate_detector
from ashfall.detector_eval.mutation import STANDARD_MUTATIONS, run_mutation_suite
from ashfall.taxonomy.phenotype_detector import DetectorMutation, PhenotypeDetector

pytestmark = pytest.mark.guard


@pytest.fixture(scope="module")
def dataset():
    return fixture_dataset()


@pytest.fixture(scope="module")
def suite(dataset):
    return run_mutation_suite(dataset)


@pytest.mark.parametrize("mutation", STANDARD_MUTATIONS, ids=lambda m: m.label)
def test_each_standard_mutant_is_caught(suite, mutation):
    outcome = next(o for o in suite.outcomes if o.mutation == mutation.label)
    assert outcome.caught, f"{mutation.label} not caught: {outcome}"
    assert outcome.criteria_met


def test_suite_reports_all_caught_and_is_content_addressed(suite, dataset):
    assert suite.all_caught and suite.missed == ()
    assert suite.report_kind == "fixture_regression"
    assert suite.report_id == run_mutation_suite(dataset).report_id
    assert set(suite.to_dict()["criterion"]) == {"f1_drop", "fp_rise"}


def test_always_mutants_fire_on_every_episode(dataset):
    for kind in ("always_collapse", "always_stumble", "always_command_mismatch"):
        detector = PhenotypeDetector(mutation=DetectorMutation(kind))
        forced = kind.removeprefix("always_")
        for episode in dataset.episodes:
            events = detector.events(episode.telemetry.as_mapping(), episode.telemetry.dt_s)
            assert events[0].phenotype == forced and events[0].frame == 0


def test_no_slip_priority_emits_command_mismatch_where_the_base_stands_down():
    slip = next(e for e in fixture_episodes() if e.episode_id == "slip_01")
    base = PhenotypeDetector().events(slip.telemetry.as_mapping(), slip.telemetry.dt_s)
    mutant = PhenotypeDetector(mutation=DetectorMutation("no_slip_priority")).events(
        slip.telemetry.as_mapping(), slip.telemetry.dt_s
    )
    assert not any(e.phenotype == "command_mismatch" for e in base)
    assert any(e.phenotype == "command_mismatch" for e in mutant)
    assert any(e.phenotype == "slip" for e in mutant)


def test_label_swap_is_an_involution_and_moves_confusion_mass(dataset):
    swap = DetectorMutation("label_swap", ("slip", "collapse"))
    swapped = PhenotypeDetector(mutation=swap)
    plain = PhenotypeDetector()
    for episode in dataset.episodes:
        a = [
            (e.phenotype, e.frame)
            for e in plain.events(episode.telemetry.as_mapping(), episode.telemetry.dt_s)
        ]
        b = [
            (e.phenotype, e.frame)
            for e in swapped.events(episode.telemetry.as_mapping(), episode.telemetry.dt_s)
        ]
        table = {"slip": "collapse", "collapse": "slip"}
        assert sorted(b) == sorted((table.get(n, n), f) for n, f in a)
    report = evaluate_detector(dataset, lambda: PhenotypeDetector(mutation=swap))
    assert report.confusion.cell("slip", "collapse") > 0
    assert report.confusion.cell("collapse", "slip") > 0


def test_mutation_configuration_is_validated():
    with pytest.raises(ValueError, match="unknown mutation"):
        DetectorMutation("always_wrong")
    with pytest.raises(ValueError, match="two distinct"):
        DetectorMutation("label_swap", ("slip", "slip"))
    with pytest.raises(ValueError, match="unknown phenotype"):
        DetectorMutation("label_swap", ("slip", "tumble"))
    with pytest.raises(ValueError, match="label_swap only"):
        DetectorMutation("always_collapse", ("slip", "collapse"))
    with pytest.raises(ValueError, match="baseline"):
        run_mutation_suite(fixture_dataset(), mutations=(DetectorMutation(),))


def test_a_blind_evaluation_is_reported_not_hidden():
    """On negatives only, a label swap cannot move anything: the suite must say so."""
    negatives = tuple(e for e in fixture_episodes() if e.category == "negative")
    manifest = DatasetManifest("fixture", "synthetic_generator", {"n.jsonl": "0" * 64}, "negatives")
    dataset = DetectorEvalDataset(negatives, manifest, "synthetic_fixture")
    report = run_mutation_suite(
        dataset, mutations=(DetectorMutation("label_swap", ("slip", "collapse")),)
    )
    assert not report.all_caught
    assert report.missed == ("label_swap(slip,collapse)",)


def test_detector_id_names_the_mutation():
    assert PhenotypeDetector().label.endswith("FailureDetector")
    assert PhenotypeDetector(mutation=DetectorMutation("always_collapse")).label.endswith(
        "+always_collapse"
    )
    assert isinstance(LabeledEpisode, type)
