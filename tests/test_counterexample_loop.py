import json
from dataclasses import replace

import numpy as np
import pytest

from ashfall.basin import BasinObservation, discover_basin
from ashfall.demo import AnalyticBackend, demo_capsule, run_demo
from ashfall.frontier import FrontierEstimator, FrontierPoint, SeverityMetric, frontier_shift
from ashfall.provenance import ExperimentManifest, content_hash, write_artifact
from ashfall.repair import FrontierSchedule, RepairCurriculum, train_with_frontier
from ashfall.reproduction import (
    FailureDescriptor,
    FailureSimilarity,
    ReproductionCandidate,
    ReproductionConfig,
    ReproductionGate,
    sample_parameters,
)
from ashfall.scenarios import ScenarioManifest


def reproduced():
    cap = demo_capsule()
    config = ReproductionConfig('baseline', FailureDescriptor('slip', .5), 45)
    return cap, ReproductionGate(config).run(cap, ReproductionCandidate(cap.capsule_id,
                            (('dynamic_friction', .3),), 45), AnalyticBackend())


def basin():
    cap, result = reproduced()
    return result, discover_basin(result, bounds={'dynamic_friction': (.1, .9)},
                    counts={'train': 10, 'validation': 4, 'held_out': 5}, allow_mock=True)


def test_reproduction_cannot_use_different_policy_or_seed_row():
    cap, result = reproduced()
    with pytest.raises(ValueError, match='baseline policy'):
        ReproductionGate(replace(result.config, baseline_policy_id='other')).run(
            cap, result.candidate, AnalyticBackend())
    with pytest.raises(ValueError, match='seed row'):
        ReproductionGate(result.config).run(
            cap, replace(result.candidate, seed_row=0), AnalyticBackend())


def test_unreproduced_and_mock_cannot_train_real_repair():
    cap, result = reproduced()
    failed = ReproductionGate(result.config).run(
        cap, replace(result.candidate, parameters=(('dynamic_friction', .9),)), AnalyticBackend())
    assert failed.status == 'UNREPRODUCED'
    with pytest.raises(ValueError, match='UNREPRODUCED'):
        failed.assert_eligible(allow_mock=True)
    with pytest.raises(ValueError, match='mock'):
        result.assert_eligible()


def test_similarity_requires_original_mode_and_available_channels():
    _, result = reproduced()
    score = FailureSimilarity(result.config).score
    assert score(FailureDescriptor('collapse', .5)) == 0
    assert score(FailureDescriptor('slip', 2)) == 0
    assert score(FailureDescriptor('slip', .5)) == 1
    detailed = FailureSimilarity(
        replace(result.config, target=FailureDescriptor('slip', .5, attitude_rms_rad=.3)))
    assert detailed.score(FailureDescriptor('slip', .5)) == 0


def test_halton_determinism_and_invalid_bounds():
    assert sample_parameters({'x': (0, 1)}, 12, 4) == sample_parameters({'x': (0, 1)}, 12, 4)
    with pytest.raises(ValueError):
        sample_parameters({'x': (1, 0)}, 2, 4)


def test_scenario_identity_and_split_integrity(tmp_path):
    result, manifest = basin()
    path = tmp_path / 'scenarios.json'
    manifest.save(path)
    assert ScenarioManifest.load(path).manifest_hash == manifest.manifest_hash
    scenario = manifest.scenarios[0]
    assert replace(scenario, split='held_out').scenario_id == scenario.scenario_id
    with pytest.raises(ValueError, match='leakage'):
        ScenarioManifest((scenario, replace(scenario, split='held_out')))
    with pytest.raises(ValueError, match='Physical'):
        ScenarioManifest((scenario, replace(scenario, split='held_out', scenario_seed=999)))
    bad = json.loads(path.read_text())
    bad['scenarios'][0]['parameters'][0][1] += .01
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='identity'):
        ScenarioManifest.load(path)


def test_held_out_cannot_be_sampled_or_refresh_frontier():
    result, manifest = basin()
    c = RepairCurriculum(manifest, [result], allow_mock=True)
    outcomes = {s.scenario_id: [False, True] for s in manifest.select('train')}
    with pytest.raises(ValueError, match='refresh'):
        c.sample(1)
    with pytest.raises(ValueError, match='training'):
        c.update({**outcomes, manifest.select('held_out')[0].scenario_id: [True]}, 'baseline')
    c.update(outcomes, 'baseline')
    draws = c.sample(1000)
    assert all(s is None or s.split == 'train' for s in draws)
    assert 400 < sum(s is None for s in draws) < 600
    with pytest.raises(ValueError, match='refresh'):
        c.sample(1, iteration=10)


def estimator():
    return FrontierEstimator(SeverityMetric('slip', 'friction', 'coefficient', -1, 1))


def points(counts):
    return [FrontierPoint(str(i), float(i), f, 10, f/10, .2) for i, f in enumerate(counts)]


def test_r50_interpolation_censoring_and_nonmonotonic_pooling():
    e = estimator()
    crossing = e.estimate(points([0, 2, 8, 10]))
    assert crossing.threshold == 1.5 and crossing.censoring == 'interpolated_crossing'
    assert e.estimate(points([0, 0])).censoring == 'above_support'
    assert e.estimate(points([10, 10])).censoring == 'below_support'
    flat = e.estimate(points([5, 5]))
    assert flat.threshold is None and flat.censoring == 'unidentifiable'
    # Raw 0, .8, .2, 1.0 pools to 0, .5, .5, 1.0: the crossing is anywhere in
    # the plateau, so the data identify an interval, never the point 1.0 the
    # earlier estimator reported from the left edge of the pooled block.
    pooled = e.estimate(points([0, 8, 2, 10]))
    assert pooled.threshold is None and pooled.censoring == 'identified_interval'
    assert pooled.threshold_interval == (1.0, 2.0)
    assert pooled.monotone_violations == 1 and not pooled.monotonic_assumption
    assert frontier_shift(crossing, pooled) is None
    a, b = e.estimate(points([0, 2, 8, 10])), e.estimate(points([0, 0, 4, 10]))
    assert frontier_shift(a, b) > 0
    with pytest.raises(ValueError):
        frontier_shift(a, replace(b, axis={'parameter': 'push'}))


def test_frontier_refuses_mixed_policy_and_evaluation_leak():
    result, manifest = basin()
    s = manifest.select('held_out')[0]
    observation = BasinObservation(s.scenario_id, 'baseline', (1, 2), (True, False), ({}, {}))
    e = FrontierEstimator(SeverityMetric('slip', 'dynamic_friction', 'coefficient', -1, 1))
    with pytest.raises(ValueError, match='Evaluation'):
        e.points(manifest, [observation])
    assert 0 < observation.probability_interval[0] < .5 < observation.probability_interval[1] < 1


def test_on_policy_callback_order_and_refresh():
    result, manifest = basin()
    c = RepairCurriculum(manifest, [result], allow_mock=True,
                          schedule=FrontierSchedule(refresh_interval=2))
    calls = []
    def evaluate(scenarios, policy):
        calls.append('evaluate')
        return {s.scenario_id: [False, True] for s in scenarios}
    def train(i, sample):
        calls.append('train')
        assert len(sample(4)) == 4
    train_with_frontier(curriculum=c, iterations=5, train_iteration=train,
                        evaluate_training=evaluate, policy_identity=lambda: 'current')
    assert calls == ['evaluate', 'train', 'train', 'evaluate',
                     'train', 'train', 'evaluate', 'train']


def test_provenance_canonical_and_write_once(tmp_path):
    assert content_hash({'a': 1, 'b': 2}) == content_hash({'b': 2, 'a': 1})
    a = ExperimentManifest({'seed': 11}, '2026-01-01')
    assert a.experiment_id == replace(a, created_at='2026-02-01').experiment_id
    write_artifact(tmp_path/'x.json', a.to_dict())
    with pytest.raises(FileExistsError):
        write_artifact(tmp_path/'x.json', {'changed': True})
    with pytest.raises(ValueError):
        content_hash({'x': np.nan})


def test_full_loop_is_deterministic_software_evidence(tmp_path):
    a, b = run_demo(tmp_path/'a'), run_demo(tmp_path/'b')
    assert a == b
    assert a['evidence_kind'] == 'mock'
    assert a['resolved_seed_row'] == 45
    # The mock nominal fixture is 32 identical all-success scenarios. A
    # zero-width cluster bootstrap used to pass that as proof the nominal
    # budgets were met; the interval is now flagged degenerate and the exact
    # binary bound from 32 clusters cannot establish a 2 pp margin, so the
    # mock verdict is honestly refused rather than accepted.
    assert not a['mock_verdict']['accepted']
    assert a['mock_verdict']['target_improved']
    reasons = ' '.join(a['mock_verdict']['reasons'])
    assert 'exact_binary_bound' in reasons and 'degenerate interval' in reasons
    assert a['frontier_shift'] > 0
    assert a['baseline_frontier']['censoring'] == 'interpolated_crossing'
    manifest = ScenarioManifest.load(tmp_path/'a/scenarios.json')
    assert not set(a['training_ids']) & {s.scenario_id for s in manifest.select('held_out')}


def test_demo_artifact_cannot_be_read_as_a_research_result(tmp_path):
    """The artifact shares RepairVerdict's schema with a real result.

    It used to store accepted: true under the key "verdict" next to a
    plausible frontier shift, so the only thing separating it from evidence
    was two adjacent strings. Quoting it now requires quoting the word mock.
    """
    result = run_demo(tmp_path/'demo')
    assert result['is_research_result'] is False
    assert 'verdict' not in result
    assert 'accepted' in result['mock_verdict']


def test_demo_records_measured_delivery_not_an_assumed_one(tmp_path):
    result = run_demo(tmp_path/'demo')
    delivery = result['delivery_evidence']
    assert delivery['delivers'] is True
    assert delivery['distinguishable'] and delivery['directional'] and delivery['sustained']
    assert delivery['seed_departure_z'] > 3.0
    assert delivery['seed_row'] == result['resolved_seed_row'] < delivery['onset_row']
