"""Deterministic analytic software demonstration. This is not locomotion evidence."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .basin import discover_basin, evaluate_basin, save_observations
from .capsule import CapsuleFrame, FailureCapsule
from .delivery import assert_delivers
from .evaluation.paired import paired_comparison
from .evaluation.regression import regression_verdict
from .frontier import FrontierEstimator, SeverityMetric, frontier_shift
from .provenance import write_artifact
from .repair import RepairCurriculum
from .reproduction import (
    FailureDescriptor,
    ReproductionCandidate,
    ReproductionConfig,
    ReproductionGate,
)


class AnalyticBackend:
    evidence_kind = 'mock'
    backend_id = 'analytic-threshold-fixture-v1'

    def __init__(self):
        self.thresholds = {'baseline': .45}

    def descriptor(self, parameters, policy_id, seed):
        severity = 1-dict(parameters)['dynamic_friction']
        failed = severity >= self.thresholds[policy_id]
        return FailureDescriptor('slip' if failed else None, .5 if failed else None,
                                 'slip' if failed else 'time_out',
                                 velocity_error_mps=.6 if failed else .02)

    def replay(self, capsule, candidate, policy_id, seed):
        return self.descriptor(candidate.parameters, policy_id, seed)

    def evaluate(self, scenario, policy_id, seed):
        return self.descriptor(scenario.parameters, policy_id, seed)

    def fit_mock_threshold(self, selected):
        """Deliberately analytic update, not PPO, driven by selected training cases."""
        self.thresholds['mock_repaired'] = self.thresholds['baseline'] + .1 * bool(selected)


# Nominal walking, then a slip that develops over ten frames and holds.
# The previous fixture was 80 byte-identical frames, so its "pre-failure" seed
# row was the same state as row 0 and the demo exercised no delivery at all.
# It was the fixture for most of the loop tests, which is why nothing noticed.
_DEMO_NOMINAL_ROWS = 40
_DEMO_ONSET_ROW = 50
_DEMO_TOTAL_ROWS = 80
_DEMO_COMMAND_VX = .5
_DEMO_SLIPPED_VX = .02


def _demo_forward_speed(row: int) -> float:
    """Body forward speed at a row. Deterministic, so the demo stays byte-stable."""
    if row < _DEMO_NOMINAL_ROWS:
        # A little structure so the nominal reference is a distribution rather
        # than a constant, without any randomness.
        return _DEMO_COMMAND_VX + .004 * (1 if row % 2 else -1)
    if row < _DEMO_ONSET_ROW:
        # Traction decays across the development window.
        span = _DEMO_ONSET_ROW - _DEMO_NOMINAL_ROWS
        progress = (row - _DEMO_NOMINAL_ROWS + 1) / span
        return _DEMO_COMMAND_VX - progress * (_DEMO_COMMAND_VX - _DEMO_SLIPPED_VX)
    return _DEMO_SLIPPED_VX


def demo_capsule():
    frames = tuple(
        CapsuleFrame(i*.02, (0., 0., .3), (0., 0., 0., 1.),
                     (_demo_forward_speed(i), 0., 0.), (0., 0., 0.), (0.,)*12, (0.,)*12,
                     (_DEMO_COMMAND_VX, 0., 0.))
        for i in range(_DEMO_TOTAL_ROWS)
    )
    return FailureCapsule('synthetic_fixture', 'quadruped_fixture', 'baseline',
                          '2026-01-01T00:00:00+00:00', .02, 'slip', _DEMO_ONSET_ROW,
                          _DEMO_NOMINAL_ROWS, _DEMO_TOTAL_ROWS - 1, frames,
                          detector_version='fixture-v1', threshold_config_version='fixture-v1')


def run_demo(output: str | Path):
    out = Path(output)
    capsule = demo_capsule()
    capsule.save(out / 'capsule.json')
    # Seed at a fraction of this capsule's own development window, and record
    # the measured delivery evidence alongside the verdict. A fixed 0.5 s
    # rollback would land before the reviewed window here and is refused.
    seed_row = capsule.resolve_seed_index('failure_onset_minus_fraction')
    delivery = assert_delivers(capsule, seed_row)
    backend = AnalyticBackend()
    config = ReproductionConfig('baseline', FailureDescriptor('slip', .5), seed_row)
    candidate = ReproductionCandidate(capsule.capsule_id, (('dynamic_friction', .3),), seed_row)
    reproduction = ReproductionGate(config).run(capsule, candidate, backend)
    reproduction.save(out / 'reproduction.json')
    manifest = discover_basin(reproduction, bounds={'dynamic_friction': (.1, .95)},
                             counts={'train': 32, 'validation': 16, 'held_out': 64},
                             search_seed=19, allow_mock=True)
    manifest.save(out / 'scenarios.json')
    baseline = evaluate_basin(manifest, 'baseline', backend.evaluate,
                              evaluation_seeds=(701, 709, 719), splits=('train',))
    save_observations(out / 'basin_baseline_train.json', baseline)
    estimator = FrontierEstimator(SeverityMetric('slip', 'dynamic_friction', 'coefficient', -1, 1))
    curriculum = RepairCurriculum(manifest, [reproduction], training_seed=11, allow_mock=True)
    curriculum.update({o.scenario_id: list(o.failures) for o in baseline}, 'baseline')
    selected = [s for s in curriculum.sample(128) if s is not None]
    backend.fit_mock_threshold(selected)
    held = {}
    records = {}
    for policy_id in ('baseline', 'mock_repaired'):
        held[policy_id] = evaluate_basin(manifest, policy_id, backend.evaluate,
                                        evaluation_seeds=(809, 811, 821), splits=('held_out',))
        save_observations(out / f'{policy_id}_held_out.json', held[policy_id])
        records[policy_id] = []
        lookup = {s.scenario_id: s for s in manifest.select('held_out')}
        for observation in held[policy_id]:
            s = lookup[observation.scenario_id]
            for seed, failed in zip(observation.evaluation_seeds, observation.failures):
                records[policy_id].append(dict(scenario_id=s.scenario_id,
                    scenario_seed=s.scenario_seed,
                    evaluation_seed=seed, parameter_sample_id=s.parameter_sample_id,
                    policy_id=policy_id, training_seed=11 if policy_id != 'baseline' else None,
                    success=not failed, tracking_error=.6 if failed else .02,
                    intervention_required=False, environment_parameters=dict(s.parameters)))
    margins = {p: estimator.estimate(estimator.points(manifest, o, splits=('held_out',)))
               for p, o in held.items()}
    effect = paired_comparison(records['baseline'], records['mock_repaired'])
    # Independently frozen nominal analytic fixture, no target outcomes in this suite.
    nominal = [dict(scenario_id=f'nominal-{i}', scenario_seed=i, evaluation_seed=900+i,
                    parameter_sample_id=f'nominal-parameters-{i}', success=True,
                    tracking_error=.02, intervention_required=False,
                    environment_parameters={'dynamic_friction': .95}, policy_id='baseline',
                    training_seed=None) for i in range(32)]
    write_artifact(out / 'nominal_fixture.json', nominal)
    candidate_nominal = [{**r, 'policy_id': 'mock_repaired', 'training_seed': 11} for r in nominal]
    verdict = regression_verdict(effect.candidate_minus_baseline, nominal, candidate_nominal)
    # The verdict is deliberately not stored under "verdict". This artifact
    # shares the RepairVerdict schema with a real result and used to read
    # accepted: true next to a plausible frontier shift, distinguished from
    # research evidence only by two adjacent strings. A reader quoting
    # mock_verdict has to quote the word mock.
    result = dict(evidence_kind='mock', is_research_result=False,
                  purpose='software wiring only; no PPO or robot result',
                  capsule_id=capsule.capsule_id, resolved_seed_row=seed_row,
                  delivery_evidence=delivery.to_dict(),
                  reproduction_status=reproduction.status, manifest_hash=manifest.manifest_hash,
                  training_scenario_draws=len(selected),
                  training_ids=sorted({s.scenario_id for s in selected}),
                  baseline_frontier=margins['baseline'].to_dict(),
                  repaired_frontier=margins['mock_repaired'].to_dict(),
                  frontier_shift=frontier_shift(margins['baseline'], margins['mock_repaired']),
                  held_out_effect=asdict(effect), mock_verdict=asdict(verdict))
    write_artifact(out / 'demo.json', result)
    return result
