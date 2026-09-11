"""Artifact-driven acceptance, with no manual success-rate substitutes."""
from __future__ import annotations


def verdict_from_files(spec):
    from phoenix.training.episode_outcomes import load_outcomes

    from .evaluation.protocol import ExperimentProtocol, NominalSuite, evidence_verdict
    from .reproduction import load_reproduction
    from .scenarios import ScenarioManifest

    def records(name):
        return [r.to_dict() for r in load_outcomes(spec[name])]
    return evidence_verdict(records('baseline_target'), records('candidate_target'),
        records('baseline_nominal'), records('candidate_nominal'),
        counterexample_manifest=ScenarioManifest.load(spec['scenarios']),
        nominal_suite=NominalSuite.load(spec['nominal_suite']),
        reproductions=[load_reproduction(p) for p in spec['reproductions']],
        protocol=ExperimentProtocol.load(spec['protocol']))
