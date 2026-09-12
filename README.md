# Ashfall

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Causal failure reproduction and phenotype-conditioned policy repair for quadruped locomotion.**

Ashfall asks whether a locomotion failure can be reproduced in simulation by a
named physical cause, whether that reproduction resembles independently
labelled failures, and only then whether a policy repaired around the failure
improves on held-out cases without degrading nominal walking. The robot is a
Unitree GO2. A sibling repository, [go2-phoenix](https://github.com/yusufdxb/go2-phoenix),
owns the simulator integration, PPO and deployment; Ashfall owns the evidence.

The project is built so that a positive, negative or null result would each be
believable. It does not yet have a result.

## Where things stand

Every claim and its evidence is in [`docs/claims_ledger.md`](docs/claims_ledger.md).
In short:

| question | status |
|---|---|
| Does a physical intervention cause its intended failure phenotype more often than a matched no-intervention counterfactual (H0)? | **Not established.** The gate is implemented and exercised on a mock surrogate; one simulator implementation smoke has run and is recorded in the ledger. No preregistered simulator run exists. |
| Is the failure detector accurate? | **Not established.** The evaluation catches five deliberate detector defects on fixture data; no independently labelled physics or hardware data has been scored. |
| Does phenotype-conditioned repair beat equal-compute alternatives (H2)? | **Not tested**, and not allowed to run until H0 passes for every retained phenotype. |
| Did the earlier failure-curriculum experiment show anything about failure curricula? | **No.** It seeded nominal states, so the treatment was never delivered. See [`docs/legacy/README.md`](docs/legacy/README.md). |

## The research story, in the order evidence must arrive

1. **Observed failure.** A physics rollout recorded with its policy hash,
   intervention, command, initial-state hash, termination reason and provenance.
2. **Phenotype definition.** Slip, collapse, stumble or blockage, command
   mismatch, contact loss and attitude loss, each with the channels it needs and
   whether a platform can observe it. Contact loss and stumble are excluded from
   the first study because nothing can currently induce or ground-truth them.
3. **Causal intervention.** Friction reduction, actuator weakening, external
   perturbation, command corruption or payload mass, applied by an adapter that
   reads the simulator back. Cause and effect are separate types; an
   intervention is never named after the failure it is hoped to produce.
4. **Matched counterfactual reproduction.** The same restored state, policy,
   command, simulator seed and scene, run once without and once with the
   intervention, plus independent nominal replicates.
5. **Validated delivery.** A pair counts only when the intervention was verified
   by readback (D1), the treatment departs from its matched control by more than
   nominal replicates depart from one another under a shrunk covariance (D2), and
   the intended phenotype appears in the treatment and not the control (D3).
6. **Repair.** Arms A standard, B uniform failure replay, C
   intervention-conditioned replay and D phenotype-conditioned repair, under one
   enforced compute budget.
7. **Held-out evaluation.** Validation selection under a hashed rule, a frozen
   candidate hash, one held-out evaluation recorded in an append-only ledger, and
   an immutable verdict.
8. **Nominal non-degradation.** A preregistered margin per nominal stratum; a
   zero-width interval at ceiling is never accepted as proof.

[`docs/methodology.md`](docs/methodology.md) explains each step and its gate;
[`docs/phase2/PREREGISTRATION.md`](docs/phase2/PREREGISTRATION.md) registers the
hypotheses H0 to H4, the endpoints and the statistics.

## Quick start

```bash
pip install -e ".[dev]"
python3 -m pytest -q            # CPU only, no simulator or GPU
ruff check . && mypy

# Mock end to end: an H0 calibration against the toy surrogate, one evidence bundle.
python3 -m ashfall.cli h0 --backend toy --spec configs/h0/friction_slip.json --output /tmp/h0_toy

# Detector evaluation and mutation suite on the regression fixture (not a validation).
python3 -m ashfall.cli detector-eval --output /tmp/detector_eval
```

The toy surrogate's evidence is labelled `mock` and its datasets are fixtures;
nothing it produces can enter a scientific claim.

For Isaac Lab, follow [`docs/runbooks/h0_isaac.md`](docs/runbooks/h0_isaac.md):

```bash
PHOENIX_ROOT=/path/to/go2-phoenix ISAAC_PYTHON=/path/to/isaac/python \
  scripts/h0_isaac_smoke.sh results/h0_smoke --publication
```

## Evidence bundles

Every scientific run writes a content-addressed directory: `manifest.json`,
`environment.json`, `provenance.json` (Ashfall and Phoenix revisions, config,
dataset and policy hashes, seeds), `metrics.json`, `verdict.json`, `index.json`
and `logs/`. A bundle that cannot name its provenance is refused. See
[`docs/data_provenance.md`](docs/data_provenance.md).

## Repository map

| path | contents |
|---|---|
| `src/ashfall/ontology.py` | interventions, phenotypes, onset windows, episodes, delivery verdicts |
| `src/ashfall/counterfactual.py`, `gates.py`, `h0.py` | matched pairs, D1 to D3, H0 calibration |
| `src/ashfall/harvest.py` | capsules and datasets from delivered simulator pairs |
| `src/ashfall/backends/` | Phoenix / Isaac Lab backend with readback adapters; mock surrogate |
| `src/ashfall/detector_eval/` | independent detector evaluation and mutation suite |
| `src/ashfall/protocol/`, `selection.py` | arms and compute budgets, hypothesis ledger, held-out locking |
| `src/ashfall/stats/`, `frontier.py`, `evaluation/` | primary endpoint, identified frontiers, interval gates |
| `src/ashfall/provenance.py`, `datasets.py`, `config_guard.py` | bundles, fixture versus scientific data, ignored-config refusal |
| `data/fixtures/`, `data/scientific/` | test fixtures (never evidence) and physics or hardware data |
| `results/legacy_row0_curriculum/` | Phase-I evidence, byte-preserved with a SHA-256 index |
| `docs/` | methodology, claims ledger, preregistration, detector protocol, limitations, architecture |

Architecture: [`docs/architecture/ashfall_v3.md`](docs/architecture/ashfall_v3.md).
Known limits: [`docs/limitations.md`](docs/limitations.md).

## Related work

Conditioning a policy on a continuous estimate of physical parameters is
UP-OSI (Yu et al., RSS 2017) and RMA (Kumar et al., RSS 2021); replaying
failure-inducing environment parameters is PLR, Robust PLR and ACCEL. Ashfall
claims neither. The axis it tests is conditioning on a discrete, observed
failure phenotype whose cause has been verified by matched intervention, and
that test is contingent on H0 and on a validated detector, neither of which
exists yet. If the non-conditioned arms match the conditioned one, that is the
result that will be reported.

## Platform

- Robot: Unitree GO2 EDU; onboard Jetson Orin NX (16 GB)
- Simulation: NVIDIA Isaac Lab 4.5 on Isaac Sim 6.0, through go2-phoenix
- Training: NVIDIA (Blackwell) consumer GPU
- Middleware: ROS 2 Humble

## License

MIT
