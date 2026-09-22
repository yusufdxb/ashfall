# Runbook: Failure-Boundary Replay

## CPU (runs today)

```bash
pip install -e ".[dev]"
python3 -m pytest -q tests/test_fbr.py                 # 30 tests

# The mechanism toy. Writes baseline.npy, capsule.json, result.json.
python3 -m ashfall.fbr.toy_study --output results/fbr_toy/<name>

# Software check only (3 seeds, a few iterations; numbers mean nothing).
python3 -m ashfall.fbr.toy_study --quick --output /tmp/fbr_quick

# EXPLORATORY budget curves from a completed primary run; aborts if the retrained final
# parameters differ from the primary run's.
python3 -m ashfall.fbr.toy_study --efficiency-from results/fbr_toy/<name> \
    --output results/fbr_toy/<name>_efficiency_exploratory
```

The committed run is `results/fbr_toy/f339087/` (commit `f339087`). The efficiency run verified
that retraining arms B, C and D from the saved baseline reproduces every final evaluation
exactly. A full rerun that also retrains the baseline has not been compared against it.

## Where the constants live

| constants | location |
|---|---|
| toy physics, ES budget | `ToySlipConfig`, `ESConfig` in `src/ashfall/fbr/toy_slip.py` (frozen after the calibration log in `FBR_METHOD.md` section 4.4) |
| toy deployment world, held-out cells, seeds, lead times, nominal fraction, neighbourhood | module constants in `src/ashfall/fbr/toy_study.py` |
| failure criterion | `FailureCriterion` (toy: `TOY_CRITERION`; GO2: `PREREGISTRATION.md` section 8) |
| acceptance thresholds | `AcceptanceConfig` in `src/ashfall/fbr/acceptance.py` |
| GO2 study constants | `docs/research/PREREGISTRATION.md` (FIXED now, BY RULE after E0) |

Every run's `result.json` records the full specification and its content hash (`spec_id`).

## Isaac Lab (NOT YET RUN)

The sequence is experiment E0 in `docs/research/EXPERIMENT.md` section 7. Prerequisites that do
not exist yet, in build order:

1. a spatial patch in the Phoenix-backed environment: per-foot friction written on patch entry
   and exit through `backends.phoenix_interventions.MaterialWriter`, with readback on every write;
2. an adapter from `fbr.boundary.ReplayDraw` to Phoenix `Scenario` objects, so boundary draws go
   through `phoenix.adaptation.scenario_bridge.install_scenario_reset` (nominal draws return
   `None`, as that API expects);
3. boundary refresh inside the PPO loop, using the frontier-probe seam in
   `ashfall.training.run_repair`, with every probe rollout counted in the run's artifact;
4. a baseline retrained on the current Phoenix environment builder (see `docs/limitations.md`,
   environment drift).

Simulator gotchas already found by the H0 smoke (`docs/runbooks/h0_isaac.md`) apply: articulation
and material buffers are Warp arrays; closing `SimulationApp` ends the interpreter, so every
artifact is written first; observation normalisation is resolved from the checkpoint.
