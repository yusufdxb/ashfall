# Runbook: H0 causal-delivery calibration on Isaac Lab

Status of this document: procedure and commands. Whether the simulator path has
been executed, and with what result, is recorded in `docs/claims_ledger.md`
under H0, not here.

## What the run establishes

For one preregistered pathway (intervention kind, parameters, intended
phenotype), `ashfall h0` runs `n_pairs` matched counterfactual pairs against
the Phoenix backend. Each pair restores the same initial state into one
persistent Isaac Lab scene, runs a control rollout with no intervention and a
treatment rollout with the intervention under the same simulator seed, plus
`replicates_per_pair` no-intervention rollouts under other seeds, and applies
gates D1 (intervention applied and read back), D2 (multivariate departure from
the matched control beyond nominal replicate variation) and D3 (intended
phenotype detected in the treatment arm and not in the control arm). The
pairwise outcomes are summarised by the exact McNemar test and the
preregistered rule in `ashfall.h0.summarise_attempts`.

The output is one evidence bundle: `manifest.json`, `environment.json`,
`provenance.json`, `metrics.json`, `verdict.json`, `index.json`,
`seed_plan.json`, and a `harvest/` directory holding both arms of every pair
as Phoenix-schema parquet, one schema-1.1 capsule per DELIVERED pair, the
attempts and episodes, and a `dataset.json` manifest of kind `scientific`.

## Preconditions

- The Isaac Lab interpreter (Isaac Lab 4.5.x on Isaac Sim 6.x here) with the
  Phoenix sibling importable. Nothing is pip-installed into that interpreter;
  the smoke script sets `PYTHONPATH` to both `src/` trees.
- A GPU with the scene free. The backend builds one environment and keeps it
  for the whole run.
- `PHOENIX_ROOT` pointing at a go2-phoenix checkout at, or interface-compatible
  with, the revision pinned in `ashfall.backends.phoenix_compat`. The CLI checks
  the interface before touching the simulator and records the actual revision
  in `provenance.json`.
- The env config must not declare blocks the simulator ignores, or every such
  block must be acknowledged with `--allow-config-block NAME`. The
  acknowledgement is written into `metrics.json`. `flat_v4.yaml` carries a
  documentation-only `terrain` block, which is why the smoke script passes
  `--allow-config-block terrain`.

## Commands

Implementation smoke (small; cannot reach alpha; expected verdict FAIL on the
p-floor reason or UNINFORMATIVE if a gate is unsupported):

```bash
PHOENIX_ROOT=~/workspace/go2-phoenix \
ISAAC_PYTHON=~/Sim/isaac-sim-venv/bin/python \
  scripts/h0_isaac_smoke.sh results/h0_smoke
```

Preregistered pathway (eight pairs, four replicates each; this is the first
confirmatory H0 row for the friction -> slip pathway):

```bash
export PHOENIX_ROOT=~/workspace/go2-phoenix
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:$PHOENIX_ROOT/src"
~/Sim/isaac-sim-venv/bin/python -m ashfall.cli h0 \
  --backend phoenix \
  --backend-config configs/h0/phoenix_flat_v4.json \
  --spec configs/h0/friction_slip.json \
  --allow-config-block terrain --allow-config-block termination \
  --allow-config-block observation.include --allow-config-block robot.init_state \
  --allow-config-block robot.actuator --allow-config-block perturbation.push_interval_s \
  --phoenix-repo "$PHOENIX_ROOT" \
  --output results/h0
```

One run per pathway. To calibrate another pathway, add a spec under
`configs/h0/` before running it, commit the spec, then run. Do not edit a spec
after its bundle exists; the spec id in the bundle would no longer match.

## Reading the bundle

- `verdict.json`: `verdict` in {PASS, FAIL, UNINFORMATIVE}, the `reasons`,
  `evidence_kind` (`simulation` here; `mock` for the toy surrogate), and
  `is_research_result`.
- `metrics.json`: `h0` (gate pass counts, treatment and control incidence,
  discordant counts, exact p and its floor), `per_pair` statuses,
  `d2_sensitivity` (whether the D2 verdict is stable across shrinkage
  intensities and floors), `acknowledged_unapplied_config_blocks`.
- `harvest/dataset.json`: the scientific dataset manifest; capsules under
  `harvest/capsules/` are the inputs the later repair arms may consume.

## What a run does not establish

- A PASS says the intervention causes the phenotype, as the detector sees it,
  at this policy, command, intensity and horizon. It says nothing about repair.
- A DELIVERED pair whose D3 rests on the threshold detector inherits the
  detector's unvalidated accuracy; see `docs/detector/validation_protocol.md`.
- Startup-mode domain randomisation is drawn once per scene from the first
  pair's seed and shared by every arm; it is recorded as `scene_seed`.
