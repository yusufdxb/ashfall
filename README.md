# Ashfall

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Turning a physical quadruped locomotion failure into a reproducible simulator counterexample, repairing the policy around that failure, and verifying the repair without degrading nominal locomotion.**

The robot is a Unitree GO2. Ashfall owns failure capsules, reproduction criteria, parameter search, frontier estimation, curriculum sampling, experimental manifests and acceptance. A sibling repository, [go2-phoenix](https://github.com/yusufdxb/go2-phoenix), owns the simulator, PPO, state restoration and deployment.

## Status: what is established and what is not

The honest answer first, because the repo previously overstated this in both directions.

| Claim | Status |
|---|---|
| A failure curriculum improves robustness | **Untested.** The Phase-I experiment measured a treatment that was never applied. Not refuted either |
| Whether a failure curriculum *delivers* its treatment can be measured | **Established, on synthetic data.** See below. This is currently the repo's strongest result |
| Six-mode failure detector is accurate | **Unvalidated.** Every test sets its input one epsilon past the threshold it tests. No labelled evaluation exists |
| The repair loop runs end to end in simulation | **No.** Three defects block it, listed under [Known defects](#known-defects). Nothing has ever run the simulator paths |
| Any result on real hardware | **No.** The capture path writes zeros where four of six mode signatures live |
| Statistical and provenance machinery | **Built and tested.** 325 tests, exact permutation tests, content-addressed artifacts, split discipline that holds under attack |

## The delivery problem

A failure curriculum resets some fraction of training environments to a recorded pre-failure state. The unexamined assumption is that the state it seeds is actually a pre-failure state. In this repo it usually was not, and that is why Phase I measured nothing.

`ashfall.delivery` measures three things before a capsule is allowed to seed a reset:

1. **Deliverable.** Can the simulator's restore contract write the channels the failure's signature lives in? It writes seven: root pose, root velocity, joint position, joint velocity, command. A mode whose signature lives outside them cannot be delivered by any seed row, friction sweep or reset strategy.
2. **Distinguishable and directional.** Does the seeded state differ from the capsule's *own* nominal window, on the feature carrying that mode's signature, in the direction the mode develops?
3. **Sustained.** Does that departure hold on every frame from the seed row through onset, rather than being a transient excursion that recovered?

Measured on this repo's own six synthetic modes, with a fixed 0.5 s pre-onset rollback at dt=0.02, which is what the production code used:

| mode | onset row | seed row | departure from nominal | delivered |
|---|---:|---:|---:|---|
| command_mismatch | 80 | 55 | +22.2 sigma | yes |
| attitude | 69 | 44 | -1.5 sigma | no, indistinguishable from nominal |
| collapse | 62 | 37 | -1.0 sigma | no, indistinguishable from nominal |
| slip | 70 | 45 | -0.1 sigma | no, indistinguishable from nominal |
| stumble | 50 | 25 | -1.1 sigma | no, and no pre-onset state exists at all |
| contact_loss | 55 | 30 | n/a | no, signature channel is unwritable |

One number cannot fit all six, because failure development time spans 5 to 30 rows across them, a factor of six. `failure_onset_minus_fraction` seeds at a fraction of each capsule's own development window instead. This table is pinned by tests in `tests/test_delivery.py` and reproduces the verdict of a simulator-side delivery probe without needing a simulator.

Two consequences worth stating plainly. `stumble` as generated has zero development time, so there is no frame where it is developing but has not happened. `contact_loss` is undeliverable by construction: contact force is an output of the physics engine given pose, joint state and friction, so no restore call writes it.

## Phase I: a null that does not mean what it looked like

Phase I asked whether raising the fraction of failure trajectories in a fine-tuning curriculum improves success rate, across 11 seeds, paired seed by seed, scored with an exact sign-flip permutation test.

| terrain | n | mean delta (ff=0.5 minus ff=0.0) | 95% CI (t, df=n-1) | seeds positive | exact two-sided sign-flip p |
|:---|---:|---:|:---|:---|---:|
| slippery | 7 | -1.099 pp | [-5.204, +3.006] pp | 4 / 7 | 0.5625 |
| slippery | 11 | -0.416 pp | [-2.823, +1.992] pp | 7 / 11 | 0.7266 |
| rough | 7 | -1.578 pp | [-6.308, +3.153] pp | 2 / 7 | 0.3906 |
| rough | 11 | +0.548 pp | [-3.463, +4.560] pp | 5 / 11 | 0.7676 |

Those numbers reproduce exactly from the committed per-seed metrics under [`results/legacy_row0_curriculum/`](results/legacy_row0_curriculum) and are pinned by a regression test. **What they do not do is answer the question.** Two defects, both found after publication:

- **The treatment was never delivered.** The reset bridge seeded row 0 of each trajectory, which in every shipped synthetic trajectory is a nominal gait state, so roughly half the environments reset to a lightly jittered stand. The result is uninformative about failure curricula rather than evidence against them. The hypothesis is untested, not refuted, and not supported.
- **The terrain contrast is mislabelled.** A `terrain:` config block is silently dropped upstream, so the "rough" and "slippery" arms were built on the same terrain and differed only in friction randomization. That is a real contrast, but not the one the column headers name.

The exact sign-flip p has a floor of 2 divided by 2 to the power of the number of sign-carrying seeds. At n=11 with one exactly-zero delta the floor is 4/2048, so alpha=0.05 was structurally reachable and the test did not approach it. That part of the original analysis stands.

Full reclassification: [`docs/phase2/HYPOTHESIS.md`](docs/phase2/HYPOTHESIS.md). Implementation audit of the legacy pipeline: [`docs/audits/legacy_ashfall_audit.md`](docs/audits/legacy_ashfall_audit.md).

## Failure taxonomy

| Mode | Severity | Detection | Deliverable |
|---|---|---|---|
| Body Collapse | 5 | Instantaneous: base_height < 0.15 m | yes |
| Attitude Loss | 4 | Instantaneous: \|pitch\| > 0.8 rad or \|roll\| > 0.6 rad | yes |
| Foot Slip | 3 | Sustained: cmd_speed > 0.3 m/s and actual_speed < 0.05 m/s for 0.5 s | yes |
| Stumble | 2 | Instantaneous: max \|joint_vel\| > 15 rad/s with >= 2 feet in contact | yes |
| Contact Loss | 2 | Sustained: >= 2 feet below 5 N for >= 0.1 s | **no** |
| Command Mismatch | 1 | Sustained: \|cmd - actual\| > 0.4 m/s for > 1.0 s, excludes slip | yes |

Rendered from `ashfall.taxonomy.schema`. Detection thresholds are event *triggers*, not identified causes: low tracking speed does not distinguish a slip from a blockage, contact dropout without gait phase does not distinguish a failure from normal flight, and an intended crouch trips the collapse threshold. Those confusions are recorded as test evidence in `tests/test_detector_labeling.py` rather than tuned away.

**The detector is not validated.** The 18 synthetic trajectories it is exercised against were authored around its own thresholds, so they establish schema integration and threshold regression, not accuracy. No precision, recall or false-positive rate has been measured on independently labelled data. The procedure for producing that evaluation is written up in [`docs/detector/hand_labeling.md`](docs/detector/hand_labeling.md) and has not been run.

## Known defects

Open, specific, and documented rather than discovered by the reader. Full list with quoted lines in [`docs/audits/v2_implementation_audit.md`](docs/audits/v2_implementation_audit.md).

- `PhoenixBackend.evaluate` does not exist, and `cli.py` calls it. `ashfall evaluate` is the only producer of episode evidence, so the repair loop cannot complete a frontier refresh.
- `create_backend` splats its config unfiltered, so passing the provenance key that `training.py` itself reads raises `TypeError`.
- `evidence_verdict` requires an `environment_parameters` dict equal to the frozen scenario parameters, while the real backend injects ten metadata keys into that field, and requires a `nominal_group` field `EpisodeOutcome` does not have. A clean simulator run would currently be rejected with a message implying simulator drift.
- Two synthetic generators are kinematically inconsistent: position advances at the full commanded speed while the logged body velocity is zeroed, so a "slipping" robot translates at 0.6 m/s.
- CI runs tests only. Lint is configured and never invoked, and the simulator-boundary suite is skipped there for want of torch.

## Architecture

[`docs/architecture/ashfall_v2.md`](docs/architecture/ashfall_v2.md) has the system boundary, the capsule contract, the reproduction and split discipline, and a list of the ways this design can fail.

| Path | Contents |
|---|---|
| `src/ashfall/capsule.py` | Versioned failure record. Content-addressed, explicit nulls, seed row resolved from onset |
| `src/ashfall/delivery.py` | The delivery gate: deliverability, distinguishability, direction, persistence |
| `src/ashfall/reproduction.py` | Frozen reproduction gate plus Halton parameter search. Every attempt retained |
| `src/ashfall/scenarios.py`, `basin.py` | Scenario manifest and split discipline. Identity excludes the split label |
| `src/ashfall/frontier.py` | Weighted isotonic failure probability along one declared severity axis, censored not extrapolated |
| `src/ashfall/repair.py`, `training.py` | Reset sampling and the PPO integration. Probe transitions never enter PPO |
| `src/ashfall/evaluation/` | Frozen arms, per-stratum nominal budgets, cluster-resampled intervals, evidence verdict |
| `src/ashfall/taxonomy/` | Six-mode detector, taxonomy metadata, window labelling machinery |
| `src/ashfall/analysis/` | Paired multi-seed analysis, exact permutation tests, recurrence tables, report rendering |
| `src/ashfall/synth/` | Synthetic failure fixtures and adversarial negatives |
| `src/ashfall/provenance.py` | Canonical JSON hashing, write-once artifacts, run manifests |
| `results/legacy_row0_curriculum/` | Phase-I evidence, byte-preserved, with a SHA-256 provenance index |

## Quick start

```bash
pip install -e ".[dev]"
python3 -m pytest tests/ -q                        # 325 tests, no GPU or simulator needed
ruff check .
python3 -m ashfall.cli demo --output /tmp/demo     # deterministic CPU walk through the loop
```

An editable install also puts an `ashfall` console script on the path; the
module form above is used throughout this README because it works either way.

The demo is software wiring only. It stamps `is_research_result: false`, labels its evidence `mock`, and stores its verdict under `mock_verdict` so it cannot be quoted as a result.

The simulator path, once the defects above are closed, is:

```bash
# Archive a recording under its content hash, then review it into capsules.
python3 -m ashfall.cli capture --trajectory <recording.parquet> --output capsules/
python3 -m ashfall.cli capsule build --trajectory <recording.parquet> \
    --source hardware --robot go2 --checkpoint <sha256> \
    --timestamp 2026-01-01T00:00:00+00:00 --pre-seconds 0.5 --output capsules/

# Reproduce the failure under a frozen gate, then freeze the scenario splits.
python3 -m ashfall.cli reproduce --capsule <capsule.json> --config <gate.json> \
    --backend-config <backend.json> --output reproductions/
python3 -m ashfall.cli basin --reproduction <reproduction.json> --config <basin.json> \
    --output scenarios.json

# Evaluate, estimate the frontier, repair, and accept or reject.
python3 -m ashfall.cli evaluate --scenarios scenarios.json --capsules <capsule.json> \
    --backend-config <backend.json> --policy-id <hash> --evaluation-seeds 1009 1013 1019 \
    --split held_out --output episodes.jsonl
python3 -m ashfall.cli frontier --scenarios scenarios.json --observations <observations.json> \
    --severity <severity.json> --output frontier.json
python3 -m ashfall.cli repair --config <repair.json>
python3 -m ashfall.cli verdict --config <evidence.json> --output verdict.json
```

`python3 -m ashfall.cli <command> --help` documents each one. `evaluate` is the
step blocked by the first defect listed above. Experiment orchestration for the legacy Phase-I path lives in `scripts/`; note that those scripts pin the historical row-0 reset behaviour so the archived numbers keep reproducing, and should not be used for new work.

## Related work

The delivery gate is instrumentation, not a learning contribution. For the learning claim Phase II intends to test, the neighbours are online system identification with a continuous latent (UP-OSI, Yu et al., RSS 2017; RMA, Kumar et al., RSS 2021) and replaying failure-inducing environment parameters (PLR and Robust PLR, Jiang et al., 2021; ACCEL, Parker-Holder et al., 2022). Ashfall claims neither. The only axis claimed as new is conditioning on a discrete, causally named, independently interveneable failure mode, and that claim is contingent on a validated detector, which does not yet exist. The novelty boundary and the mandatory baselines are written out in [`docs/phase2/HYPOTHESIS.md`](docs/phase2/HYPOTHESIS.md).

## Hardware

- Robot: Unitree GO2 EDU
- Onboard compute: Jetson Orin NX (16 GB)
- Training: NVIDIA (Blackwell) consumer GPU
- Simulation: NVIDIA Isaac Lab (Isaac Sim 4.5+)
- Middleware: ROS 2 Humble

## License

MIT
