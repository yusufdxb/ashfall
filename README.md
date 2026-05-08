# Ashfall

**Failure-driven robot learning for the Unitree GO2.**

Ashfall is a failure-driven policy adaptation system for quadruped locomotion. It detects hardware failures, extracts replayable failure segments, reconstructs them in simulation with controlled variation, and fine-tunes the locomotion policy to reduce repeated failures over time.

## The Ashfall Loop

```
                    +-------------------+
                    |  Baseline Policy  |
                    |  (PPO, Isaac Lab) |
                    +--------+----------+
                             |
                             v
                    +-------------------+
                    |  Deploy on GO2    |
                    |  (ONNX, ROS 2)   |
                    +--------+----------+
                             |
                             v
                    +-------------------+
                    |  Failure Detector |
                    |  (6-mode taxonomy)|
                    +--------+----------+
                             |
                             v
                    +-------------------+
                    |  Trajectory Log   |
                    |  (Parquet, 50 Hz) |
                    +--------+----------+
                             |
                             v
                    +-------------------+
                    |  Replay in Sim    |
                    |  (Halton DR)      |
                    +--------+----------+
                             |
                             v
                    +-------------------+
                    |  Fine-Tune Policy |
                    |  (Failure Curric.)|
                    +--------+----------+
                             |
                             v
                    +-------------------+
                    |  Evaluate         |
                    |  (Baseline vs     |
                    |   Adapted vs Ctrl)|
                    +-------------------+
```

## Failure Taxonomy

Ashfall classifies quadruped locomotion failures into 6 modes, ordered by severity:

| Mode | Sev. | Detection | Sim Replay Strategy |
|------|------|-----------|---------------------|
| Body Collapse | 5 | base_height < 0.15 m | Vary terrain + joint stiffness |
| Attitude Loss | 4 | \|pitch\| > 0.8 rad or \|roll\| > 0.6 rad | Sweep friction + push forces |
| Foot Slip | 3 | cmd > 0.3 m/s, actual < 0.05 m/s for 0.5 s | Low-friction terrain, sweep coeff. |
| Stumble | 2 | max \|joint_vel\| > 15 rad/s with feet in contact | Add terrain obstacles at swing height |
| Contact Loss | 2 | >= 2 feet below 5N for >= 0.1 s | Vary slope and surface irregularity |
| Command Mismatch | 1 | \|cmd - actual\| > 0.4 m/s for > 1.0 s | Sweep mass + actuator strength |

## Results (Simulation)

Status as of 2026-05-07: the failure-fraction curriculum produces a directionally consistent slippery lift across seeds, but does not yet clear statistical significance at n=3. Seed-scaling work is in progress. Numbers below reflect the multi-seed pilot, not the earlier single-seed framing.

### 2026-05-07 multi-seed pilot (n=3, paired)

2 ff values (0.0, 0.5) x 3 seeds (42, 123, 7) on Phoenix `audit-fixes-2026-04-16` + commit `d42ee01` (FailureCurriculum seed-propagation fix). 200-iter PPO fine-tune from rough baseline, 128-140 eval episodes per cell per terrain.

| terrain  | ff=0.0 mean (SE) | ff=0.5 mean (SE) | paired delta | per-seed signs | exact sign-flip p |
|----------|------------------|------------------|--------------|----------------|-------------------|
| slippery | 0.895 (0.011)    | 0.916 (0.005)    | +0.0214      | 3/3 positive   | 0.250 (n=3 floor) |
| rough    | 0.946 (0.020)    | 0.912 (0.014)    | -0.0336      | 1/3 positive   | 0.500             |

Honest reading:
- **Slippery: directionally consistent, not yet rigour-passing.** 3/3 seeds positive but the 95% CI on the paired mean delta is `[-2.43, +6.71] pp`, crossing zero. The +5.1 pp single-seed value reported in v0.3.0 was on the high side of the seed distribution; cross-seed mean is roughly half.
- **Rough: regresses on average.** 1/3 seeds positive, mean delta -3.36 pp. v0.3.0's single-seed positive on rough was a coincidence.
- **n=3 sign-flip floor is p=0.25 by construction.** Significance at alpha=0.05 is mathematically unreachable until n>=5; the next gate is seed scaling, not a new ablation axis.

Full numbers: [`notes/2026-05-07-multiseed-verdict.md`](notes/2026-05-07-multiseed-verdict.md). Methodology: [`docs/methodology/2026-05-07-ff-sweep-rigor.md`](docs/methodology/2026-05-07-ff-sweep-rigor.md).

### Earlier results (single-seed, kept for context)

The v0.2.0 baseline-vs-adapted comparison and the v0.3.0 6-cell `failure_fraction` sweep (over {0.0, 0.1, 0.25, 0.5, 0.75, 1.0}) were both single-seed (42). Point estimates from those runs were not reproducible at the +5.1 pp / +6.1 pp magnitude under the multi-seed pilot, and no cell was significant after Holm-Bonferroni adjustment with single-seed n~130. The v0.3.0 framing is superseded by the multi-seed pilot above; details are in `notes/2026-05-07-sweep-verification.md` and `docs/methodology/2026-05-07-ff-sweep-rigor.md` for history.

The mode-subset ablation at fixed ff=0.5 is scaffolded under [`configs/ablations/failure_modes/`](configs/ablations/failure_modes/) but is gated behind seed scaling: running it at n=3 would inherit the same p-floor.

### Taxonomy validation (2026-04-19, no GPU)

The 6-mode `FailureDetector` was exercised against 18 synth parquets (6 modes × 3 variants) generated by `scripts/generate_failures.sh`. Every parquet's designed failure mode was correctly detected with zero cross-fires:

| mode | detected | cross-fires |
|---|---:|---:|
| attitude | 3 / 3 | 0 |
| collapse | 3 / 3 | 0 |
| slip | 3 / 3 | 0 |
| stumble | 3 / 3 | 0 |
| contact_loss | 3 / 3 | 0 |
| command_mismatch | 3 / 3 | 0 |

The ablation-sweep generator (`scripts/run_ablation.sh`) produces 6 `failure_fraction` cells (0.0, 0.1, 0.25, 0.5, 0.75, 1.0) with per-cell `commands.sh` stubs ready to execute inside Isaac Lab. The analysis pipeline (`scripts/analyze.sh`) consumes the results directory and writes `results/REPORT.md` — full tables + plots — with zero experiments populated until training runs land.

## Project Structure

```
ashfall/
  src/ashfall/
    taxonomy/          # 6-mode failure detector (pure numpy)
      detector.py      # Stateful multi-mode failure classifier
      schema.py        # Taxonomy metadata and table generation
    experiment/        # Experiment management
      schema.py        # Config/result dataclasses
      runner.py        # Pipeline orchestration (generates Isaac Lab commands)
      sweep.py         # Ablation sweep generation
    evaluation/        # Comparison framework
      harness.py       # Multi-condition comparison + bootstrap CI
      metrics.py       # Failure-specific metrics (recurrence, intervention)
    analysis/          # Post-hoc analysis
      plots.py         # Matplotlib visualizations
      tables.py        # Markdown table generation
      report.py        # Auto-generated experiment report
    synth/             # Synthetic failure generation
      generator.py     # Generates training data for all 6 failure modes
  configs/
    experiments/       # Named experiment configs (baseline, adapted, control, ablation)
    taxonomy.yaml      # Failure detection thresholds
  scripts/             # Shell scripts for reproducible runs
  data/failures/       # Failure trajectory Parquets (synthetic + hardware)
  results/             # Experiment outputs (metrics, plots, reports)
  tests/               # 56 unit tests
```

## Dependencies

Ashfall builds on [go2-phoenix](https://github.com/yusufdxb/go2-phoenix) for Isaac Lab simulation, PPO training, and sim-to-real export. Phoenix handles the sim/training/deployment side; Ashfall adds the experiment, evaluation, and adaptation orchestration layer.

| Component | Source |
|-----------|--------|
| Isaac Lab GO2 env | go2-phoenix `sim_env/` |
| PPO training | go2-phoenix `training/` (rsl_rl) |
| ONNX export | go2-phoenix `sim2real/` |
| ROS 2 deploy | go2-phoenix `sim2real/` |
| Failure detection | **Ashfall** `taxonomy/` (extends Phoenix's 3 modes to 6) |
| Trajectory logging | go2-phoenix `real_world/` |
| Replay + DR | go2-phoenix `replay/` |
| Failure curriculum | go2-phoenix `adaptation/` |
| Experiment management | **Ashfall** `experiment/` |
| Evaluation harness | **Ashfall** `evaluation/` |
| Analysis pipeline | **Ashfall** `analysis/` |
| Synthetic failures | **Ashfall** `synth/` |

## Quick Start

```bash
# Install
cd ~/Projects/ashfall
pip install -e ".[dev]"

# Run tests (no GPU required)
python3 -m pytest tests/ -v

# Generate synthetic failure data
./scripts/generate_failures.sh

# Prepare an experiment (generates Isaac Lab commands)
./scripts/run_experiment.sh configs/experiments/baseline.yaml

# Generate analysis report
./scripts/analyze.sh results/
```

### Running Experiments (requires Isaac Lab + GPU)

```bash
# Set environment
export ISAACLAB_PATH=$HOME/IsaacLab
export PHOENIX_ROOT=$HOME/workspace/go2-phoenix

# Train baseline (500 iters, ~28 min on RTX 5070)
./scripts/run_experiment.sh configs/experiments/baseline.yaml

# Train adapted policy (200 iters from baseline checkpoint)
./scripts/run_experiment.sh configs/experiments/adapted.yaml

# Run control condition
./scripts/run_experiment.sh configs/experiments/control_random.yaml

# Run ablation sweep
./scripts/run_ablation.sh configs/experiments/ablation_sweep.yaml

# Generate report
./scripts/analyze.sh
```

## Ablation Plan

| Axis | Values | Hypothesis |
|------|--------|------------|
| Failure fraction | 0.0, 0.1, 0.25, 0.5, 0.75, 1.0 | More failure data improves adaptation up to a point |
| Failure modes | single-mode vs all-mode | Multi-mode curriculum is more robust |
| Adaptation iters | 50, 100, 200, 400 | Diminishing returns past 200 iters |
| Domain randomization | narrow vs wide | Wider DR improves transfer but may hurt convergence |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Success rate | Episodes completing without early termination |
| Mean return | Average cumulative reward per episode |
| Failure rate | Failures per episode |
| Intervention count | Episodes requiring human intervention (collapse/attitude) |
| Failure recurrence | Same failure mode re-occurring after adaptation |
| Recovery time | Steps from failure detection to stable state |
| Velocity tracking error | Mean command-actual velocity difference |

## Hardware

- **Robot:** Unitree GO2 EDU
- **Onboard compute:** Jetson Orin NX (16 GB)
- **Training GPU:** NVIDIA RTX 5070 (mewtwo)
- **Sim:** NVIDIA Isaac Lab (Isaac Sim 4.5+)
- **Middleware:** ROS 2 Humble

## Limitations

- **n=3 multi-seed pilot is underpowered.** The exact two-sided sign-flip permutation test at n=3 has a p-floor of 0.25; alpha=0.05 is mathematically unreachable. Seed scaling to n>=5 is the gate before any positive significance claim.
- **Real hardware failures not yet collected.** Synthetic failures are physics-approximate, not sim-grade. The first hardware session with the adapted policy will produce ground-truth failure data that closes the loop.
- **Per-episode metric arrays not retained by Phoenix `evaluate.py`.** The current evaluation pipeline emits aggregate scalars per cell, which limits BCa bootstrap and per-mode breakdown to curriculum-input pool composition rather than eval-time failure-mode counts. A Phoenix-side patch to retain per-episode results is tracked as the prerequisite for proper failure-mode ablation.
- **No real-robot deployment validation yet.** The ONNX policy passes parity checks but has not been exercised on the live GO2.
- **Mode-subset ablation gated behind seed scaling.** Configs are scaffolded but running them at n=3 would inherit the same significance ceiling.

## What Makes This Different

This is not a wrapper around existing tools. Ashfall contributes:

1. **A 6-mode failure taxonomy** grounded in quadruped locomotion failure literature, with stateful detection and per-mode suppression.
2. **An experiment framework** that manages baselines, conditions, ablations, and statistical comparisons as first-class objects.
3. **A failure-specific evaluation layer** that tracks intervention count, failure recurrence, and recovery time beyond standard RL metrics.
4. **Synthetic failure generation** that produces structurally correct training data matching the Phoenix Parquet schema for all failure modes.
5. **Integration with a validated sim-to-real pipeline** (go2-phoenix) that has proven baseline and fine-tune results.

The system is designed so that the next hardware session can close the full loop: deploy baseline, collect real failures, replay in sim, adapt, and evaluate.

## License

MIT
