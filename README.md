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

Using Phoenix's baseline (500-iter PPO on rough terrain) and adapted (200-iter warm-start on slippery terrain) policies:

| Condition | Environment | Success Rate | Mean Return | Failure Rate |
|-----------|-------------|-------------|-------------|--------------|
| baseline | flat | 100.0% | 19.50 | 0.0% |
| baseline | rough | 100.0% | 18.95 | 0.0% |
| baseline | slippery | 90.6% | 15.90 | 9.4% |
| adapted | flat | 100.0% | 19.20 | 0.0% |
| adapted | rough | 96.9% | 17.56 | 3.1% |
| adapted | slippery | **100.0%** | 16.64 | **0.0%** |

The adapted policy eliminates all failures on slippery terrain (+9.4% success rate) at the cost of a minor regression on rough terrain (-3.1%).

> **Rigor caveat (added 2026-05-07).** The v0.2.0 baseline-vs-adapted comparison above used a less stringent control (raw v0.2.0 checkpoint, no slippery fine-tune). The proper control is `failure_fraction=0.0` *with* slippery fine-tune; under that control the v0.3.0 sweep finds a point-estimate slippery optimum at `ff=0.5` (+5.1 pp) but the BCa 95% CI on the difference straddles 0 and the Holm-adjusted Fisher p-value is 0.76. See [`docs/methodology/2026-05-07-ff-sweep-rigor.md`](docs/methodology/2026-05-07-ff-sweep-rigor.md) for the full statistical workup, and [`results/REPORT.md`](results/REPORT.md) for the auto-generated table.

### v0.3.0 failure-fraction sweep (2026-04-28, with 2026-05-07 rigor pass)

6 cells in `failure_fraction` over {0.0, 0.1, 0.25, 0.5, 0.75, 1.0}, single seed (42), 128-140 episodes per cell per terrain. Point-estimate optima:
- Slippery: `ff=0.50` (+5.1 pp vs ff=0.0 control), CI `[-0.017, +0.118]`, Holm p=0.76.
- Rough: `ff=0.10` (+6.1 pp vs ff=0.0 control), CI `[+0.007, +0.115]`, Holm p=0.34.
- Joint Pareto: `ff=0.75` (slippery +3.4 pp, rough +4.6 pp; no regression on either axis).

No cell is significant after multiple-comparison adjustment with single-seed n~130. The next ablation (mode-subset sweep at fixed `ff=0.5`) is scaffolded under [`configs/ablations/failure_modes/`](configs/ablations/failure_modes/) and ready to run when GPU is free.

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

- **Real hardware failures not yet collected.** Synthetic failures are physics-approximate, not sim-grade. The first hardware session with the adapted policy will produce ground-truth failure data that closes the loop.
- **Failure curriculum currently disabled** in go2-phoenix (failure_sample_fraction=0.0). The reset bridge is wired and tested but waiting for hardware-captured Parquets.
- **No real-robot deployment validation yet.** The ONNX policy passes parity checks but has not been exercised on the live GO2.
- **Bootstrap CIs** require per-episode metric arrays that are not yet collected during Isaac Lab evaluation.

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
