# Ashfall Experiment Report

Generated: 2026-04-15 21:55

Experiments found: 2

## Failure Taxonomy

Ashfall classifies quadruped locomotion failures into 6 modes, ordered by severity:

| Mode | Severity | Detection | Sim Replay Strategy |
|------|----------|-----------|---------------------|
| Body Collapse | 5 | Instantaneous: base_height < 0.15 m. | Spawn from pre-failure state, vary terrain + joint stiffness. |
| Attitude Loss | 4 | Instantaneous: |pitch| > 0.8 rad or |roll| > 0.6 rad. | Reconstruct initial pose and velocity, sweep friction + push forces. |
| Foot Slip | 3 | Sustained: cmd_speed > 0.3 m/s and actual_speed < 0.05 m/s for 0.5 s. | Reconstruct with low-friction terrain, sweep friction coefficient. |
| Stumble | 2 | Instantaneous: max |joint_vel| > 15 rad/s with >= 2 feet in contact. | Add terrain obstacles at swing-foot trajectory height. |
| Contact Loss | 2 | Sustained: >= 2 feet below 5N force for >= 0.1 s. | Vary terrain slope and surface irregularity. |
| Command Mismatch | 1 | Sustained: |cmd - actual| > 0.4 m/s for > 1.0 s (excludes slip). | Replay with same commands, sweep mass + actuator strength. |

## Condition Comparison

| Condition | Env | Success Rate | Mean Episode Return | Failure Rate |
|-----------|-----|--------------|---------------------|--------------|
| adapted | flat | 100.0% | 19.20 | 0.0% |
| adapted | rough | 96.9% | 17.56 | 3.1% |
| adapted | slippery | 100.0% | 16.64 | 0.0% |
| baseline | flat | 100.0% | 19.50 | 0.0% |
| baseline | rough | 100.0% | 18.95 | 0.0% |
| baseline | slippery | 90.6% | 15.90 | 9.4% |

## Key Findings

- **flat**: Adapted unchanged (100.0% -> 100.0%, delta=+0.0%)
- **rough**: Adapted regresses (100.0% -> 96.9%, delta=-3.1%)
- **slippery**: Adapted improves (90.6% -> 100.0%, delta=+9.4%)

## Plots

No plots generated yet. Run `python -m ashfall.analysis.plots` after experiments complete.

## Limitations

- Synthetic failure trajectories are physics-approximate, not sim-grade
- Real-hardware failure collection requires a lab session with GO2
- Bootstrap CIs require per-episode metric arrays (not yet collected)
- Training curves require TensorBoard log parsing (planned)