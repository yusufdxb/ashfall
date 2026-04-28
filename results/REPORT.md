# Ashfall v0.3.0 — Failure-Fraction Sweep

Generated: 2026-04-28 09:52
Cells discovered: 6

## Setup

Each cell warm-starts from the v0.2.0 ashfall-baseline checkpoint (500-iter PPO on rough), then fine-tunes for 200 iters on slippery with `failure_sample_fraction` set to the cell value. Failure trajectories are sampled from the synth pool at `/home/yusuf/Projects/ashfall/data/failures/`. Each adapted checkpoint is evaluated with 128 episodes × 32 envs on rough and slippery (flat is skipped because Flat-v0 obs are incompatible with the Rough-v0 trained obs).

## Results

| failure_fraction | slippery success | slippery 95% CI | rough success | rough 95% CI | slip slew sat | rough slew sat |
|---:|---:|:---:|---:|:---:|---:|---:|
| 0.00 | 0.888 | [0.824, 0.931] | 0.908 | [0.846, 0.946] | 0.327 | 0.391 |
| 0.10 | 0.902 | [0.839, 0.942] | 0.969 | [0.922, 0.988] | 0.251 | 0.317 |
| 0.25 | 0.821 | [0.750, 0.876] | 0.884 | [0.817, 0.928] | 0.292 | 0.368 |
| 0.50 | 0.939 | [0.884, 0.969] | 0.923 | [0.864, 0.958] | 0.298 | 0.377 |
| 0.75 | 0.922 | [0.863, 0.957] | 0.953 | [0.902, 0.979] | 0.381 | 0.413 |
| 1.00 | 0.900 | [0.836, 0.941] | 0.961 | [0.912, 0.983] | 0.505 | 0.500 |

**Optimum (slippery): failure_fraction = 0.50** (slippery 0.939 [0.884, 0.969], rough 0.923 [0.864, 0.958]).

## Slippery <-> Rough Pareto

Control (failure_fraction=0.0): slippery 0.888, rough 0.908.
- ff=0.10: slippery +0.013, rough +0.061
- ff=0.25: slippery -0.067, rough -0.024
- ff=0.50: slippery +0.051, rough +0.015
- ff=0.75: slippery +0.034, rough +0.046
- ff=1.00: slippery +0.012, rough +0.053

## Plots

![success](sweep_success_rate.png)
![slew](sweep_slew_saturation.png)

## Notes
- 95% CIs computed via the Wilson interval over n=num_episodes; evaluate.py emits aggregate counts only, so a full per-episode bootstrap is not available without re-rolling out.
- Flat eval skipped: Flat-v0 obs (no height_scan) are incompatible with Rough-v0 trained policies.