# Failure-Fraction Sweep: Methodology and Rigor Note (2026-05-07)

This note documents the methodology behind Ashfall v0.3.0's
failure-fraction (`ff`) sweep, the statistical re-analysis added on
2026-05-07, what survives review, and what does not. It is the
single source of truth for any external reader (advisor, reviewer,
collaborator) before they look at the numbers.

## 1. Setup

### 1.1 What the sweep measures

The Ashfall curriculum injects synthetic failure trajectories into
the PPO fine-tune buffer at a per-minibatch rate
`failure_sample_fraction` (alias `failure_fraction` or `ff`). The
sweep asks: holding the warm-start checkpoint, fine-tune duration,
and synth pool fixed, how does test-time success rate vary as `ff`
moves from 0 to 1?

### 1.2 Conditions

Every cell warm-starts from the v0.2.0 `ashfall-baseline` checkpoint
(500-iter PPO on `Rough-v0`), then fine-tunes for 200 iterations on
`Slippery-v0` with the listed `failure_fraction`:

| ff   | meaning                                                      |
|-----:|:-------------------------------------------------------------|
| 0.00 | control; pure slippery fine-tune, no failure replay          |
| 0.10 | 10% of each minibatch sourced from synth failure pool        |
| 0.25 | 25%                                                          |
| 0.50 | 50%                                                          |
| 0.75 | 75%                                                          |
| 1.00 | failure-only fine-tune                                       |

The control is `ff=0.0` *with* the slippery fine-tune. This is the
correct control because it isolates the **curriculum effect** from
the **fine-tune effect**: any lift attributable to seeing the
slippery distribution at all (rather than to seeing failures) is
absorbed into the control, and only the marginal value of the
failure-replay channel shows up in the cell-vs-control delta.

A naive control (using the raw v0.2.0 checkpoint with no slippery
fine-tune at all) would conflate both effects and report inflated
deltas. The earlier v0.2.0 paper-style reading of "+5.1pp at
ff=0.25" used a less stringent control; this sweep uses the right
one and the v0.2.0 number does not replicate against it.

### 1.3 Eval protocol

Each adapted checkpoint is rolled out for 128 episodes per
environment with 32 parallel envs. The reported `num_episodes` per
cell is the count of *terminated* episodes inside the eval window
(128 to 140 across the sweep, depending on mean episode length).
Two terrains are evaluated:

- `Slippery-v0`: in-distribution for the fine-tune
- `Rough-v0`: out-of-distribution; tests whether failure replay
  preserved the warm-start's rough proficiency

`Flat-v0` is intentionally skipped: its observation space lacks the
height_scan that the Rough-v0 trained policy expects.

### 1.4 Seeds

Every cell ran with `training.seed = 42` and `device = cuda:0` on the
mewtwo workstation. There is **one training run per cell**. This is
a single-seed pilot; cross-seed variance is unmeasured.

## 2. Statistical approach

### 2.1 Available signal

Phoenix's `evaluate.py` collapses each rollout to aggregate scalars
before writing `metrics_<env>.json`. There are no per-episode success
indicators or per-episode failure-mode labels at eval time. The
binomial summary `(num_episodes, success_rate)` is the entire
useful per-cell output.

### 2.2 Confidence intervals

For each cell's success rate we report a Wilson 95% interval (correct
coverage near 0/1; standard for binomial proportions of n in this
range).

For each non-control cell's *difference vs control*, we run a paired
BCa bootstrap on Bernoulli arrays reconstructed from the binomial
summary. With n_a, n_b near 130 and the bootstrap operating on
indicator arrays, this matches a classical normal-approximation CI on
`p_b - p_a` to 2 decimal places, but it produces the right CI shape
near boundaries and lets us cross-check against Fisher's exact.

Note the limitation: because per-episode arrays are not retained,
bootstrap-on-indicators carries no extra information beyond the
binomial summary. It is a representational convenience, not an
information gain.

### 2.3 Tests and multiple-comparison control

Each non-control cell is tested against `ff=0.0` on the same terrain
using Fisher's exact two-sided test (closed-form, exact). Across the
five non-control cells per terrain we apply Holm-Bonferroni step-down
adjustment, controlling the family-wise error rate at alpha=0.05 per
terrain. We do **not** pool across terrains because the slippery
and rough deltas are *a priori* expected to differ in sign and
magnitude: pooling would over-correct.

### 2.4 What we deliberately did not do

- No paired tests across cells. Episodes are independent rollouts
  and not aligned by index between cells; pretending otherwise would
  inflate apparent precision.
- No per-mode breakdown at eval time. The required per-episode
  failure-mode labels were not collected by `evaluate.py`. We report
  the curriculum-input pool composition instead, clearly labeled.
- No cross-seed analysis. Single-seed is a noted limitation.

## 3. Headline result with CIs

Point estimates and BCa 95% CIs on the difference vs `ff=0.0` control
(Holm-adjusted Fisher p-values in parentheses):

### Slippery (control success rate 0.888, n=134)

| ff   | success | delta vs ff=0.0 | 95% CI            | Holm p   |
|-----:|--------:|----------------:|:------------------|---------:|
| 0.10 | 0.902   | +0.013          | [-0.062, +0.088]  | 1.000    |
| 0.25 | 0.821   | -0.067          | [-0.153, +0.014]  | 0.635    |
| 0.50 | 0.939   | **+0.051**      | [-0.017, +0.118]  | 0.761    |
| 0.75 | 0.922   | +0.034          | [-0.041, +0.102]  | 1.000    |
| 1.00 | 0.900   | +0.012          | [-0.064, +0.086]  | 1.000    |

### Rough (control success rate 0.908, n=130)

| ff   | success | delta vs ff=0.0 | 95% CI            | Holm p   |
|-----:|--------:|----------------:|:------------------|---------:|
| 0.10 | 0.969   | **+0.061**      | [+0.007, +0.115]  | 0.340    |
| 0.25 | 0.884   | -0.024          | [-0.101, +0.046]  | 1.000    |
| 0.50 | 0.923   | +0.015          | [-0.054, +0.085]  | 1.000    |
| 0.75 | 0.953   | +0.046          | [-0.016, +0.107]  | 0.663    |
| 1.00 | 0.961   | +0.053          | [-0.009, +0.108]  | 0.521    |

### Plain reading

- The point-estimate optimum is `ff=0.50` on slippery (+5.1 pp) and
  `ff=0.10` on rough (+6.1 pp).
- **No** ff cell is significant after Holm correction at alpha=0.05
  on either terrain.
- The largest pre-correction signal is `ff=0.10` on rough
  (raw Fisher p=0.068, Holm p=0.340).
- Cl-on-difference for the slippery optimum is `[-0.017, +0.118]`:
  it just clears zero at 1-sigma, not at 2-sigma. The point estimate
  is real; the certainty is not.

The honest one-line summary: **the v0.3.0 headline (`ff=0.5`
slippery optimum, `+5.1 pp` lift) is a credible best-bet starting
point for the next ablation, but is not statistically distinguishable
from the no-curriculum control at alpha=0.05 with single-seed, n~130
per cell**. Anyone reading the v0.3.0 result without this CI table
will overstate the evidence.

## 4. Per-failure-mode breakdown

The eval phase did not log per-episode failure-mode counts, so a
true eval-time breakdown is unavailable. The available proxy is the
**curriculum-input pool composition**:

| mode             | n_traj | n_active_steps | active_share |
|:-----------------|-------:|---------------:|-------------:|
| command_mismatch | 3      | 90             | 0.337        |
| slip             | 3      | 60             | 0.225        |
| attitude         | 3      | 33             | 0.124        |
| contact_loss     | 3      | 30             | 0.112        |
| stumble          | 3      | 30             | 0.112        |
| collapse         | 3      | 24             | 0.090        |

The pool is balanced 3-trajectories-per-mode but skewed in active
failure-flag duration toward `command_mismatch` (0.34) and `slip`
(0.22). This composition is identical across every ff cell; only
the per-minibatch sampling rate varies. Therefore any lift the
curriculum delivers is disproportionately attributable to those two
channels by step-count exposure. Confirming or refuting that
attribution is the explicit purpose of the next ablation (see
section 6).

## 5. Limitations

1. **Single seed.** Every cell uses `seed=42`. Cross-seed variance
   is the dominant source of uncertainty in PPO fine-tuning; we
   have measured zero of it. A reviewer would be right to demand
   3-5 seeds per cell before treating any point estimate as
   load-bearing. The right next-step rebuild of this sweep is at
   ff in {0.0, 0.5}, three seeds each, before adding the
   mode-subset axis.
2. **Sim-only evaluation.** Both terrains are Isaac Lab procedural
   environments. There is no real-hardware corroboration for any
   slippery or rough number in this report. The CaresLab GO2 has
   not seen any of these checkpoints.
3. **No per-episode mode logging at eval time.** The breakdown in
   section 4 is curriculum-input, not eval-time. The mode-subset
   ablation in section 6 will partially address this by varying
   the input axis directly.
4. **n is approximate.** Each cell is configured for 128 episodes
   but reports 128 to 140 terminations depending on episode length.
   We use the actual `num_episodes` per cell for all CIs.
5. **Bootstrap-on-indicators carries no extra information beyond
   the binomial summary.** It is reported for shape and to enable
   permutation tests, not as additional rigor.
6. **Curriculum dosage interpretation is a step-count model, not a
   minibatch-frequency model.** The pool's active-step skew matters
   only if the buffer samples are step-weighted; if they are
   trajectory-weighted, exposure is uniform across modes. We use
   the step-count model because Phoenix's ReplayBuffer samples
   timesteps, not whole trajectories.

## 6. Next ablation: mode-subset sweep

To test which failure modes carry the lift, fix `ff=0.5` (the
point-estimate slippery optimum) and sweep over six failure-mode
subsets:

| subset                                         | hypothesis                              |
|:-----------------------------------------------|:----------------------------------------|
| `[]` (all-modes; equivalent to v0.3.0 ff=0.5)  | reproduces ff=0.5 result                |
| `[slip]`                                       | slip alone explains slippery lift       |
| `[command_mismatch]`                           | tracking-error replay alone explains    |
| `[slip, command_mismatch]`                     | step-count majority is sufficient       |
| `[attitude, collapse]`                         | severe modes alone are insufficient     |
| `[slip, attitude, collapse]`                   | severe + slip is the right minimal set  |

Configs and a wrapper script live under
`configs/ablations/failure_modes/` and
`scripts/run_failure_modes_ablation.sh`. They are scaffolded but
**not run** in this session; the GPU is reserved for Zeus. Yusuf
will trigger the run when free.

Expected outcome: if the ff=0.5 lift is real, at least one of the
single-mode or two-mode cells should reproduce it. If none do, the
v0.3.0 finding is an interaction effect across modes, which would
be useful information either way.

When this sweep runs, the eval logger should be patched to retain
per-episode failure-mode counts so the eval-time breakdown becomes
available for the v0.4.0 report. That patch is itemized in
`docs/experiments/2026-05-07-failure-mode-ablation-plan.md`.
