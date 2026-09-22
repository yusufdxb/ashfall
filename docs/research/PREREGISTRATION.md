# Preregistration: Failure-Boundary Replay, first study

Status: **REGISTERED, NOT RUN, GATED OFF** (toy v2 returned NO-GO under amendment A1, item 7). No GO2 simulation or hardware result exists for this protocol.
This document replaces the Phase-II H0 to H4 preregistration, which is archived unchanged at
[`docs/legacy/phase2/PREREGISTRATION.md`](../legacy/phase2/PREREGISTRATION.md). Explanations
are in [`EXPERIMENT.md`](EXPERIMENT.md); this page is the contract.

Values are of three kinds: **FIXED** (set now), **BY RULE** (set in experiment E0 by the rule
written here, on the baseline and the development world only, then frozen by amendment before
E1), and **MEASURED** (recorded, not chosen). Nothing is set by looking at an arm comparison in
the deployment world.

## 1. Question

Can a real quadruped failure define a local simulation curriculum near the observed failure
boundary that improves robustness more efficiently than broad domain randomization?

## 2. Hypotheses

| id | statement | role |
|---|---|---|
| P | Under equal fine-tuning compute, FBR (arm D) has lower held-out traverse failure than broad friction DR (arm B). | primary, confirmatory |
| S | FBR (D) has lower held-out traverse failure than nominal-seed boundary replay (C). | key secondary, tested only if P rejects with D better (fixed sequence, same alpha) |
| N | FBR (D) stays within the nominal margins against the baseline (A). | condition on any repair claim |

Secondary hypotheses about individual held-out dimensions are tested only after P rejects, with
Holm correction. Everything else is descriptive.

## 3. Arms (FIXED)

A no repair; B broad DR; C nominal-seed boundary replay; D FBR; E friction band from D's
initial boundaries with nominal states (ablation, not in P, S or N). All trained arms start from
one baseline checkpoint and receive identical PPO iterations, environments, steps per
environment, epochs and minibatches, checked by `assert_equal_compute` on each run's
`TrainingArtifact`. Simulator rollouts spent outside PPO (reconstruction, boundary estimation
and refresh) are counted and reported per arm; the arms whose sampler needs them (C, D) receive
the same number of such rollouts, and B receives none.

## 4. Endpoint (FIXED)

Held-out traverse failure rate: per (arm, training seed), the equal-weight mean over six
held-out cells of the fraction of 512 episodes that terminate or fail to reach the far line
within the time limit, evaluated in the deployment world, final checkpoint, with evaluation
scenario seeds identical across arms within a training seed.

## 5. Tests (FIXED)

- P and S: exact two-sided sign-flip permutation test on per-seed paired differences
  (`ashfall.stats.paired_seed_effect`), alpha 0.05; report the mean difference in percentage
  points, the 95% t interval, the test-inverted interval, Hedges g_z and the p floor.
- N: one-sided 95% bounds across seeds; nominal success drop at most **2 pp**, tracking RMSE
  increase at most **0.05 m/s**; both must pass; a zero-width interval falls back to the exact
  Clopper-Pearson bound; a degenerate continuous interval fails closed.
- Minimal practically important effect for P: **10 pp**.

## 6. Seeds (FIXED)

n = 12 training seeds: 11, 23, 47, 59, 71, 83, 97, 101, 113, 127, 131, 149, identical in every arm.
No optional stopping; no seed added or dropped after any evaluation; every seed reported. A run
that crashes is rerun with the same seed; a run that completes is data.

## 7. Worlds and held-out cells

- Deployment world dynamics offsets: motor strength 0.95, payload +1.0 kg (FIXED).
- Confirmatory patch placement, length, friction and command: BY RULE, chosen in the development
  world pilot so that the baseline's failure on the patch is between 30 and 70% and then moved
  to a placement not used in development.
- Six held-out cells (friction one step easier and harder, patch nearer and farther, command
  slower and faster): BY RULE, each chosen on arm A alone so that A fails 25 to 85%, using a
  calibration seed list disjoint from the evaluation seed list. The held-out manifest is hashed
  and recorded in `selection.HeldOutLedger` before any fine-tune.

## 8. FBR constants

- Failure criterion: combined tilt above 0.40 rad (Phoenix's software attitude intervention
  threshold) for 3 consecutive policy steps, or a terminal flag (simulator termination; on
  hardware an operator catch or e-stop), identical in simulation and on the robot (FIXED).
- Acceptance: trajectory margin 0.2, onset tolerance 0.3 s, pass fraction 2/3 over 6 replicate
  pairs per friction value (FIXED).
- Capsule set: the first three deployment failures that pass acceptance, in trial order (FIXED).
- Lead times: the three smallest of {0.3, 0.6, 0.9, 1.2} s whose boundaries are identified, one
  of which must precede patch entry; `max_lead_s` 1.5 s (BY RULE).
- Band half-width `w` in {0.05, 0.1} and nominal fraction in {0.3, 0.5}: the combination with the
  lowest development-world patch failure among those inside the nominal margins in two
  development seeds (BY RULE).
- Boundary refresh every 25% of iterations (FIXED).
- Broad DR range for B: the baseline's friction range extended down to the lowest friction on
  any held-out cell (BY RULE, recorded).

## 9. Kill criteria

- P not rejected and the 95% upper bound of the D minus B improvement below 10 pp: the hypothesis
  is falsified at the minimal effect, and the paper reports that.
- P rejected with D worse: falsified.
- N fails: no repair claim is made.
- Any compute violation: the comparison is discarded and rerun, never reported with a caveat.
- No capsule passes acceptance: the study stops; the result is a reconstruction failure, not a
  repair result.

## 10. Hardware (FIXED design, run only after Phoenix's walking gate)

One D and one B policy, from a training seed declared before training, 20 trials each on the
original patch in randomised blocks. The only confirmatory hardware test is a one-sided
Boschloo exact test on traverse success, alpha 0.05. Moved-patch and changed-severity trials are
descriptive. Every trial is kept.

## 11. Amendments

- **A0 (registered here):** the toy pilot (`results/fbr_toy/f339087/`) found FBR level with
  broad DR and outside the nominal margin; the lead-time rule, the three-capsule set and the
  development-world choice of `w` and nominal fraction were added in response, before any GO2
  run.
- **A1 (registered 2026-09-22, before any GO2 run and before toy v2 was run):** after a hostile
  review. (1) An ADR arm is added; hypothesis P becomes intersection-union: D beats both B and
  ADR, each at alpha 0.05. S (D vs C) stays the fixed-sequence secondary. (2) Out-of-training
  simulator steps are counted; B and ADR get extra iterations until their totals are at least
  the largest C or D total. (3) K = 6 deployment worlds, one capsule each; per-seed effects are
  averaged over worlds; the three-capsule single-world set of A0 is withdrawn. (4) FBR is the
  minimal form: seed at patch entry, one boundary estimate, one-sided band [0.02, mu_b + w] with
  w = 0.05, replay share 0.5 over broad DR, sensor-noise resets; the lead-time rule, refresh and
  the development-world tuning of A0 are withdrawn. (5) Crossing time is added to N with a margin
  of +10% of the baseline's median. (6) Hardware (section 10) is a demonstration unless 3 seeds
  per arm are deployed. (7) The GO2 study runs only if toy v2 returns GO.
- The next amendment records the BY RULE values from E0, with the evidence bundle that produced
  them, and freezes them before E1.
