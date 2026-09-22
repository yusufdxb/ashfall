# Failure-Boundary Replay (FBR)

> **Archived 2026-09-22.** Historical design record of study 1 (FBR), kept unchanged below this
> note. FBR returned NO-GO at its preregistered toy gate, so nothing described here as planned,
> pending or NOT YET TESTED was run, and none of it will be. The hardware plan it cites
> (`HARDWARE_DEMO.md`) was never run and is not part of the public archive. Final status:
> [`docs/ARCHIVE.md`](../ARCHIVE.md).

## 1. The method in one paragraph

A deployed locomotion policy fails on the robot. Ashfall cuts a short **failure capsule** out of
the log around the onset: the pre-failure state and action history, the command, the policy
hash, the onset time and the predeclared criterion that labelled it. In simulation it restores
that pre-failure state at a few predeclared lead times, checks that some friction values
reproduce the failure (three checks, section 3), and, for each restored state, locates the
friction at which the current policy's failure probability crosses one half. The fine-tuning
budget is then spent around those boundaries: most resets restore a capsule state with friction
drawn from a small band around its boundary, the rest are ordinary nominal resets. The boundary
is re-estimated as the policy improves. The comparison that matters is against broad friction
randomisation and against the same machinery seeded from the baseline's own successful states,
all at equal compute.

## 2. The failure capsule

The capsule is `ashfall.capsule.FailureCapsule`; FBR adds a contract on top of it
(`ashfall.fbr.capsule`) rather than a new type.

| field | why FBR needs it | source on hardware |
|---|---|---|
| `checkpoint_sha256` | the boundary is a property of one policy | deployment manifest |
| `frames[*].timestamp_s` | onset and lead times are in seconds | bridge log clock |
| `base_quat`, `base_ang_vel_body` | the failure criterion; restore | IMU |
| `joint_pos`, `joint_vel`, `action` | restore; gait phase lives here | `/lowstate`, policy output |
| `command_vel` | restore; the command is held during replay | bridge log |
| `base_lin_vel_body` | restore | estimate only on the GO2 (null when not fresh) |
| `base_pos` | restore, and the patch geometry relative to the robot | external: tape marks and video |
| `failure_onset_index`, `pre_failure_start_index` | the window | criterion (`fbr.criterion.detect_onset`) |
| `event_descriptor.criterion`, `criterion_id` | a replay is judged by the same criterion | written by `extract_capsule` |
| `surface_terrain_metadata` | patch position and length (known); friction left null unless measured | lab record |
| `provenance` | revisions, seeds, versions | evidence bundle |

What the capsule is not: a universal record for every robot failure. It is exactly what the
first study needs. Channels the GO2 does not measure stay null and are handled by the
simulator reset as declared assumptions (`HARDWARE_DEMO.md`).

**Onset.** `detect_onset` returns the first frame of the first run of `debounce_frames`
consecutive frames whose combined roll and pitch tilt exceeds `max_tilt_rad`, or the first
explicit terminal flag (simulator termination; on hardware an operator catch or e-stop),
whichever is earlier. An optional stall clause needs a velocity estimate and refuses to run
without one.

**Seed states.** `seed_points` returns one frame per predeclared lead time before onset. A seed
is refused if it is outside the reviewed window, more than `max_lead_s` (1.0 s) before onset,
missing a reset channel, physically implausible, or already failing under the criterion. Row 0
receives no exemption: that is the Phase-I defect (`ASHFALL_SIMPLIFICATION_AUDIT.md` section 3).

## 3. When a replay counts as a reproduction

`ashfall.fbr.acceptance`. From one restored seed state, a treated replay (the candidate
friction) and a matched control (full friction) share the state, command, policy and simulator
seed. A pair passes when all three hold:

1. **delivered**: the friction was read back from the simulator (`InterventionReceipt.verified`);
2. **trajectory match**: the treated replay's scaled RMS distance to the recording, from the seed
   to the recorded onset, is at most `(1 - margin)` times the control's (margin 0.2);
3. **failure match**: the treated replay meets the capsule's criterion within 0.3 s of the
   recorded onset, and the control does not within that window.

A friction value is REPRODUCED when at least two thirds of its replicate pairs pass. A capsule
with no reproduced value is UNREPRODUCED and is never trained on. A failure that the matched
full-friction control also produces (for example a fall caused by the state alone) is not a
friction failure and is rejected by check 3.

These are the v3 D1/D2/D3 gates restated against the observed failure. The stricter D2
(Mahalanobis departure against nominal replicates under Ledoit-Wolf shrinkage) remains in
`ashfall.gates` as an appendix-level variant.

## 4. Why boundary replay should help, and how strongly that can be claimed

### 4.1 An elementary bound

Let `xi` be a training context (state, command, friction) and `R in {0, 1}` the success of one
rollout from it under policy `pi_theta`, with success probability `p(xi; theta)`. The
score-function identity gives

    grad_theta p(xi; theta) = E[(R - p) * grad_theta log pi_theta(tau)]

and Cauchy-Schwarz gives

    || grad_theta p(xi; theta) || <= sqrt(p (1 - p)) * sqrt(E || grad_theta log pi_theta(tau) ||^2).

A context the policy always solves (`p = 1`) or always fails (`p = 0`) contributes nothing to the
gradient of its own success probability, whatever the score norm. For a training distribution
`q` over contexts, the **boundary mass**

    beta(q; theta) = E_{xi ~ q} [ sqrt(p(xi; theta) (1 - p(xi; theta))) ]

bounds (up to the score-norm factor) how much of the success-probability gradient `q` can
supply. **Boundary dilution** is the condition that `beta(q)` is small for the broad
distribution a practitioner would otherwise use, because most of its contexts are far from the
failure. This is the same intuition as goals of intermediate difficulty (Florensa et al.) and
learning-progress curricula; it is stated here because it gives a quantity that can be measured
for each arm's training distribution.

What the bound does not say. It concerns the gradient of binary success only. PPO on a shaped
locomotion reward also receives gradient from rollouts whose outcome is certain (a survived
episode still carries a tracking reward, a failed one still carries a survival-time signal).
And a larger gradient in the boundary region does not imply better performance on held-out
conditions: concentrating training can overfit the neighbourhood or degrade nominal behaviour.
The bound motivates FBR. It does not establish that FBR works.

### 4.2 The toy measurement: the slip cart-pole

`ashfall.fbr.toy_slip`, run by `python -m ashfall.fbr.toy_study`. A pole balances on a cart that
must track a commanded speed; a periodic torque stands in for gait excitation (its phase is part
of the state); the drive force is traction-limited, `|F| <= mu(x) (M + m) g`, and saturates at the
kinetic limit when the request exceeds it (a slip); a 1 m patch of low friction lies on the
path. The baseline is trained on patch friction in [0.35, 0.8]. A hidden deployment world has
patch friction 0.08. The capsule is extracted from the first deployment failure that the
acceptance checks reproduce, using the real `extract_capsule`, `seed_points` and
`accept_reconstruction`, with sensor noise on the recorded channels and the friction left out.
All arms train with the same evolution-strategies optimiser, the same 92,160 rollout slots of
equal horizon, and the same 12 training seeds. Boundary estimation and refresh for arms C and D add
192,000 simulator steps outside that budget (1.19% of the 16.1M training steps); arm B has none.
This is disclosed, not equalised, in the toy; the GO2 protocol counts and reports it per arm. Evidence kind `toy_mechanism`; nothing here is a
GO2 result.

The toy's physics constants were calibrated on the baseline alone (the untrained controller and
the trained baseline, never an arm) until the baseline succeeded in its training range and
failed at low friction; every calibration attempt is listed in section 4.4. The arm comparison
was run once, at commit `f339087`, and is committed at `results/fbr_toy/f339087/result.json`.

**Reconstruction.** The capsule came from the second deployment trial (no earlier failure was
rejected). Friction values from 0.04 to 0.24 were REPRODUCED; the hidden value 0.08 is inside
that range. The range is wide: several frictions produce the same failure from the same state,
which is the identifiability limit the hardware study will also face.

**Boundary mass at the baseline** (256 sampled training contexts, 16 replicates each):

| training distribution | boundary mass | contexts with p in (0.1, 0.9) | always succeeds | always fails |
|---|---|---|---|---|
| B broad DR, friction U[0.05, 0.8] | 0.115 | 23% | 70% | 0% |
| C nominal-seed boundary replay | 0.294 | 61% | 26% | 2% |
| D FBR | 0.275 | 58% | 30% | 3% |

The mechanism is present: broad DR spent 70% of its contexts where the baseline never fails,
and boundary replay concentrated 2.4 (FBR) to 2.6 (nominal-seed) times more boundary mass into the same budget.

**Learning outcome** (mean over 12 seeds, final parameters, 256 episodes per cell with common
random numbers):

| arm | held-out failure (6 cells) | deployment condition | nominal success | tracking RMSE (m/s) |
|---|---|---|---|---|
| A no repair | 50.6% | 57.0% | 97.3% | 0.513 |
| B broad DR | 35.3% | 33.2% | 96.2% | 0.515 |
| C nominal-seed boundary | 36.9% | 32.9% | 94.7% | 0.520 |
| D FBR | 35.6% | 31.5% | 90.4% | 0.545 |
| E friction band, nominal states | 46.4% | 47.9% | 96.8% | 0.508 |

Paired by seed (exact sign-flip test, 95% t interval):

| comparison | mean (D minus comparator) | 95% CI | exact p |
|---|---|---|---|
| **D vs B, held-out failure (primary)** | **+0.3 pp** | **-4.7 to +5.3** | **0.898** |
| D vs C, held-out failure (fixed sequence: not tested, descriptive) | -1.2 pp | -6.6 to +4.1 | 0.622 |
| D vs E, held-out failure | -10.8 pp | -16.5 to -5.1 | 0.0024 |
| D vs A, held-out failure | -15.0 pp | -18.7 to -11.2 | 0.0005 |
| D vs A, nominal success | -6.8 pp | -8.8 to -4.8 | 0.0005 |
| D vs A, tracking RMSE | +0.032 m/s | +0.025 to +0.039 | 0.0005 |

**Reading, in the terms of the preregistered interpretation table.** FBR is not better than
broad DR on held-out failure at this budget, and it fails the 2 pp nominal non-degradation
margin. Under the toy's version of the protocol this is a null on the primary and a failed
non-degradation condition. Four observations, all descriptive:

1. More boundary mass did not buy better held-out performance. The bound in 4.1 is a necessary
   condition on where gradient can come from, and the toy shows it is not sufficient.
2. Against broad DR, FBR was better on three held-out cells (harder friction 45.3 vs 49.1%,
   nearer patch 27.5 vs 32.6%, faster command 16.5 vs 21.1%) and worse on three (easier
   friction 18.5 vs 16.8%, farther patch 41.6 vs 34.4%, slower command 64.4 vs 58.0%), and the
   pooled difference is zero. That is consistent with a local repair that does not travel
   (three seeds from one capsule cover one gait phase, while held-out cells enter the patch at
   uncontrolled phases), but the toy does not test that explanation.
3. The capsule's seed boundaries sat at friction 0.26 and 0.40, well above the deployment
   value, and the 0.2 s seed was dropped because its boundary was not identified (the state was
   already doomed at every friction). Seeds close to onset are near-doomed states; FBR spent
   much of its budget on late recoveries rather than on the patch-entry transition.
4. Taking the failure's friction band without its state (arm E) was clearly worse than both,
   so in this toy the value of FBR's samples came from the restored states, not from knowing
   the friction.

These are the toy's findings. They are a pilot for the GO2 design, and they changed it
(`EXPERIMENT.md` section 3): lead times are now chosen by a predeclared rule on the baseline
alone so that seeds include the approach before patch entry, and several capsules are used
rather than one.

**Correction (commit `f3ccc1b`).** The run above was produced with a defect: the toy's
boundary refresh parsed the censoring reason out of an error message, got the seed row instead
(seed ids contain colons), and so never clamped a seed that had become robust. Rerun with the
fix (`results/fbr_toy/f3ccc1b_corrected/result.json`, same specification hash): arms A, B, D and
E are identical to the last digit (no FBR seed ever became robust, and the retrained baseline
reproduced exactly); arm C changed from 36.9% to 35.1% held-out failure and from 94.7% to 96.0%
nominal success. D minus B is unchanged (+0.3 pp, p 0.898); D minus C becomes +0.5 pp
(95% CI -3.5 to +4.5, p 0.783). The dropped 0.2 s FBR seed is now recorded as `below_support`
(doomed at every friction). A review of the artifacts also found that all three FBR seeds were
already 0.34 to 0.74 s inside the patch when they were taken (entry at row 91, seeds at rows 108
to 128): FBR trained on recoveries from states already slipping, with a band (about 0.13 to 0.45)
that never contained the deployment friction 0.08.

### 4.3 Exploratory: budget curves

Added after the primary result and labelled exploratory: whether FBR reaches a given held-out
failure with fewer rollouts even though it ends level with broad DR. The run re-trains B, C and
D from the same inputs, keeps snapshots at 25, 50 and 75% of the budget, and checks that the
final parameters reproduce the primary run exactly. Output:
`results/fbr_toy/f339087_efficiency_exploratory/efficiency.json`. Results are in section 4.5.

### 4.4 Calibration log (baseline only, before any arm)

| attempt | change | baseline outcome | action |
|---|---|---|---|
| 1 | hand-tuned gains | every rollout failed (sign error in the feedback) | replaced by an LQR gain |
| 2 | LQR, excitation 4, patch 0.5 m | baseline robust at every friction (no boundary) | patch lengthened |
| 3 | patch 1.0 and 1.5 m, excitation 3 to 5 | failures only below friction 0.1 | stronger excitation |
| 4 | tracking weight 2 and 4 | unchanged | weight left at 0.5 |
| 5 | excitation 8 and 11, noise 0.8 and 2.5 | excitation 11, noise 2.5: baseline 3 to 7% failure in its range, 24% at 0.15, 46% at 0.10, 69% at 0.05 | frozen |

### 4.5 Exploratory budget-curve result

Exploratory, not preregistered. Mean held-out failure over 12 seeds at each fraction of the
92,160-slot budget (768 slots per iteration), and the paired D minus B difference:

| budget | B broad DR | C nominal-seed | D FBR | D minus B (95% CI), exact p |
|---|---|---|---|---|
| 25% (iteration 30) | 37.2% | 37.3% | 36.0% | -1.2 pp (-6.2 to +3.8), 0.618 |
| 50% (60) | 36.0% | 38.9% | 36.6% | +0.6 pp (-4.5 to +5.6), 0.807 |
| 75% (90) | 34.5% | 39.1% | 35.9% | +1.4 pp (-4.5 to +7.3), 0.616 |
| 100% (120) | 35.3% | 36.9% | 35.6% | +0.3 pp (-4.7 to +5.3), 0.898 |

Retraining from the saved baseline reproduced every final evaluation of the primary run exactly,
for every arm and seed. There is no budget at which FBR is ahead of broad DR in
this toy: both arms take most of their gain (from 50.6% for the baseline) within the first
quarter of the budget, and the rest is flat. The toy therefore does not support "more
efficiently" either.

### 4.6 Toy v2: the preregistered decision gate (result: NO-GO)

Registered in `TOY_V2_PREREGISTRATION.md` and committed with its code at `8f49f02` before it
ran; run once; `results/fbr_toy_v2/8f49f02/result.json`. Six hidden deployment worlds, one
capsule each, 12 training seeds, the minimal FBR seeded at patch entry, a real ADR arm, and the
out-of-training simulation of C and D charged to B and ADR (per-seed simulator steps: B and ADR
16,262,400; D at most 16,215,760; C 16,144,175).

| arm | held-out failure | deployment condition | nominal success | tracking RMSE (m/s) |
|---|---|---|---|---|
| A no repair | 48.0% | 48.6% | 97.3% | 0.513 |
| B broad DR | 35.1% | 33.9% | 96.3% | 0.515 |
| ADR | 39.1% | 38.0% | 96.1% | 0.512 |
| C entry (nominal seed) | 28.8% | 26.3% | 91.9% | 0.550 |
| D FBR entry | 32.4% | 29.4% | 90.7% | 0.550 |

Per seed, averaged over the six worlds (exact sign-flip test, 95% t interval):

| comparison | mean | 95% CI | exact p |
|---|---|---|---|
| D vs ADR, held-out (primary, part 1) | -6.7 pp | -9.4 to -4.1 | 0.0010 |
| D vs B, held-out (primary, part 2) | -2.7 pp | -6.1 to +0.6 | 0.105 |
| D vs C, held-out (descriptive) | **+3.6 pp** | +1.5 to +5.7 | 0.0029 |
| ADR vs B, held-out (descriptive) | +4.0 pp | +1.2 to +6.9 | 0.014 |
| D vs A, nominal success (90% interval) | -6.6 pp | -8.4 to -4.7 | 0.0005 |
| D vs A, tracking RMSE (90% interval) | +0.038 m/s | +0.032 to +0.043 | 0.0005 |

**Decision under the registered rule: NO-GO.** The intersection-union primary does not reject
(D is better than ADR but not significantly better than broad DR), and non-degradation fails.
The GO2 FBR study is not run as designed.

What the gate showed, descriptively:

1. **The real failure did not help; it hurt.** Replay seeded from a nominal rollout's patch
   entry (C) beat replay seeded from the failure's patch entry (D) by 3.6 pp. In the setting most
   favourable to FBR, the failure's own state carried no value beyond "a state at the patch
   entrance". This is the answer to the research question in this toy, and it is no.
2. **Replay at patch entry does move held-out failure.** Both replay arms beat broad DR in
   mean (C by 6.3 pp, D by 2.7 pp) and beat ADR, which in this toy did worse than broad DR.
   Entry anchoring fixed v1's mis-seeding: the friction boundaries from entry seeds (0.06 to 0.12)
   now sit at the deployment frictions (0.06 to 0.12).
3. **Both replay arms cost nominal performance** (about 5 to 7 pp of nominal success and 0.04 m/s
   of tracking), even with half of their resets drawn from broad DR. The degradation is a
   property of replaying one local state, not of where the state came from.
4. **Reconstruction from the entry seed identified friction poorly.** The acceptance checks
   reproduced the failure mostly at the lowest grid friction (0.04); the hidden friction was
   inside the reproduced range in 1 of 6 worlds. From 0.7 to 0.9 s before onset, the failure
   is not specific enough to pin down the friction.
5. **ADR as implemented here** (lower bound starts at 0.35, steps of 0.03 every 10 iterations)
   never caught up with broad DR in 120 iterations. Its thresholds were fixed before the run and
   were not tuned; a tuned ADR could do better, which would not change the D vs C finding.

## 5. The training distribution

`ashfall.fbr.boundary.BoundarySampler`, one class for arms C and D:

- with probability `nominal_fraction` (0.3) a draw is an ordinary nominal reset from the
  baseline's training distribution;
- otherwise a boundary draw: a seed state chosen uniformly, friction uniform on
  `[mu_lo - w, mu_hi + w]` around that seed's identified boundary interval (`w = 0.05`, clipped
  to the declared support), a command jitter of 0.05 m/s, and backend-declared state noise;
- every `refresh_every` iterations the boundaries are re-estimated with the current policy;
  a seed that becomes robust over the whole support is clamped to the hardest friction, a seed
  that becomes doomed at every friction is dropped.

Arm C builds the same sampler from seeds drawn out of the baseline's successful nominal
rollouts, at the same timing relative to patch entry as the capsule's seeds. Arm B uses
`BroadSampler` (friction uniform on the broad range, nominal resets). No arm inserts recorded
transitions into PPO; the only thing that differs between arms is the reset distribution.

## 6. What is implemented, and what is not

| piece | status |
|---|---|
| criterion, onset, capsule contract, seed refusal | VERIFIED on CPU (`tests/test_fbr.py`) |
| boundary estimation, samplers, boundary mass | VERIFIED on CPU |
| acceptance checks | VERIFIED on CPU; exercised end to end in the toy |
| toy study | VERIFIED (run once, committed) |
| Isaac Lab FBR training | NOT YET TESTED: the reset plumbing exists (`ashfall.training.run_repair`, Phoenix `install_scenario_reset`) but a `BoundarySampler` to `Scenario` adapter, a spatial patch (per-foot friction switching on entry and exit), and boundary refresh inside the PPO loop are not written |
| hardware capsule extraction | NOT YET TESTED: no GO2 walking policy is hardware-validated yet (Phoenix `deploy_contract.py`) |
