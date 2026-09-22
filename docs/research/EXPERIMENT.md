# The FBR experiment

> **Archived 2026-09-22.** Historical design record of study 1 (FBR), kept unchanged below this
> note. FBR returned NO-GO at its preregistered toy gate, so nothing described here as planned,
> pending or NOT YET TESTED was run, and none of it will be. The hardware plan it cites
> (`HARDWARE_DEMO.md`) was never run and is not part of the public archive. Final status:
> [`docs/ARCHIVE.md`](../ARCHIVE.md).

> **Gate: CLOSED.** Nothing below runs on the GO2, in simulation or on hardware, unless the
> preregistered toy v2 ([`TOY_V2_PREREGISTRATION.md`](TOY_V2_PREREGISTRATION.md)) returns GO.
> It returned **NO-GO** on 2026-09-22 (`results/fbr_toy_v2/8f49f02/`, `FBR_METHOD.md` section
> 4.6). This page is kept as the design that would run if a revised method passed a new gate.
> The design was revised after a hostile review (section 8); where this page and the first
> preregistration disagree, amendment A1 in `PREREGISTRATION.md` governs.

One failure family (low-friction traverse failure: slip is the physical mechanism, but the
endpoint is traversal, not detected slip), one robot (Unitree GO2), one
physical axis (patch friction), one primary endpoint. The confirmatory protocol is frozen in
[`PREREGISTRATION.md`](PREREGISTRATION.md); this document explains it. Nothing below has run on
Isaac Lab or on the robot. The CPU toy that piloted it is in [`FBR_METHOD.md`](FBR_METHOD.md)
section 4.

## 1. Worlds

| world | what it is | used for |
|---|---|---|
| **training world** | Phoenix's flat GO2 task (Isaac Lab), the environment the baseline was trained in | all fine-tuning |
| **development world** | training world plus a low-friction patch at a development placement, friction and speed, with the dynamics offsets below | choosing FBR's hyperparameters (lead times, band width, nominal fraction) on the baseline and in pilot fine-tunes. Never used for any confirmatory number |
| **deployment world** | training world plus the confirmatory patch, and dynamics offsets the training world does not have (motor strength scaled 0.95, +1.0 kg payload), frozen before any fine-tune | the proxy for reality: failures are collected here, capsules are extracted from its logs as hardware logs would be, and held-out cells are evaluated here |

The dynamics offsets exist so that reconstruction is not trivially exact: the training world
that FBR replays in is not the world that failed. The patch is spatial: each foot's friction
switches when that foot's world x position enters or leaves the patch (event-driven material
writes with readback). The robot does not observe friction.

## 2. Arms

All trained arms start from one baseline checkpoint and use identical PPO iterations,
environments, steps per environment, epochs and minibatches, so environment steps and optimiser
updates are equal by construction; `ashfall.protocol.budget.assert_equal_compute` refuses any
comparison where the recorded `TrainingArtifact`s disagree. The arms differ only in the reset
distribution.

| arm | reset distribution | question it answers |
|---|---|---|
| A no repair | none (baseline evaluated) | how much there is to repair |
| B broad DR | nominal resets, patch friction uniform over the broad range | the standard alternative |
| ADR | nominal resets, patch friction uniform on [lo, hi]; lo widened when success at lo is high, narrowed when low (probes counted) | the standard simulator-found boundary (added in A1) |
| C nominal-seed boundary replay | the minimal FBR sampler seeded from a nominal rollout's patch-entry state | does the failure's own entry state matter? |
| D FBR | the minimal FBR sampler seeded from each capsule's patch-entry state | the method |
| E friction band (ablation) | nominal resets, patch friction drawn from D's initial boundary band | is it only the friction value (SimOpt-style reading)? |

E is an ablation, not a comparator in the confirmatory tests. Out-of-training steps (seed
search, reconstruction, boundary estimation, ADR probes) are counted; B and ADR receive extra
PPO iterations so their totals are at least the largest C or D total (A1).

## 3. What the toy pilot changed (first response; items 1 to 3 withdrawn by A1, see section 8)

The slip cart-pole (12 seeds, `results/fbr_toy/f339087/`) found FBR level with broad DR on
held-out failure and outside the nominal margin, with FBR's seed boundaries far above the
deployment friction because seeds 0.2 to 0.6 s before onset are late, near-doomed states. The
GO2 design takes three lessons from it, all fixed before any GO2 fine-tune:

1. **Lead times by rule, not by default.** Candidate leads are {0.3, 0.6, 0.9, 1.2} s. On the
   baseline alone, in the development world, the confirmatory set is the three smallest leads
   whose boundaries are identified inside the support, and it must include one lead whose seed
   is before patch entry. `max_lead_s` is 1.5 s for the GO2.
2. **Several capsules, not one.** The capsule set is the first three deployment failures that
   pass the acceptance checks, in trial order. This spreads the seed states over gait phases.
3. **Nominal fraction and band width chosen in the development world.** Candidates
   `nominal_fraction` in {0.3, 0.5} and `w` in {0.05, 0.1}; the rule picks the combination with
   the lowest development-world patch failure among those that stay inside the nominal margin
   in two development seeds. The confirmatory world is untouched by this choice.

## 4. Held-out conditions

Six cells in the deployment world, frozen (hashed) before training, each chosen on arm A alone
so that A fails between 25 and 85% (a cell A always passes or always fails cannot show repair):

| dimension | cells |
|---|---|
| h1 severity | patch friction one step easier and one step harder than the confirmatory value |
| h2 placement | patch moved nearer and farther along the path (a different approach distance and entry phase) |
| h3 command | commanded speed one step slower and one step faster, inside the gait envelope |

The confirmatory condition itself is reported separately as the repair target, never as
held-out. Every h1 friction lies inside B's randomisation range, so B is never scored outside
its own training support. The calibration of cells uses a calibration seed list disjoint from
the evaluation seed list.

## 5. Endpoint and statistics (from the statistics design memo)

- **Primary endpoint.** Held-out traverse failure rate: per (arm, training seed), the mean over
  the six cells, equal weights, of the fraction of 512 episodes that fail (termination or not
  reaching the far line within the time limit). Common random numbers: identical evaluation
  scenario seeds across arms within a training seed. Final checkpoint only, no checkpoint
  selection.
- **Primary hypothesis (A1).** D fails less than both B and ADR (intersection-union: each
  comparison rejected at alpha 0.05 with D better; no adjustment is needed). Originally: D fails
  less than B. Paired by training seed, exact two-sided
  sign-flip test (`ashfall.stats.paired_seed_effect`), alpha 0.05, with the p floor `2/2^n`
  reported. Effect: mean paired difference in percentage points with a 95% t interval, the
  test-inverted interval, and Hedges g_z. Minimal practically important effect: 10 pp.
- **Key secondary, fixed sequence.** D against C at the same alpha, tested only if the primary
  rejects with D better. Family-wise error stays at 0.05 without splitting alpha.
- **Non-degradation condition** (intersection-union, all must pass): against A, nominal
  success drop at most 2 pp, tracking RMSE increase at most 0.05 m/s and median crossing time
  increase at most 10% (A1), one-sided 95% bounds
  across seeds. A zero-width interval at ceiling is refused and replaced by the exact
  Clopper-Pearson bound (`ashfall.evaluation.paired`).
- **Seeds.** n = 12, the same integers in every arm. Monte Carlo power of the exact test: 0.88
  at a 10 pp effect with a 10 pp paired SD (planning value scaled from the Phase-I between-seed
  spread computed from the 44 committed per-seed files), 0.75 if the SD is 12 pp. D against C
  at a 5 pp effect has power 0.34; a null there will usually read "not distinguished", and the
  paper will say so. No optional stopping, no added seeds, every seed reported; a crashed run is
  rerun with the same seed, a completed bad run is data.
- **Secondaries (Holm within family, only after the primary rejects):** failure per held-out
  dimension. Descriptive: per-cell failure, attitude RMS, time to failure, training-world
  held-out failure, sample efficiency at 25/50/75/100% of iterations on a validation set,
  boundary mass of each arm's training distribution at the baseline.

## 6. What falsifies the hypothesis

| result | reading |
|---|---|
| D vs B rejects with D better, D vs C rejects with D better, non-degradation passes | full claim: in simulation, FBR is a better use of the budget than broad DR, and the deployment failure carried value beyond a simulator-found boundary |
| D vs B rejects, D vs C does not | boundary-local training helps; the deployment origin is not shown to matter. If the D vs C upper bound is below 5 pp: "attributable to local boundary training" |
| D vs B does not reject and its 95% upper bound is below 10 pp | **falsified at the minimal effect** |
| D vs B does not reject and the interval spans 0 and 10 pp | inconclusive; not rescued by adding seeds |
| D vs B rejects with D worse | falsified |
| non-degradation fails | no repair claim; "repair with degradation", magnitude reported first |

The toy result (a null on the primary and a failed non-degradation condition) would, at GO2
scale, be read as "inconclusive or falsified, and degrading". That outcome would be reported.

## 7. Sequence of experiments

**E0, first simulator experiment (plumbing, not a result).**
1. Spatial patch in Isaac Lab: per-foot friction switched on entry and exit, with readback on
   every write; unit check that a foot outside the patch keeps the ground friction.
2. Retrain the baseline on the current Phoenix environment builder (removes the April-versus-June
   environment drift recorded in `docs/limitations.md`), with the ordinary friction DR range.
3. In the development world, collect baseline trials until three failures pass the acceptance
   checks; confirm the reconstructed friction range and the matched controls.
4. Boundary estimation from the seed states; apply the lead-time rule.
5. One fine-tune each of B and D with one development seed, with `TrainingArtifact`s, to measure
   wall-clock per fine-tune and confirm equal budgets.
6. Choose nominal fraction and band width by the rule in section 3.

**E1, the multi-seed simulation study (confirmatory).** Freeze the deployment world, the
held-out manifest and every constant in `PREREGISTRATION.md`; collect the capsules in the
deployment world; train B, C, D, E for 12 seeds (48 fine-tunes); evaluate A to E once on the
held-out manifest through the held-out ledger; write one evidence bundle.

**E2, first GO2 experiment.** Only after Phoenix's walking gate is hardware-validated
(`HARDWARE_DEMO.md` section 1). Reference-surface trials, then baseline trials on the measured
patch until the first reviewed failures; extract capsules from the bridge log; reconstruct in
the training world and run FBR and broad DR from the same baseline with one seed declared before
training.

**E3, held-out physical robustness test.** Redeploy the declared D and B policies: 20 trials
each on the original patch in randomised blocks (the only confirmatory hardware test: one-sided
Boschloo exact test on traverse success), then the moved patch and the second surface or speed,
descriptive only. Every trial is kept, including aborts and operator catches.

## 8. Revisions after review (amendment A1)

A hostile-reviewer pass (CoRL framing) found problems that changed the design before any GO2
run. What changed and why:

| problem | change |
|---|---|
| "boundary-focused beats broad DR" is already expected, and arm C is an FBR ablation rather than the standard boundary method | a real **ADR arm** on patch friction (lower bound widened or narrowed by success at the bound). The primary becomes an intersection-union test: D must beat both B and ADR |
| boundary estimation, refresh and reconstruction spent simulator steps that B did not | all out-of-training steps are counted and B and ADR get extra PPO iterations until their totals are at least the largest C or D total |
| one deployment world, one failure, 12 seeds | K = 6 deployment worlds, one capsule each, seeds nested within worlds; per-seed effects averaged over worlds |
| seeds 0.2 to 0.6 s before onset were already slipping (toy v1: all three inside the patch) | the minimal FBR seeds at patch entry: the last frame before the patch, whose position comes from the lab layout |
| FBR had tuning knobs the baselines did not (refresh, band, nominal fraction, lead set) | the minimal FBR has none tuned: one boundary estimate, one-sided band [mu_min, mu_b + w], replay share 0.5 over broad DR, sensor-noise resets; the development-world tuning rule is withdrawn |
| a simulated deployment world does not test hardware origin | in simulation the claim is "deployment-originated"; the capsule receives only GO2-available channels with sensor noise; hardware origin is tested only on the robot |
| 20 trials per policy, one seed per arm, is weak confirmatory evidence | the hardware session is a **demonstration** with predeclared counts; if a hardware number is reported as a test, 3 seeds per arm are deployed |
| a slower, stiffer gait passes a traverse endpoint | crossing time joins nominal success and tracking in the non-degradation set; per-foot slip velocity and time on patch are reported in simulation |

Criticisms deliberately not acted on, because they add arms without moving the claim: SimOpt,
BayesSim or DROPO arms (with one scalar, a global refit is arm E), RMA or UP-OSI (complementary),
real-world fine-tuning (what FBR is meant to avoid), PLR or ACCEL (ADR is the canonical boundary
method on one continuous axis), a second failure family, stronger theory.
