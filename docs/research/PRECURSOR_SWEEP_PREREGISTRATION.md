# Preregistration: replay start time along a failed trajectory (CPU toy)

Status: **REGISTERED before running.** Committed together with its code
(`src/ashfall/fbr/time_sweep.py`); stage 1 is run once, stage 2 is run once and only if stage 1
licenses it. Evidence kind `toy_mechanism`. Nothing here is a GO2 result.

This is a **new hypothesis**. FBR failed its own gate (`docs/results/FBR_TOY_NEGATIVE_RESULT.md`);
that result stands and is not revised by anything below. No method is named in this document.
A name is chosen only if the stage-2 gate returns GO.

## Questions

- **Primary.** Does repair effectiveness vary systematically with where along an observed failed
  trajectory replay begins, measured relative to failure onset?
- **Secondary.** Is there a pre-failure interval that outperforms both failure-onset replay and
  ordinary entry-state replay, and does it survive the controls that would explain it away
  (a successful trajectory at the same place, a simulator-found failure, broad DR, ADR)?

Hypothesis under test, stated before data, **unverified**: at onset the recoverable set has
collapsed, so replay there teaches little; far before onset replay is no longer specific to the
failure; an interior interval may balance the two.

## Fixed inputs (unchanged from toy v2)

- Physics, policy class, optimiser: `ToySlipConfig()`, `ESConfig()` defaults.
- Baseline: the frozen toy v2 `baseline.npy`, SHA-256 `2bb2a3bf...e0f3`, checked at load.
- Failure criterion: `TOY_CRITERION` (|tilt| > 0.6 rad). **Failure onset** is the first frame at
  which it fires (the capsule's `failure_onset_index`); it is defined by the criterion alone,
  independently of any replay result.
- Replay mixture for every replay arm: replay share 0.5, the other half broad-DR draws on
  [0.05, 0.8]; replay friction uniform on [0.02, mu_b + 0.05]; reset noise = the capsule sensor
  noise (tilt 0.005 rad, pitch rate 0.02 rad/s, speed 0.02 m/s); no command jitter.
  These are the toy v2 values. They are **not** retuned, including to rescue the nominal margin.
- Evaluation: 6 held-out cells per world (toy v2 `held_out`), 256 episodes each, common random
  numbers; nominal success and tracking RMSE from the toy v1 `evaluate` nominal batch.

## Replay start times

For each failed trajectory with onset time `T`, the seed state is the last recorded frame whose
timestamp is at most `T - d`, for

`d` in {2.0, 1.5, 1.0, 0.75, 0.5, 0.25} s, plus **T0**.

**T0 is the last frame strictly before onset** (0.02 s before). The onset frame itself already
meets the criterion, and the existing capsule contract refuses such a seed as tautological.
**T+0.25 s is excluded**: in this toy the episode terminates at onset, so no post-onset state
exists to restore. Both choices are fixed here, before any sweep result.

Additional per-trajectory start states:

- **start**: the first recorded frame of the failed trajectory (episode start);
- **entry**: the last frame before patch entry (exactly toy v2 arm D).

The friction band `mu_b` for every replay arm that uses a given real failure is the boundary
estimated once, at the baseline, from that failure's **entry** state (toy v2 procedure). So within
one failure, all start-time arms share the friction distribution, the reset noise, the replay
share, the optimiser, the ES random stream and the evaluation: **only the start state differs.**

## Failures (capsules)

- **Discovery worlds** (stage 1): toy v2 W1 to W6, with the toy v2 capsule rule plus one added
  eligibility condition fixed here: onset at least 2.02 s after the first frame, so every offset
  exists. All six toy v2 capsules satisfy it (onsets 2.32 to 3.14 s); the code re-derives them
  and asserts they match the frozen toy v2 capsule files byte for byte.
- **Hidden replication worlds** (stage 2 only), fixed now, none run yet:
  W7 (0.075, 0.95, 1.05), W8 (0.065, 1.15, 1.2), W9 (0.11, 0.75, 1.25), W10 (0.085, 1.05, 0.95),
  W11 (0.095, 0.85, 1.1), W12 (0.055, 1.0, 1.15), as (patch friction, patch start m,
  command m/s). Same capsule rule.

## Arms

Global (one policy per training seed): **A** no repair; **B** broad DR; **ADR** (toy v2);
**C** nominal-entry replay (toy v2 arm C).

Per discovery world (one policy per world and training seed):

| arm | seed state |
|---|---|
| `real_d{d}` (6) and `real_T0` | the failed trajectory at the offset |
| `real_start`, `real_entry` | as above |
| `match_d{d}` (6) and `match_T0` | a **successful** baseline traverse of the same world, the frame at the position (x) closest to the real offset frame; same sensor-noise model; same friction band as the real failure. Controls for "same place and command, but not a failing trajectory". The successful traverse is the first success in that world under a fixed seed stream. |
| `sim_d{d}` (6), `sim_T0`, `sim_entry` | a **simulator-found** failure: the baseline is run in random simulated worlds (friction U[0.05, 0.8], patch start and command from the nominal ranges) under a fixed seed stream until the first failure with onset at least 2.02 s and a pre-patch frame; offsets taken the same way; friction band from that failure's own entry boundary. No knowledge of the deployment world is used. |

Stage 2 per hidden world: `real_P*`, `real_T0`, `real_entry`, `match_P*`, `sim_P*`, and a
secondary `real_Rsel` (below). Global arms as above.

## Compute

Every arm trains 120 ES iterations from the same baseline with identical rollout slots.
Out-of-training simulator steps are charged: reconstruction/acceptance and the boundary estimate
to every arm that uses a real failure; the success search to `match_*`; the failure search and
its boundary to `sim_*`; the recoverability probes to `real_Rsel`; the nominal search to C.
B and ADR receive extra iterations until their totals are at least the largest charged total of
any replay arm; the run aborts if any replay arm exceeds B or ADR.

Not charged, stated here: the stage-1 sweep that selects P* is a one-off hyperparameter search.
It is shared by `real_P*` and `sim_P*`; B and ADR do not use a start time. This is a limitation of
the toy protocol and is reported with the result.

## Seeds

Stage 1: the toy v2 twelve (11, 23, 47, 59, 71, 83, 97, 101, 113, 127, 131, 149).
Stage 2: twelve new seeds (151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211).

## Diagnostics (per start state, at the baseline policy, no training; descriptive)

- recoverability `R(t)`: baseline traverse success from the restored state (with reset noise)
  with friction drawn from the replay band, 256 replicates; also at the world's hidden friction;
- survival horizon: mean time to failure from the restored state (capped at the horizon);
- immediate termination: fraction failing within 0.1 s;
- training signal: boundary mass (mean sqrt(p(1-p)) over replay contexts, 16 replicates each);
- ES gradient: mean norm of the ES gradient estimate over 8 batches of the arm's training mix,
  and the spread (SD) of population fitness;
- `R(t)` on a 0.1 s grid along each failed trajectory, real and simulator-found.

## Stage 1 analysis (discovery)

Unit of inference: the training seed. Per seed, each arm's held-out failure is averaged over the
six worlds. Tests are exact two-sided sign-flip over the 12 paired seed means.

1. **Omnibus (primary question).** Statistic: variance across the seven fixed-offset real arms
   (d = 2.0 ... 0.25 and T0) of their seed-averaged means. Null: within each seed, the seven arm
   labels are exchangeable. 20,000 within-seed label permutations, RNG seed 2026.
   Structure exists if p < 0.05.
2. **Pre-failure vs onset.** Each of the six `real_d{d}` minus `real_T0`; Holm correction over
   the six.
3. **Curve class**, decided in this order:
   - **A (no structure)**: omnibus p >= 0.05.
   - **D (failure-state optimum)**: otherwise, if the lowest-failure fixed offset is T0.
   - **E (patch-entry optimum)**: otherwise, if `real_entry` has lower mean failure than the best
     fixed offset t* and beats it with p < 0.05.
   - **B (earlier is better)**: otherwise, if t* is d = 2.0 (the earliest fixed offset).
   - **C (interior optimum)**: otherwise, t* is interior and beats both T0 and d = 2.0 with
     p < 0.05 each. If it does not, the class is **A** (an interior minimum that is not resolved).
4. **Selection.** P* is the pre-failure fixed offset (d in {2.0, ..., 0.25}) with the lowest mean
   held-out failure on the discovery worlds. Selected once, here, never revised.
5. **Stage-2 licence.** Stage 2 runs only if `real_P*` beats `real_T0` at Holm-adjusted p < 0.05.
   Otherwise: NO-GO, stage 2 is not run, and the precursor hypothesis is archived as negative.

Reported descriptively in stage 1: every arm against B, ADR, C and A; `real` against `match` and
`sim` at every offset; nominal success and tracking by offset; all diagnostics; per-world curves.

## Stage 2 analysis (hidden worlds W7 to W12, new seeds): the GO gate

Same unit and test. GO requires **all** of:

- **G1** `real_P*` beats `real_T0` (mean < 0, p <= 0.05).
- **G2** `real_P*` beats each of `real_entry` (the old FBR seed), C (nominal entry), `match_P*`
  (successful trajectory at the same place) and `sim_P*` (simulator-found precursor)
  (each mean < 0, p <= 0.05).
- **G3** `real_P*` beats B and ADR (each mean < 0, p <= 0.05), and its mean is at least 3 pp below
  the better of the two.
- **G4** non-degradation against A, unchanged from toy v2: nominal success one-sided 95% lower
  bound above -2 pp (zero between-seed SD fails closed); tracking RMSE upper bound below
  +0.05 m/s.
- **G5** direction reproduces per world: `real_P*` has lower mean held-out failure than
  `real_T0`, than B and than `sim_P*` in at least 4 of the 6 hidden worlds, each.

G1 to G5 form an intersection-union test: no multiplicity adjustment. Anything short of all five
is **NO-GO**: the precursor method is not built, Isaac Lab and GO2 time are not spent on it, FBR
and the precursor hypothesis are archived as negative results, and Ashfall stays dormant until
real Phoenix walking data suggests a different hypothesis.

Secondary, stage 2, descriptive only: `real_Rsel`, the latest frame of the failed trajectory whose
baseline recoverability at the replay band is at least 0.5 (probes charged), against `real_P*`.

## Prior expectation, stated so it cannot be moved later

Replay arms at this replay share failed the nominal margin in toy v2 (-5.4 and -6.5 pp). Unless
the start time itself changes that, G4 is likely to fail. The margin is kept because no
scientific reason to relax it exists before the data.

## What would not change the decision

Adding offsets, seeds, worlds or arms after seeing results; changing replay share, band, noise,
ADR thresholds or the nominal margin; choosing P* on stage-2 data; switching the primary endpoint.
