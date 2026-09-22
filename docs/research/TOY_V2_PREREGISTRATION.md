# Preregistration: toy v2, the decision gate before GO2 compute

Status: **REGISTERED before running.** Committed together with its code
(`src/ashfall/fbr/toy_study_v2.py`); run once afterwards. Evidence kind `toy_mechanism`.

## Why it exists

Toy v1 (`results/fbr_toy/f339087/`, rerun after a bug fix at `f3ccc1b`) was a null on the
primary and failed the nominal margin. A hostile review of the design found that (1) FBR's seeds
were states already slipping on the patch, so FBR never trained at the deployment friction;
(2) arm C was an FBR ablation, not ADR, which is the baseline the novelty claim must beat;
(3) C and D spent simulator steps that B did not; (4) one failure was used for every seed.
Toy v2 fixes all four, in the setting most favourable to FBR (exact physics apart from sensor
noise, one axis, full-state capsules). A loss here is decisive against the GO2 FBR plan; a win
only licenses the GO2 simulation study.

## Design (FIXED)

- **Physics and optimiser:** identical to v1 (`ToySlipConfig`, `ESConfig` defaults); baseline
  retrained with seed 0 for 200 iterations exactly as in v1.
- **Six hidden deployment worlds** (patch friction, patch start, command):
  W1 (0.08, 1.0, 1.1), W2 (0.06, 0.8, 1.0), W3 (0.10, 1.2, 1.2), W4 (0.12, 0.9, 1.3),
  W5 (0.07, 1.1, 0.9), W6 (0.09, 0.7, 1.15). One capsule per world: the first baseline failure
  whose patch-entry seed passes acceptance (delivered, treated no farther from the recording than
  its full-friction control, failure within 0.3 s of the recorded onset while the control
  survives, in at least 4 of 6 replicate pairs).
- **Held-out cells per world:** friction +0.03 and -0.02 (floor 0.03), patch start -0.3 and
  +0.3 m, command -0.25 and +0.25 m/s; 256 episodes each, common random numbers.
- **Arms:**
  - A no repair.
  - B broad DR: friction U[0.05, 0.8], nominal traverses.
  - ADR: nominal traverses with friction U[lo, 0.8]; lo starts at 0.35; every 10 iterations the
    policy is evaluated on 64 traverses at friction lo; success at least 0.8 lowers lo by 0.03
    (floor 0.02), success at most 0.5 raises it by 0.03. Probe rollouts are counted.
  - C entry: the minimal FBR sampler seeded from the patch-entry state of the first successful
    nominal rollout whose boundary is identified.
  - D FBR entry (the minimal FBR): one seed per capsule, the last frame before patch entry;
    boundary estimated once at the baseline (20 frictions, 8 replicates), never refreshed;
    replay friction uniform on [0.02, mu_b + 0.05]; replay share 0.5, the other half broad-DR
    draws; reset noise equal to the capsule sensor noise (tilt 0.005 rad, pitch rate 0.02 rad/s,
    speed 0.02 m/s); no command jitter. One D policy per (world, seed).
- **Compute:** C and D train 120 iterations. Their out-of-training steps (seed search,
  reconstruction, boundary estimate) are counted; B and ADR receive extra iterations until their
  total simulator steps are at least the largest C or D total. The run aborts if any C or D
  total exceeds B's or ADR's.
- **Seeds:** the v1 twelve (11, 23, 47, 59, 71, 83, 97, 101, 113, 127, 131, 149).

## Analysis (FIXED)

- Per training seed, held-out failure is averaged over the six worlds' held-out cells; paired
  differences across the 12 seeds; exact two-sided sign-flip test, alpha 0.05.
- **Primary (intersection-union):** D beats ADR and D beats B, each rejected at 0.05 with D
  better. No adjustment is needed for an intersection-union test.
- **Non-degradation:** D against A, nominal success one-sided 95% lower bound above -2 pp (a
  zero between-seed SD fails closed), tracking RMSE upper bound below +0.05 m/s.
- **Decision rule for the GO2 plan:** GO for experiment E0 only if the primary rejects, the mean
  D minus ADR is at most -5 pp, and non-degradation passes. Otherwise NO-GO: the GO2 FBR study
  is not run as designed; the capsule and acceptance code stay as tooling and the toy results are
  written up as a negative result.
- Descriptive: D against C (does the failure's entry state carry value beyond a nominal entry
  state?), ADR against B, per-world effects, boundary mass at the baseline, the ADR lower-bound
  trajectory.

## What would not change the decision

Adding seeds, worlds or arms after seeing the result; re-tuning replay share, band width or ADR
thresholds; switching the primary comparator.
