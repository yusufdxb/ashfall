# Recoverability retrospective: analysis plan (frozen before recoverability is estimated)

Status: **EXPLORATORY analysis plan, committed with its code before any new recoverability
estimate is computed.** It is not a confirmatory preregistration. The repair outcomes it
analyses were already seen (they are the frozen precursor sweep), and the hypothesis it probes
was suggested by them (`docs/results/ASHFALL_HYPOTHESIS_TRANSITION.md`). What this file fixes
in advance is everything that is still free: the recoverability definition, the estimator,
the models, the metrics and the decision rules for Phases 2 to 6. Nothing below may be tuned
after the estimates exist; deviations are reported as deviations.

Evidence kind: `toy_mechanism` (slip cart-pole). No GO2 or quadruped claim.

## Data (read only)

- `results/precursor_toy/3205f02/stage1.json`: discovery worlds W1 to W6, 12 training seeds,
  24 per-world replay arms each (144 states).
- `results/precursor_toy/3205f02/stage2/stage2.json`: hidden worlds W7 to W12, 12 new seeds,
  6 per-world replay arms each (36 states).
- Frozen baseline policy `results/fbr_toy_v2/8f49f02/baseline.npy` (SHA-256 `2bb2a3bf...e0f3`).

Global arms (A, B, ADR, C) are reference points, not dataset rows: each is one policy evaluated
on every world, so its rows would not be independent.

## Definitions

- **State** `s`: a restorable toy state (`x, v, theta, theta_dot, phase, v_cmd, patch_start`)
  exactly as the sweep stored it for a replay arm.
- **Provenance**: `failed_real` (arms `real_*`, the deployment failure), `failed_sim` (arms
  `sim_*`, simulator-found failure), `successful` (arms `match_*`, a successful baseline
  traverse matched by position).
- **Recoverability, primary** `R_band(s)`: success probability of the frozen baseline from `s`
  over the full 3.5 s continuation horizon, with patch friction `mu ~ U(0.02, band_hi)`
  (`band_hi` = the arm's own one-sided replay band edge, `min(0.8, mu_b + 0.05)`) and the
  capsule sensor-noise jitter the replay arms used (`SEED_NOISE`). This is the distribution
  the replay half of the arm's training batch is drawn from, so `R_band` is the success rate
  the optimiser saw at the start of repair. It uses no hidden-world friction and no outcome.
- **Recoverability, secondary** `R_ref(s)`: same, with one common reference band
  `mu ~ U(0.02, 0.20)` for every state (removes the arm-specific band from the definition).
- **Estimator**: `ashfall.recoverability.estimator.estimate`, `n = 1024` independent
  continuation rollouts per state and distribution, Wilson 95% interval, noise streams hashed
  per state and disjoint from evaluation seeds. A second independent repeat (`repeat = 1`,
  `n = 1024`) of `R_band` measures test-retest reliability. Every continuation step is counted.
- **Repair value** `V(s)`: held-out failure of the unrepaired baseline A minus held-out failure
  of the arm repaired from `s`, in the same world, averaged over the 12 paired training seeds
  (positive = repair helped). Budget: each replay arm trained 120 ES iterations.
- **Time before failure** `lead_s`: as stored by the sweep; matched-success states inherit the
  lead of the real state they were matched to (they were constructed that way).
- **Distance to hazard** `d = patch_start - x` (metres, negative once on the patch).

## Models (Phase 4)

All models include world fixed effects, because the question is which state to pick *within* a
world, and the baseline failure rate differs by world. Every continuous predictor gets the same
flexible family so the comparison is fair: **quadratic is primary**; linear and a natural cubic
spline (3 df) are reported as secondary.

| model | predictors (plus world FE) |
|---|---|
| A | time before failure |
| B | distance to hazard |
| C | provenance (2 dummies) |
| D | recoverability `R_band` |
| E | `R_band` + provenance |
| F | `R_band` + time before failure |

Metrics, discovery data (144 states):

1. **Leave-one-world-out CV** (primary): fit on five worlds, predict the world-centred repair
   value of the sixth; report CV RMSE and CV R^2 against predicting the world mean.
2. AIC (Gaussian, OLS) on the full discovery set.
3. Mean within-world Spearman correlation between fitted and observed values.

"`R` explains repair value better than time" requires **both** a lower LOWO CV RMSE for D than
for A and a higher CV R^2; the same rule for B. Anything else is reported as "not better".

## Non-monotonicity test (Phase 4)

On model D (quadratic, discovery data): **interior optimum** iff (i) the quadratic coefficient
is negative, (ii) the vertex `R*` lies inside the observed `R_band` range, and (iii) the fitted
slope is positive at the 10th percentile of observed `R_band` and negative at the 90th
percentile, each with a two-level bootstrap 95% interval excluding zero (2,000 resamples:
worlds with replacement, then training seeds with replacement, paired across arms, recomputing
`V` from the per-seed deltas; `R` held fixed). The bootstrap interval for `R*` is reported
whether or not the test passes. If the observed `R_band` range does not reach high values
(90th percentile below 0.8), the high-recoverability side is reported as **not observed**,
not as a decline.

## Provenance after conditioning on recoverability (Phase 5)

- Model E vs D: RSS reduction from adding provenance; p-value from 10,000 permutations of the
  provenance labels within world (R and V fixed). Report the provenance coefficients with
  bootstrap intervals.
- LOWO CV RMSE of E vs D.
- "Provenance adds little or nothing" iff permutation p >= 0.05 **and** E does not improve LOWO
  CV RMSE by more than 5%.
- Plot `V` vs `R_band` for all three provenances on identical axes.

## Held-out prediction (Phase 6, uses stage 2; no training)

Fit on discovery data only. Selection rules, fixed now:

- **R rule**: fit model D; `R*` = argmax of the fitted quadratic over [0, 1]. In each hidden
  world, pick the candidate whose `R_band` is closest to `R*`.
- **Time rule** (comparator): fit model A; `lead*` = argmax over the observed lead range; pick
  the candidate whose lead is closest.
- **Onset rule**: `real_T0`. **Random**: expected value over the world's candidates.

Candidates per hidden world: the six stage-2 arms (`real_T0, real_entry, real_d1.5,
match_d1.5, sim_d1.5, real_Rsel`). Report per world: chosen arm, its `V`, the best candidate's
`V`, regret, and the out-of-sample R^2 and Spearman of models A, B, D on the hidden states.
Stage 2 offers few, clustered candidates (three of six sit at T-1.5 s), so this test is weak
by construction; it is labelled exploratory. A proper blind test needs new worlds and new
training, which belongs to a new preregistration (Phase 7) and only if Phases 4 to 6 show a
meaningful relationship.

## Gate for going on to Phase 7

Phase 7 (a new preregistration) is written only if **all** hold on the discovery data:
D beats A and B on LOWO CV (rule above), the interior-optimum test passes or the relationship
is at least monotone with a bootstrap-significant slope, provenance adds little (rule above),
and the R rule's mean regret on hidden worlds is no worse than the time rule's.
It is also conditional on the literature review
(`docs/research/RECOVERABILITY_RELATED_WORK.md`) leaving a distinct claim.
