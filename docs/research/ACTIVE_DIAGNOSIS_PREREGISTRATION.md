# Active simulator diagnosis, ground-truth toy: preregistration

Status: **frozen before the confirmatory (evaluation) run.** Code at `2b3e40e`
(`src/ashfall/active_diag/`); development calibration
`results/active_diag/2b3e40e/calibration.json`, SHA-256
`efca9e4ceadde1325c2e713d67980ed96adbbc8cd656d680b645aa55804ffe72`. The evaluation stage
(`python -m ashfall.active_diag.study eval --output results/active_diag/2b3e40e`) reads that file,
refuses to run without it, and fits nothing. No evaluation instance has been built.

Evidence kind: `toy_mechanism`. Hypothesis: `docs/results/ACTIVE_DIAGNOSIS_HYPOTHESIS_TRANSITION.md`
(H5). Adapted question after the literature audit (`docs/research/ACTIVE_DIAGNOSIS_RELATED_WORK.md`):
closed-set safe active discrimination is prior art (Ni et al. 2026 and active fault diagnosis); the
test is whether safe active probing adds **open-set** detection (UNKNOWN) over passive calibrated
abstention, an optimized fixed probe and random safe probes, without harming known diagnosis.

## Frozen design

- **Known mechanisms** (in the 10-parameter library), each with the FCSI nuisance mismatch:
  patch friction x0.5 (`mu_scale`), sliding friction 0.4 (`kinetic_ratio`), braking friction x0.4
  (`aniso`), command latency 1.5 steps (`action_delay`), sensor latency 1 step (`obs_delay`).
- **Development-only unknown mechanisms**: actuator deadzone 9 N; speed-dependent patch friction
  (x(1 - 0.6|v|)).
- **Evaluation-only unknown mechanisms** (the frozen FCSI ones): unmapped strip after the patch
  (x0.15); actuator dropout on the patch when cos(gait phase) > 0.
- **Split**: development 3 instances per known + 3 per development unknown (global indices from
  2000); evaluation 6 per known + 6 per evaluation unknown = 30 known + 12 unknown (indices from
  3000). No overlap with FCSI instances (0 to 28, 900).
- **Evidence**: one failure trace at a condition the nominal simulator predicts as safe (nominal
  failure <= 0.05, real >= 0.25); 8 successful nominal logs are used only to calibrate the residual
  scale.
- **Hypotheses and weights**: 10 mechanisms x 13-point grids (+ nominal value) + `none`; tempered,
  grid-marginalized one-step residual likelihood with the action-consistency channel, per-row cap
  50, rows above 0.35 rad tilt excluded. Temperature 1.0 (fitted on development knowns). Called
  weights, not probabilities.
- **Probe family**: 81 probes = test-lane friction {0.3, 0.5, 0.8} x speed {0.6, 0.8, 1.0} x
  {none, sine 3/6 N at 1/3 Hz, pulse +-4/+-8 N}; a probe is one slow crossing (3.5 s). No stationary
  probe: the frozen policy cannot hold still.
- **Safety**: hard admissibility; under every plausible hypothesis (weight >= 0.02, MAP and next
  grid value, plus nominal), 32 draws, violation rate <= 1/32; bounds tilt <= 0.35 rad, no fall,
  speed <= 2 m/s, probe force <= 8 N. Real execution switches the excitation off above 0.35 rad and
  counts a violation.
- **Information objective (active)**: Monte Carlo expected information gain about the mechanism
  label, in the update's own likelihood, with a 0.05 floor mixture; minus a 0.01 x |A|/8 cost.
- **Loop**: at least one probe, at most 3; stop when top weight >= 0.9 (after the first probe);
  active also stops if the best EIG < 0.01. ABORT only if a probe is needed (top weight < 0.9) and
  none is admissible.
- **UNKNOWN test**: CUSUM peak of capped six-channel one-step residuals under the best library
  explanation at fine parameter resolution; thresholds (largest statistic on development knowns):
  passive_abstain 3.9, fixed 5.3, random 5.3, active 5.3.
- **Fixed probe** (optimized on development, the strongest of 81): `cross(mu=0.5, v=0.6)` (no
  excitation).
- **Final status**: UNKNOWN if statistic > threshold; else KNOWN(top) if top weight >= 0.9; else
  UNRESOLVED; ABORT as above. Oracle: best single admissible probe in hindsight (upper bound only).

## Primary endpoints and gate (all must hold for GO; evaluation set)

| gate | rule |
|---|---|
| A1 known non-inferior | active known-correct >= max(passive_abstain, fixed, random), or (active >= 27/30 and mean entropy <= 0.5 x passive's) |
| A2 open-set gain | active UNKNOWN on unknowns >= passive_abstain + 3, and >= 8/12 |
| A3 false UNKNOWN | active UNKNOWN on knowns <= 3/30 |
| A4 safety | 0 safety violations in executed active probes |
| A5 probe budget | at most 3 probes |
| A6 beats random | active total correct > random, exact McNemar p <= 0.05 |
| A7 beats fixed | active total correct >= fixed + 4 |

GO => misspecification stress test, then quadruped cross-simulator study under new
preregistrations. NO-GO => report which gates failed and stop; archive.

## Development results, disclosed before the evaluation run

Development scores (known correct + unknown rejected, of 21): passive_abstain 18, fixed 21,
random 19, active 21. On development data the fixed probe already matched active selection, and
passive abstention already rejected 5 of 6 development unknowns (active 6 of 6). Probing resolved
the two development cases where the failure trace left two known mechanisms at 0.5/0.5. **On this
evidence the prediction is NO-GO on A2 and A7**; the evaluation run tests it on new instances and
unseen unknown mechanisms.

## Development changes, disclosed (all before this file)

1. Stationary probes dropped (the policy cannot hold still); probes are slow crossings.
2. Rows above 0.35 rad tilt excluded, per-row caps in the likelihood (50) and the UNKNOWN
   statistic (99.9% chi-square): single rows at a fall or a patch edge produced d^2 in the
   thousands under the exact true physics.
3. Evidence restricted to the failure trace: with the nominal logs as evidence, passive weights were
   1.00 on every development case, leaving nothing to probe.
4. ABORT only when a probe is needed; thresholds from all development knowns (excluding aborted cases
   had lowered the probing methods' thresholds artificially); a floor-friction probe lane added.
5. The UNKNOWN test refines the best parameter (latency 1.5 sits between grid points).
