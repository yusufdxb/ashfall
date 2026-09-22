# FCSI ground-truth toy: preregistration (Gate 2)

Status: **frozen before the registered run.** Committed together with the code that runs it
(`src/ashfall/fcsi/`, entry `python -m ashfall.fcsi.toy_study`). No registered instance was
built, fitted or scored before this commit; all development used the unregistered
`QUICK_WORLD` (friction collapse 0.5, instance seeds offset by 900) and the unit tests.

Evidence kind: `toy_mechanism` (cart-pole with a low-friction patch). Nothing here is a GO2,
quadruped or Isaac Lab result. Hypothesis: `docs/research/FCSI_HYPOTHESIS_TRANSITION.md` (H4).
Novelty context: `docs/research/FCSI_RELATED_WORK.md` (Gate 1 passed as a combination claim only).

## Question

When the simulator predicts success and the pseudo-real world fails, does failure-conditioned
sparse identification (FCSI) repair the simulator better than global system identification
given the same data? The first gate is **simulator repair**, not policy improvement.

## Worlds (hidden truth; the identifiers never see these)

Nominal simulator: `toy_world.Physics.of()`. Every pseudo-real world also carries a small
nuisance mismatch (pole mass 0.31 instead of 0.30, cart damping 0.03 instead of 0) and
measurement noise on every logged channel.

| world | kind | hidden mechanism | in library? | instances |
|---|---|---|---|---|
| K1_kinetic | known | patch sliding friction ratio 0.4 (nominal 0.85) | yes (`kinetic_ratio`) | 6 |
| K2_latency | known | command-to-force latency 1.5 steps = 30 ms (nominal 0) | yes (`action_delay`) | 6 |
| K3_braking | known | patch braking-direction friction x0.4 (nominal x1) | yes (`aniso`) | 6 |
| U1_strip | unknown | unmapped strip after the patch, friction x0.15 | **no** | 4 |
| U2_dropout | unknown | actuator dropout on the patch when cos(gait phase) > 0 | **no** | 4 |
| N_harmless | negative | cart damping 0.4, no failure-critical change | (yes, but harmless) | 3 |

Magnitudes were chosen from failure-rate scans of the worlds alone (no identifier existed yet)
so that the nominal model predicts success where the pseudo-real world fails.

## Evidence per instance (identical for every method)

Deterministic from the instance index: 8 successful nominal real logs at benign conditions
(patch friction U(0.5, 0.8)); one **target failure log** at a discrepant condition (nominal
failure rate <= 0.05 and real >= 0.25, screened with 128 episodes each); one **validation
failure log** at a second discrepant condition. N_harmless instances get a target log that
does *not* fail (real and nominal failure <= 0.05) and no validation log.

Held out from every method, used only for scoring: 4 more nominal logs and 12 conditions
(patch friction U(0.10, 0.45)) with real and nominal failure rates from 256 episodes each.

## Methods (all fit the same 10-parameter library, `src/ashfall/fcsi/mechanisms/`)

- **FCSI** (primary): divergence localization (one-step reset-based residuals, robust nominal
  scale, CUSUM with 1% false-alarm threshold), then single mechanisms (25-point grid + 9-point
  refinement over the prior range), pairs of the three most promising singles only if no single
  is feasible; lexicographic rule: feasible = event probability >= max(0.1, 5 x nominal model's)
  from 0.5 s before the divergence (128 draws, same phenotype, fall onset within 0.3 s), nominal
  logs stay successful >= 0.8, pre-divergence residual NLL within 2 sd; then sparsest, then lowest
  residual NLL; posterior weights from evidence (NLL - log p_event + BIC); equivalence class
  within 2.3; abstain (UNEXPLAINED) if nothing is feasible. Six acceptance checks
  (`acceptance.py`); failing any makes the status `unexplained` and leaves the simulator
  unpatched.
- **FCSI_no_event** (ablation): same search, event and feasibility ignored, best residual fit wins.
- **G0 whole-trajectory** (SimOpt-like), **G0b multiple shooting** (5-step reset-based
  Mahalanobis), **G1 DROPO-like** (one-step transition likelihood over a Gaussian parameter
  distribution; point estimate = mean). Each: 256-point scrambled Halton design + 3 bounded
  Nelder-Mead restarts of 250 evaluations over all 10 parameters, weak ridge toward nominal;
  G0b and G1 hold the two open-loop-invisible sensing parameters at nominal. Simulator steps
  are counted for every method and reported.

## Metrics

Per instance and method (`evaluation.py`): held-out failure-rate MAE over the 12 conditions (and
on the discrepant subset); target and validation event probability; nominal one-step d^2 per row
and success reproduction on the 4 held-out nominal logs; implied mechanism (FCSI: accepted top
hypothesis; globals: largest normalized change); number of parameters changed by > 5% of span;
divergence time from the detector vs the instrumented truth effect onset.

## Gate 2 (all must hold for GO)

| gate | rule |
|---|---|
| G2a identification | FCSI names the true mechanism in >= 14 of 18 known instances, and more often than each global baseline |
| G2b held-out prediction | FCSI's mean held-out MAE is lower than every global baseline's, and against the best global baseline the paired exact sign-flip p <= 0.05 (18 known instances) |
| G2c failure reproduction | FCSI's patch reproduces the target event by the feasibility rule (>= max(0.1, 5 x nominal)) in >= 14 of 18 known instances |
| G2d nominal fidelity | FCSI's mean nominal d^2 per row on held-out nominal logs <= 1.05 x the unpatched nominal model's |
| G2e abstention | FCSI returns UNEXPLAINED in >= 6 of 8 unknown-mechanism instances |
| G2f negative control | FCSI proposes no patch in 3 of 3 harmless instances |

GO => Phase J (quadruped cross-simulator study) under a new preregistration. NO-GO => report
exactly which gate failed and why; no rescue by retuning on these instances.

Reported without gating: confusion of implied mechanisms (truth x inferred) per method; FCSI
posterior weight vs correctness; FCSI_no_event vs FCSI (does the event term matter); false
diagnosis rate on unknown mechanisms; abstention precision; compute per method.

## Development deviations, disclosed

All made on `QUICK_WORLD` or on world scans before any registered instance existed:

1. Divergence residuals use one step (k = 1), not five: a 5-step detector missed latency and
   braking-friction mismatches entirely.
2. The residual scale is robust (trimmed) and the CUSUM threshold comes from a chi-square
   false-alarm rate: calibrating on "no nominal log may alarm" made the detector blind, because
   nominal logs can contain the hidden mechanism.
3. Contact parameters apply to the patch surface only, and episodes start in steady motion:
   otherwise every mechanism acted at the from-rest start and nothing was failure-localized.
4. The event rule is relative (>= max(0.1, 5 x nominal)): on the quick world the true physics
   reproduced its own logged fall only 23% of the time, so any absolute 0.5 threshold rejects
   the truth.
5. The validation check uses a factor 2, not 5: on the quick world the correct patch reached 3.8x
   the nominal model on the validation log.
6. Global baselines: sensing parameters frozen for open-loop fits, and a weak ridge toward
   nominal. Both can only help the baselines.

These are choices made while looking at one unregistered instance of one mechanism; they may
still favour FCSI on friction-collapse-like failures. That risk is why K2 (latency) and K3
(braking friction) and the unknown worlds are part of the gate.
