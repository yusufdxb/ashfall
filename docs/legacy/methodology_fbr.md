# Methodology: Failure-Boundary Replay

The one method of the first Ashfall study, in the order the evidence is produced. The full
method, with its mechanism argument and the toy results, is
[`research/FBR_METHOD.md`](research/FBR_METHOD.md); the protocol is
[`research/PREREGISTRATION.md`](research/PREREGISTRATION.md); the status of every claim is
[`claims_ledger.md`](claims_ledger.md). The previous causal-reproduction methodology (H0 to H4,
D1/D2/D3, six phenotypes) is archived unchanged at
[`legacy/methodology_v3_causal.md`](legacy/methodology_v3_causal.md).

Status words: **VERIFIED** (executed here and observed), **INFERRED** (read or reasoned, not
executed), **NOT YET TESTED** (code exists, nothing has run it where the claim needs it).

| step | what happens | code | status |
|---|---|---|---|
| 1. failure | a deployed policy crosses the predeclared criterion (tilt above threshold for 3 steps, or a termination or operator catch) | `fbr.criterion.detect_onset` | VERIFIED on CPU; toy capsules; NOT YET TESTED on GO2 logs |
| 2. capsule | pre-onset history, command, policy hash and criterion id around the onset | `fbr.capsule.extract_capsule`, `capsule.FailureCapsule` | VERIFIED on CPU and in the toy |
| 3. seed states | one frame per predeclared lead time, refused if outside the window, too far from onset, unresettable, implausible or already failing | `fbr.capsule.seed_points`, `assert_near_onset` | VERIFIED; regression tests for the Phase-I row-0 defect |
| 4. reconstruction | matched treated and full-friction control replays from each seed; delivered, trajectory match, failure match | `fbr.acceptance.accept_reconstruction` | VERIFIED in the toy; NOT YET TESTED on Isaac Lab |
| 5. boundary | friction at which the current policy's failure probability crosses one half, per seed, from the isotonic frontier; censored or unidentified boundaries refused | `fbr.boundary.estimate_boundary`, `frontier.FrontierEstimator` | VERIFIED |
| 6. curriculum | boundary draws around each seed's boundary mixed with nominal resets; refresh as the policy improves; one sampler class for FBR and for the nominal-seed control | `fbr.boundary.BoundarySampler`, `BroadSampler` | VERIFIED in the toy; the Isaac Lab adapter to Phoenix scenario resets is NOT YET WRITTEN |
| 7. equal compute | identical PPO iterations, environments, steps, epochs, minibatches; out-of-PPO rollouts counted | `protocol.budget.assert_equal_compute` | VERIFIED as software; NOT YET TESTED on a real run |
| 8. held-out evaluation | six frozen cells, common random numbers, final checkpoint, one evaluation per (arm, seed) through the ledger | `selection.HeldOutLedger`, `stats.paired_seed_effect` | VERIFIED as software |
| 9. non-degradation | nominal success and tracking margins, exact fallback for zero-width intervals | `evaluation.paired`, `evaluation.regression` | VERIFIED as software |
| 10. redeployment | the declared FBR and broad-DR policies on the physical patch and its variants | Phoenix deployment | NOT YET TESTED; blocked on Phoenix's walking gate |

What this methodology does not do:

- It does not identify the physical friction. Reconstruction accepts a range of simulator
  frictions that reproduce the failure; that range is reported, not collapsed to a number.
- It does not detect slip on the GO2. The endpoint is a behavioural traverse failure, because the
  stock GO2 has no calibrated per-foot contact.
- It does not restore hidden controller state. A restored row is a state-only seed unless
  controller history is supplied; Phoenix's reset telemetry records which.
- It does not put recorded transitions into PPO. Only the reset distribution differs between arms.
