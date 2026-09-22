# Active simulator diagnosis, ground-truth toy: result

Status: **FINAL. The preregistered gate returned NO-GO.** Do not edit the numbers in this file.
No misspecification stress test, quadruped cross-simulator study, GO2 study, simulator patching or
policy retraining is run for this hypothesis.

Evidence kind: `toy_mechanism` (cart-pole with a low-friction patch). Nothing here is a GO2,
quadruped or Isaac Lab result.

- Preregistration: `docs/research/ACTIVE_DIAGNOSIS_PREREGISTRATION.md` (`b86006d`), code `2b3e40e`,
  development calibration SHA-256 `efca9e4c...` (unchanged in the evaluation run).
- Run: `python -m ashfall.active_diag.study eval --output results/active_diag/2b3e40e`, 42
  evaluation instances (30 known-mechanism, 12 unseen unknown-mechanism), 109 s wall on 22 CPU
  workers.
- Deviation: the first evaluation launch crashed before writing any result on a tie-handling bug in
  the safety set (`'aniso@0.3' is not in list`); fixed at `1ea85eb` with a regression test. Untied
  behaviour is unchanged and no development case had a tie, so the calibration stands
  (`results/active_diag/eval_notes.txt`).

## Verdict, plainly

1. **Active probe selection adds nothing over a fixed or a random safe probe.** Total correct of
   42: active 38, optimized fixed probe 38, random admissible probe 37 (McNemar p = 1.0 against
   both).
2. **Active probing adds almost nothing to open-set detection.** UNKNOWN on unseen mechanisms:
   active 11 of 12, passive calibrated abstention 10 of 12 (gate: +3). Unknown detection comes from
   the failure trace itself; the probe barely changes it.
3. **Probing does help closed-set diagnosis, but any probe does, and that part is prior art.**
   Known mechanisms correctly identified: active 27 of 30, fixed 28, random 28, passive with
   abstention 17 (it left 8 cases UNRESOLVED and called 6 UNKNOWN). Probing versus passive: 11
   instances gained, 0 lost (p 0.001). Safe active discrimination among known models is
   established (Ni et al. 2026; active fault diagnosis).
4. **Safety certification under the library fails exactly when the library is wrong.** Probes
   admitted as safe under every plausible library hypothesis caused real safety violations in
   3 of 18 executed active probes on unknown-mechanism worlds (all falls, all in the
   actuator-dropout world) against 1 of 24 on known-mechanism worlds (a 0.36 rad tilt, no fall).
   Fixed and random probes show the same pattern (unknown worlds 2 of 24 and 3 of 20). A probe's
   safety can only be certified under the models one has, and an UNKNOWN diagnosis means those
   models are wrong.
5. **Therefore: NO-GO** (A2, A4, A6 and A7 fail).

## Gate (preregistered)

| gate | rule | result | pass |
|---|---|---|---|
| A1 known non-inferior | active >= max(passive_abstain, fixed, random), or >= 27/30 with entropy halved | 27 vs max 28; entropy 2.5e-15 vs 0.217 | yes (second clause) |
| A2 open-set gain | active UNKNOWN >= passive_abstain + 3, and >= 8/12 | 11 vs 10 | **no** |
| A3 false UNKNOWN | <= 3/30 | 3/30 | yes |
| A4 safety | 0 violations in active probes | 4 (3 on unknown worlds) | **no** |
| A5 probe budget | <= 3 probes | max 3, mean 1.0 | yes |
| A6 beats random | total > random, McNemar p <= 0.05 | 38 vs 37, p 1.0 | **no** |
| A7 beats fixed | total >= fixed + 4 | 38 vs 38 | **no** |
| **decision** | | | **NO-GO** |

## Full table (evaluation set)

| method | known correct /30 | unknown rejected /12 | false UNKNOWN /30 | UNRESOLVED /42 | ABORT /42 | total /42 | real safety violations | mean probes | mean simulated steps |
|---|---|---|---|---|---|---|---|---|---|
| passive (forced) | 22 | 0 | 0 | 0 | 0 | 22 | n/a | 0 | 15k |
| passive + abstention | 17 | 10 | 6 | 8 | 0 | 27 | n/a | 0 | 15k |
| fixed probe | 28 | 10 | 2 | 1 | 1 | 38 | 3 | 1.17 | 121k |
| random safe probe | 28 | 9 | 2 | 0 | 1 | 37 | 4 | 1.07 | 3.4M |
| **active (EIG)** | 27 | 11 | 3 | 0 | 0 | 38 | 4 | 1.00 | 3.5M |
| oracle single probe (upper bound) | | | | | | 35 any-correct | | 1 | |

Mean expected information gain of the chosen active probe: 0.42 nats. Mean physical probe time:
3.5 s per diagnosis. Selecting a probe cost about 3.5M simulated steps (safety check of 81
candidates under every plausible hypothesis plus EIG); the fixed probe needs about 121k. On the 6
latency instances no probe was admissible, but the failure trace was already decisive (weight 1.00),
so the probing methods reported the passive diagnosis unconfirmed.

## What this shows and does not show

- Failure traces often leave two contact mechanisms (patch friction, braking friction, sliding
  friction) observationally tied; one gentle crossing of the suspect material resolves them. This
  helps, but it does not need a chosen probe: the plain crossing is as good.
- Unseen mechanisms are detected mainly because the best library model still leaves a persistent
  residual on the failure trace. Probes add evidence but rarely change the verdict (active +1 over
  passive).
- The negative safety finding does not depend on the diagnosis method: any probe certified under
  a closed library is unverified under unmodelled dynamics.
- Limitations: one toy, a small probe family, probes that are full slow crossings (the policy
  cannot hold still), 12 unknown instances from two families. A probe space with finer, local,
  lower-energy actions could behave differently.

## Falsified in this toy

- "Actively choosing a safe probe by expected information gain improves unknown-mechanism
  rejection over passive calibrated abstention": **not supported** (11 vs 10 of 12).
- "Information-based probe selection beats random or fixed probes": **falsified here** (38 vs 37
  and 38).
- "Probes certified safe under every plausible hypothesis are safe": **false when the true
  mechanism is outside the library** (falls in 3 of 18 active probes on unknown worlds).

## Artifact index (SHA-256; immutable, checked by `tests/test_frozen_results.py`)

```
0726bdf5d0a89fa1e9da14927321e20aa8a16ccddc4fe033f918b4054582cd08  results/active_diag/2b3e40e/eval_result.json
db83fc04b87d023b2fdc9dd8a710dd905f144a7ad2cf67eff502453308a57d4a  results/active_diag/2b3e40e_eval.log
9f24a2fdb65484d671865ff4408e2222b6a766f68695386250269b97997cff5c  results/active_diag/eval_notes.txt
efca9e4ceadde1325c2e713d67980ed96adbbc8cd656d680b645aa55804ffe72  results/active_diag/2b3e40e/calibration.json
99ee054e0993e6ee778270f7d05e0da325e3f83fbf189c7a3ba3cf6686cf2c18  results/active_diag/2b3e40e/dev_runs.json
```
