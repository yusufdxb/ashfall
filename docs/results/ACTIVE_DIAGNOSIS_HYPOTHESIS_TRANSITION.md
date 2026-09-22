# Ashfall hypothesis transition: from failure-conditioned fitting to active diagnosis

Status: **record of a change of question.** Nothing in the frozen results below changes; all
stay byte-identical and hash-tested (`tests/test_frozen_results.py`,
`tests/test_recoverability_history.py`).

| hypothesis | verdict | record | commit |
|---|---|---|---|
| H1 Failure-Boundary Replay | **NO-GO** | `docs/results/FBR_TOY_NEGATIVE_RESULT.md` | `48c8c71` |
| H2 precursor replay | **NO-GO** | `docs/results/PRECURSOR_SWEEP_RESULT.md` | `dfec130` |
| H3 recoverability-guided replay | **NO-GO** | `docs/results/RECOVERABILITY_RETROSPECTIVE_RESULT.md` | `61bb95d` |
| H4 failure-critical system identification (FCSI) | **NO-GO** | `docs/results/FCSI_TOY_RESULT.md` | `d49daeb` |

None of these is a partial success. Branches `research/ashfall-failure-boundary-replay`,
`research/ashfall-recoverability` and `research/ashfall-failure-critical-sysid` are not rewritten.

## FCSI result

Failure-conditioned event reproduction did not beat strong reset-based global fitting: the
registered method named the true mechanism in 7 of 18 known-mechanism instances against 14 of 18
for multiple shooting, and predicted held-out failures worse. **FCSI failed.**

## Unexpected useful result (exploratory, from the same run)

A simple sparse one-step dynamics selector (FCSI's no-event ablation: one mechanism at a time,
best reset-based residual likelihood wins) named the true mechanism in 18 of 18 known cases. But:

- it had no principled way to recognize a mechanism outside its library;
- forced to choose, it named the wrong mechanism in 8 of 8 unknown-mechanism cases.

Closed-set diagnosis worked; open-world diagnosis did not.

## New hypothesis (H5)

The next step may not be better *passive* confidence calibration. Instead: after a failure
leaves several simulator explanations plausible, **actively choose a safe experiment whose
predicted outcomes differ most across the remaining hypotheses**, run it, update, and end in one of
four explicit states: KNOWN (one library mechanism explains everything), UNKNOWN (no library
mechanism predicts the observations), UNRESOLVED (no admissible probe separates what is left),
ABORT (probe safety cannot be guaranteed or was violated).

**This is a new hypothesis.** It is motivated by the FCSI result and was not preregistered
before. Two outcomes would make it unnecessary and are tested as baselines, not assumed away:
passive diagnosis with calibrated abstention already rejecting unknown mechanisms, and one
optimized fixed probe doing as well as active selection.

The 18/18 closed-set result was exploratory and was measured on instances whose results were
already seen; it is not reused as confirmatory evidence here. The confirmatory study uses new
instances and new seeds.
