# Lessons learned

Technical lessons from five negative Ashfall studies. Each points to the frozen result that
supports it; the numbers are from CPU toys and do not transfer by themselves. The formal record
is [`ARCHIVE.md`](ARCHIVE.md).

## Strong baselines decide the result

Every proposed method beat something weak and lost to something strong.

| Proposed method | Beat | Lost to, or tied with |
|---|---|---|
| FBR | ADR (-6.7 pp, p 0.0010) | broad DR (not significant), ordinary patch-entry state (+3.6 pp against FBR) |
| Precursor replay | onset replay, old FBR seed, broad DR, ADR | simulator-found and successful states at the same time, nominal entry |
| Recoverability rule | onset replay, random candidate | a timing rule (regret 3.4 vs 1.9 pp) |
| FCSI | whole-trajectory fitting (7 vs 3 of 18) | multiple shooting (14 of 18) |
| Active probing | passive forced choice | fixed probe (38 vs 38 of 42), random probe (37) |

The comparisons that ended each study were cheap to add: broad DR and ADR at matched compute,
a successful state at the same time and place, a simulator-found state, multiple shooting, a
fixed probe. Against the weaker baselines alone, three of the five studies would have looked
like wins: FBR against ADR, precursor replay against broad DR and ADR, active probing against
passive diagnosis.

Sources: [FBR](results/FBR_TOY_NEGATIVE_RESULT.md), [precursor](results/PRECURSOR_SWEEP_RESULT.md),
[recoverability](results/RECOVERABILITY_RETROSPECTIVE_RESULT.md), [FCSI](results/FCSI_TOY_RESULT.md),
[active diagnosis](results/ACTIVE_DIAGNOSIS_TOY_RESULT.md).

## Matched controls separate "where" from "whose"

The provenance question (does it matter that this state came from a real failure?) was only
answerable because each replay arm had a control at the same time and position: a successful
trajectory, a simulator-found failure and a nominal entry state. The precursor sweep found a
large timing effect; the matched controls showed that the effect belonged to the time before the
hazard, not to the failure. An experiment that varies start time without these controls would
have attributed the timing effect to the real failure.

## Failure outcomes carry less information than dynamics

A failure is one realization of a stochastic process. In the FCSI toy, a patch with the correct
mechanism reproduced the logged failure only 2% to 41% of the time. Used as a constraint, the
event rejected correct models and admitted sensing mechanisms (observation delay, tilt bias)
that destabilize the loop and fail in some other way. The same logs, scored by one-step
residual likelihood, identified every known mechanism (exploratory, 18 of 18). The part of a
failure log that identifies physics is the state trajectory before and around divergence, not
the terminal event.

## A closed set hides its own failures

Closed-set selection looked excellent on known mechanisms and gave no sign of trouble on unknown
ones: the sparse selector named a library mechanism for all 8 out-of-library instances, with no
gain in prediction (MAE 0.310 vs 0.311 unpatched). Accuracy on known mechanisms says nothing
about behaviour outside the library. Open-set evaluation needs its own held-out mechanism
families, as the active-diagnosis study used (12 instances from two unseen families).

## Complexity has to pay for itself

Information-theoretic probe selection checked 81 candidate probes for safety under every plausible
hypothesis and scored expected information gain, about 3.5M simulated steps per diagnosis. A
fixed gentle crossing, about 121k steps, gave the same total (38 of 42). Most of the diagnostic
value came from probing at all (probing gained 11 instances and lost 0 against passive diagnosis),
not from choosing the probe. The fixed and random probes are the baselines that exposed this.

## Open-world safety is harder than closed-set safety

A probe admitted as safe under every plausible library hypothesis caused falls in 3 of 18 active
probes on unknown-mechanism worlds, against 1 of 24 on known-mechanism worlds (a tilt, no fall).
The fixed and random probes showed the same pattern. A safety certificate computed over a model
set holds only for dynamics inside that set, and an UNKNOWN verdict is exactly the case where the
set is known to be wrong. This is a known caveat of model-based safety; the toy made it concrete.

## Preregistration stopped the drift

Each study froze its gate, its baselines and its code before the run. That had three effects:

- **No rescue by retuning.** No gate was loosened after a result, including the one prior
  expectation that turned out wrong (the precursor preregistration predicted the nominal margin
  would fail; it passed, and the study still stopped on provenance).
- **Exploratory findings stay exploratory.** The sparse one-step selector and divergence
  localization are reported as observations, not promoted into a method after the fact.
- **Expensive work waited for evidence.** Every gate stood in front of Isaac Lab and GO2 time.
  Because none passed, no quadruped simulation or hardware session was spent on a hypothesis the
  toy had already rejected.

## Novelty audits before the registered run

The recoverability, FCSI and active-diagnosis studies each ran a literature audit before the
method was finalized and before any registered run. Recoverability: the intermediate-difficulty
principle is prior art. FCSI: novel only as a combination. Active diagnosis: safe closed-set
active discrimination was already published, so the question was narrowed to open-set detection. The audits changed what was tested, not only how it was
described.

## Deviations are recorded where they happen

Two runs needed a restart: the first FCSI launch stalled from BLAS oversubscription and wrote
nothing, and the first active-diagnosis evaluation crashed on a tie bug before writing results.
Both are noted beside the artifacts (`results/fcsi_toy/run_notes.txt`,
`results/active_diag/eval_notes.txt`) with what changed and why the registered analysis stands.
