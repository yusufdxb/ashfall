# Ashfall: the research question

> **Archived 2026-09-22.** Historical design record of study 1 (FBR), kept unchanged below this
> note. FBR returned NO-GO at its preregistered toy gate, so nothing described here as planned,
> pending or NOT YET TESTED was run, and none of it will be. The hardware plan it cites
> (`HARDWARE_DEMO.md`) was never run and is not part of the public archive. Final status:
> [`docs/ARCHIVE.md`](../ARCHIVE.md).

## The question

> **Can a real quadruped failure define a local simulation curriculum near the observed
> failure boundary that improves robustness more efficiently than broad domain randomization?**

Public version: **can a robot learn more from where it actually failed than from random
simulated difficulty?**

This is Ashfall's only primary question. Classifying failures, proving causality for many
intervention types, conditioning on phenotypes and building a failure ontology are not the
question. Some of that machinery supports the answer; none of it is the headline.

## The bottleneck it targets

A policy fine-tuned with broad friction randomisation sees many friction values, but most of
its rollouts are either easy (the policy always succeeds) or hopeless (it always fails). A
binary outcome that is always the same carries no gradient signal about success. The
rollouts that could teach the policy about a specific failure, a low-friction patch entered at
a particular speed and gait phase, are a small fraction of the budget. We call this **boundary
dilution** and define it in `FBR_METHOD.md` section 4: the boundary mass
`beta(q) = E_{xi ~ q}[sqrt(p(xi)(1 - p(xi)))]` of a training distribution `q` is small when
few of its contexts sit where the current policy's outcome is uncertain.

A hardware failure is direct evidence that the deployed policy is at or past a boundary in one
specific place: one state, one command, one surface. Failure-Boundary Replay spends the
fine-tuning budget there.

## What the literature review changed

The related-work audit (`RELATED_WORK.md`) found that every ingredient is known separately:
resetting to recorded states (reverse curriculum, Florensa et al. 2017; Salimans and Chen 2018;
Go-Explore), training at a success boundary (ADR, OpenAI 2019; the velocity-command box
curriculum of Margolis et al. 2022; the terrain game of Rudin et al. 2022; ALP-GMM), and using
real rollouts to reshape simulation parameters (SimOpt, BayesSim, DROPO, RialTo). It also found
that "boundary-focused sampling beats broad DR" is already expected from that literature.

So the question is kept, and one comparison is added to make it worth asking:

- **Primary comparison (the question as posed):** FBR against broad friction DR, equal compute.
- **Key secondary, tested in fixed sequence after the primary:** FBR against *nominal-seed
  boundary replay*, the same machinery seeded from the baseline's own successful rollouts,
  which is a simulator-discovered boundary in the spirit of ADR and PLR. This is the comparison
  that decides whether the real failure adds anything beyond "train at a boundary".
- **Ablation:** a friction band taken from the failure's boundary but applied from nominal
  states (the SimOpt-style reading: is it only the friction value?).

If FBR beats broad DR but not nominal-seed boundary replay, the honest conclusion is that
boundary-local training helps and the hardware origin is not shown to matter. The paper would
then say that, and the contribution would shrink to the capsule-and-acceptance protocol plus a
controlled negative result on hardware seeding.

## Why these distinctions are real, not renamings

| method | where the hard case comes from | what is replayed | local physics around it | restores the pre-failure state |
|---|---|---|---|---|
| PLR / Robust PLR | levels the simulator generated, scored by regret | a level seed | no (discrete levels) | no |
| ACCEL | simulator levels, mutated | an edited level | yes, by mutation | no |
| ADR | the edge of a simulator randomisation range | the range | yes, expands its edge | no |
| broad DR | a hand-designed range | nothing | no, uniform | no |
| SimOpt / BayesSim | real rollouts, fitted globally | a parameter distribution | global refit | no |
| RMA / UP-OSI | online history at deployment | nothing (adaptation module) | no retraining | no |
| **FBR** | **one physical failure on the robot** | **the pre-failure state and command** | **friction neighbourhood around that state's boundary** | **yes** |

FBR is not claimed as a new family. It is claimed as a specific, testable combination: the seed
comes from a physical deployment failure, the replay restores that failure's pre-onset state and
command, the physics is varied locally around that state's boundary, and the comparison is made
at equal fine-tuning compute against both broad DR and a simulator-discovered boundary.

## Relation to Phoenix

| | Phoenix | Ashfall |
|---|---|---|
| what changes on the robot | the hardware: actuator response drifts over time | the task environment: the robot meets a surface it fails on |
| signal | persistent commanded-versus-measured actuator residuals | where and when a rollout crosses a predeclared failure criterion |
| question | can hardware degradation observations shape the next actuator training distribution? | can training near an observed failure boundary beat broad random training? |
| owns | deployment, base PPO, ONNX export, hardware monitor, actuator adaptation | failure capsule, boundary reconstruction, local curriculum, the comparative repair experiment |

Ashfall calls Phoenix as its simulator, training and deployment backend and does not duplicate
that infrastructure.

## The answer so far

In the slip cart-pole, the most favourable setting FBR will get: no. Toy v1 put FBR level with
broad randomization. Toy v2, preregistered as a decision gate after a hostile review, fixed the
seeding, added a real ADR arm, charged all extra simulation to the baselines and used six hidden
deployment worlds: FBR beat ADR but not broad randomization, cost nominal performance, and the
same replay seeded from an ordinary state at the patch entrance beat the replay seeded from the
failure. The registered decision is NO-GO for the GO2 study as designed. What survived is
narrower: replaying the patch-entry state helps held-out failure against ADR, whatever the source
of that state, at a cost in nominal performance. Whether that transfers to a quadruped is
untested.

## What would answer it

- **Simulation (proxy for reality):** a hidden deployment-world configuration produces the
  failure; the capsule is extracted from its log exactly as a hardware log would be; arms are
  fine-tuned in the training world; held-out cells are evaluated in the deployment world. Twelve
  training seeds, equal compute, one preregistered endpoint (`EXPERIMENT.md`).
- **Hardware:** a GO2 on a secured low-friction patch; one baseline failure becomes the
  capsule; FBR and broad-DR policies are redeployed on the same patch and on a moved or changed
  patch, with predeclared trial counts (`HARDWARE_DEMO.md`).

Neither has been run. The status of every claim is in `docs/claims_ledger.md`.
