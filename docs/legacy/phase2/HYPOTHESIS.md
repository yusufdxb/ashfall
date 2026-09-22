# Ashfall Phase II: hypotheses

Status: PREREGISTERED, NOT YET RUN. No Phase-II data exists. The full protocol, endpoints,
statistics, selection rule, ledgers and kill criteria are in
[PREREGISTRATION.md](PREREGISTRATION.md); this file states the hierarchy and the history that
made it necessary.

## The hierarchy

    H0 causal delivery -> H1 phenotype fidelity -> H2 repair -> H3 generalization -> H4 non-degradation

> **H0 (causal delivery).** A physical intervention induces the intended phenotype significantly
> more often than a matched no-intervention counterfactual: same restored state, same policy,
> same command, same simulator seed, same environment, same timing; control receives no
> intervention, treatment receives it. Delivery requires three gates: D1 the intervention was
> verifiably applied, D2 the trajectory departed from the matched control beyond independent
> nominal variation, D3 the intended, independently defined phenotype occurred.

> **H1 (phenotype fidelity).** Reproduced failures dynamically resemble independently labelled
> reference failures of the same phenotype.

> **H2 (repair).** Failure-phenotype-conditioned adaptation reduces held-out failure incidence,
> aggregated over the preregistered validated phenotype set, versus equal-compute
> non-conditioned alternatives.

> **H3 (generalization).** Repair generalizes to withheld causes and intensities that produce the
> same phenotype.

> **H4 (non-degradation).** Repair does not exceed a preregistered nominal-performance
> degradation margin.

No downstream hypothesis executes if its prerequisite gate is not PASS, and H1 to H4 require H0
PASS for every phenotype in the retained set. `ashfall.protocol.hypotheses.HypothesisLedger`
enforces this mechanically and records every verdict in a hash chain. UNINFORMATIVE is a verdict,
not a pass.

The cause-effect vocabulary is `ashfall.ontology`: an intervention is applied, a phenotype is
observed, and no one-to-one mapping between them is assumed.

## Why the hierarchy exists: Phase I and Gate A

Phase I tested whether increasing the fraction of failure trajectories in fine-tuning improves
robustness, and returned a paired multi-seed null at n=11 (slippery mean -0.4155 pp, exact
sign-flip p=0.726562; rough mean +0.5485 pp, p=0.767578). Those
numbers reproduce exactly from the committed per-seed `metrics_*.json` under
`results/legacy_row0_curriculum/` and are pinned by a regression test. Nothing here revises them.

What re-analysis changed is their scope, twice:

1. **The treatment was never delivered.** `phoenix.adaptation.fine_tune` installed the curriculum
   reset bridge with `seed_row_strategy="first"` and `write_velocity=False`. Every failure reset
   loaded row 0 of a trajectory, and row 0 of all 18 shipped synthetic trajectories is a nominal
   gait state (base height 0.295 to 0.308 m, near-zero roll and pitch, forward speed about
   0.5 m/s) with onset at rows 50 to 80. At `failure_fraction=0.5` half the environments reset to
   a lightly jittered stand. The n=11 null is therefore **uninformative about failure curricula,
   not evidence against them**: a correctly executed, correctly analysed measurement of a
   treatment that was not applied. The hypothesis was neither refuted nor supported.
2. **The terrain contrast is not what it is named.** `go2_env_cfg.py` never read the `terrain:`
   block, so `rough.yaml` and `slippery.yaml` both built on the same upstream terrain and differed
   only in `domain_randomization.friction_range`. Every Phase-I "terrain" comparison is a
   friction-randomization comparison on identical terrain.

**Gate A (the first H0) ran on 2026-09-11 and failed.** With the corrected production strategy
(`failure_onset_minus_seconds`, 0.5 s) the curriculum still seeded a nominal gait state for five
of six failure modes, because a fixed rollback of 25 rows at dt=0.02 lands back inside the
50-row stable prefix for every mode except command mismatch, and failure development time varies
by a factor of six across modes. `contact_loss` was undeliverable by state restoration by
construction (its only signature channel, contact force, is a physics output no reset writes),
and all five real hardware captures carried no onset label and zero-filled base position.

Those findings are why the redesign replaced the single concept `failure_mode` with an explicit
cause, response and phenotype; replaced index arithmetic against a trajectory's own prefix with a
matched counterfactual pair and three gates; and placed H0 in front of everything else as a
condition that must be PASS per phenotype before any repair experiment may run.

## Novelty boundary

Conditioning a policy on an estimated continuous latent of the physical parameters is UP-OSI
(RSS 2017) and RMA (RSS 2021). Replaying failure-inducing environment parameters is PLR, Robust
PLR and ACCEL. Phase II claims neither. Arm C (intervention-conditioned replay) is the closest
in-house analogue of parameter-conditioned replay and is a mandatory comparator. The only axis
claimed as new is conditioning on a discrete, causally delivered, independently detected failure
phenotype (arm D), and that claim is contingent on H0 PASS and a validated detector for each
retained phenotype. If arms A, B or C match or beat arm D at equal compute, the honest result is
that the discrete factorization buys interpretability and nothing else, and that is what will be
reported.
