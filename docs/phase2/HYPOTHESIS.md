# Ashfall Phase II: hypothesis

Status: PREREGISTERED, NOT YET RUN. No Phase-II data exists.

## Phase I stands as a record, but not as an answer

Phase I tested whether increasing the fraction of failure trajectories in
fine-tuning improves robustness, and returned a paired multi-seed null at n=11
(slippery mean -0.4155 pp, exact sign-flip p=0.726562; rough mean +0.5485 pp,
p=0.767578). Those numbers reproduce exactly from the committed per-seed
`metrics_*.json`. Nothing here revises them, and the archived analysis stays as
written.

What re-analysis changed is not the result but its scope.

## The treatment was never delivered

`phoenix.adaptation.fine_tune` installed the curriculum reset bridge with no
delivery arguments, inheriting `seed_row_strategy="first"` and
`write_velocity=False`. Every failure reset therefore loaded row 0 of a
trajectory. Row 0 of all 18 shipped synthetic trajectories is a nominal gait
state (base height 0.295 to 0.308 m, |roll| about 0, |pitch| under 3 degrees,
forward speed about 0.5 m/s), and failure onset is at rows 50 to 80.
`write_velocity=False` then discarded even that forward velocity.

At `failure_fraction=0.5` the experiment reset half the environments to a
lightly jittered stand. A `failure_onset_minus_k` strategy existed in the code
and was unit-tested, but no production call site ever selected it.

The consequence is precise and limited: **the n=11 null is uninformative about
failure curricula, not evidence against them.** It is a correctly executed,
correctly analysed measurement of a treatment that was not applied. The
hypothesis is untested, not refuted. It is not thereby supported either.

Fixed on `research/ashfall-curriculum-delivery` in go2-phoenix (commit
`cc44261`), where both arguments are now required keyword arguments and four
regression tests pin the defect.

## A second defect: the terrain contrast is not what it is named

`go2_env_cfg.py` documents which YAML sections it wires. `terrain` appears in
neither the wired list nor the "present but not wired" warning list, and
`_UNWIRED_TOP_LEVEL` does not cover it, so a `terrain:` block is dropped with no
warning. `rough.yaml` (generator terrain: stairs, slopes, obstacles) and
`slippery.yaml` (flat terrain plus friction patches) therefore both build on the
same upstream rough terrain. The only difference that takes effect is
`domain_randomization.friction_range`. The friction-patch mechanism that gives
"slippery" its name is discarded entirely.

Every Phase-I terrain comparison is therefore a friction-randomization
comparison on identical terrain. That is a real contrast, but it is not the one
the configs, the reports, or the README describe.

## Phase-II hypotheses

> **H0 (delivery, prerequisite).** With the delivery fix applied, the states at
> which curriculum-seeded environments begin are measurably distinct from
> nominal reset states, and distinguishable in the direction of the intended
> failure mode.

H0 is instrumentation, not science, and it is verified by direct measurement of
seeded initial states rather than by any downstream outcome. Nothing below runs
until H0 passes. Phase I is the argument for why this gate exists.

> **H1 (mechanism conditioning).** Adaptation conditioned on a discrete,
> causally named failure mode, which perturbs the physical parameter that
> produces that mode, reduces recurrence of that mode on a frozen mode-specific
> challenge suite, relative to a compute-matched uniform failure curriculum.

> **H2 (retention).** H1 is achieved without degrading nominal locomotion
> beyond a preregistered margin.

H1 and H2 are judged jointly. A treatment that satisfies H1 by walking slower,
lower, or with a higher duty factor fails, and the covariate gates in
PREREGISTRATION.md exist to detect exactly that.

## Novelty boundary

Conditioning a policy on an estimated continuous latent of the physical
parameters is UP-OSI (RSS 2017) and RMA (RSS 2021). Replaying
failure-inducing environment parameters is PLR, Robust PLR and ACCEL. Phase II
does not claim either. The only axis claimed as new is conditioning on a
**discrete, causally named, independently interveneable failure mode**, whose
label is produced by a validated detector and whose intervention is a named
physical parameter, in contrast to RMA's uninterpretable continuous latent.

If the mandatory baselines (RMA, automatic domain randomization, terrain-
curriculum PPO, Robust PLR) match or beat mode conditioning, the honest result
is that the discrete factorization buys interpretability and nothing else, and
that is what will be reported.
