# Current limitations

Kept current with the claims ledger. Each item says what is limited, why, and what would lift it.

## The FBR study

- **No GO2 result exists.** The simulation study is preregistered and not run; the hardware study
  is blocked on Phoenix's walking gate (walking is not yet hardware-validated on the robot).
- **The preregistered gate returned NO-GO.** Toy v2: FBR at patch entry did not beat broad DR
  (p 0.105), degraded nominal success by 6.6 pp, and lost to the same replay seeded from a
  nominal entry state. The GO2 study is not run as designed.
- **The mechanism toy is a null.** In the slip cart-pole, boundary replay concentrated more
  boundary mass but did not lower held-out failure relative to broad DR at any budget, and it
  degraded nominal success by 6.8 pp. The GO2 design changed in response (lead-time rule, three
  capsules, development-world choice of band and nominal fraction); whether those changes help
  is unknown.
- **The Isaac Lab path for FBR is not written.** Missing: a sampler-to-scenario adapter, a spatial
  patch with per-foot friction switching, in-loop boundary refresh.
- **Reconstruction is not identification.** A range of simulator frictions reproduces the same
  failure (0.04 to 0.24 in the toy). The range is reported; no coefficient is claimed.
- **One failure family, one robot, few capsules.** Seeds generalise over training randomness, not
  over failures.
- **The endpoint is behavioural.** Slip is not detected on the stock GO2; traverse failure is.
- **Out-of-budget rollouts.** Boundary estimation and refresh cost the boundary arms simulator
  rollouts that broad DR does not spend (1.19% extra steps in the toy); they are counted and
  disclosed, not hidden inside the budget.
- **The deployment world is a simulator.** In simulation, "real failure" means a failure in a
  hidden simulator configuration; only the hardware study tests a physical failure.

## Earlier limitations (v3 causal design, still true of that machinery)

## Evidence

- **No simulator H0 pass exists for any pathway.** The matched-pair path is
  verified on the toy surrogate (mock evidence) and its simulator status is
  recorded in `docs/claims_ledger.md`. Until H0 passes on simulator evidence
  for every retained phenotype, no adaptation experiment may run.
- **No detector validation exists.** The mutation suite catches five
  deliberate defects on the fixture dataset; accuracy on independently
  labelled physics or hardware data is unmeasured. D3, and therefore every
  DELIVERED verdict, inherits that.
- **No Phase-II training, selection or held-out evaluation has run.** The
  protocol, budget, selection and ledger machinery are tested as software
  only.
- **Phase I is uninformative.** See `docs/legacy/README.md`.

## Scope

- **Two phenotypes are excluded from the first study.** `contact_loss` has no
  supported inducing intervention and no gait-phase ground truth; `stumble`
  needs a terrain geometry intervention the backend cannot apply. They are
  marked unsupported, not validated by omission.
- **Hardware is out of scope for collapse, slip and contact loss.** The GO2
  capture path has no validated ground-relative height and no calibrated
  per-foot contact. Attitude loss and command mismatch are observable there.
- **Friction is a robot-shape material coefficient**, verified by readback in
  the simulator. It is not a measurement of surface friction, and the
  intensity axis `1 - dynamic_friction` is a simulator axis.
- **Terrain geometry cannot be intervened per environment** on the Isaac Lab
  task Phoenix uses; terrain is fixed by the task id.

## Simulator stack

- **The baseline policy predates the current environment.** v3b was trained in
  April; actuator latency, motor-strength randomisation and an action rate
  limiter were wired into Phoenix's environment builder in June and are active
  today. Matched pairs share the environment, so causal delivery is still a fair
  comparison, but nominal behaviour is not the trained behaviour. Lifting this
  needs either a baseline retrained on the current builder or a builder flag that
  reproduces the April environment, recorded in provenance.
- **Robot-shape friction combines with the terrain material** under the PhysX
  combine mode of the task. D1 verifies the robot-shape coefficient by readback;
  the effective contact friction is not measured.
- **Checkpoints carry no observation-normalizer statistics**, so the actor runs
  with an identity normalizer; the backend now resolves this from the checkpoint
  instead of trusting the train YAML.

- **A contact-sensor read right after a reset can carry the previous episode's
  contact.** Phoenix measured this in its harvest loop: reading contact data
  after `_reset_idx` but before a physics step stored the pre-reset base contact
  and fired a spurious illegal-contact termination one step later. The H0 backend
  reads contact only in the snapshot after `env.step`, and no episode in the
  Isaac smoke ended before frame 38, so the smoke shows no sign of it. That is
  INFERRED from the code path and the episode lengths; it has not been tested for
  the backend directly.

## Method

- **A restored row is a state-only seed.** `last_action`, the rate limiter
  and the actuator delay buffer are re-initialised by the reset unless
  controller history is supplied; the Phoenix reset telemetry names this.
- **D2 near its margin is regularisation-sensitive.** In the Isaac Lab smoke
  (friction reduction to slip, v3b, 4 pairs, margin ratio 1.5), the sensitivity
  sweep over shrinkage intensities and diagonal floors changed the D2 verdict
  for at least one pair whose primary ratio sat close to the margin. The CLI now
  sweeps every pair by default and records the agreement fraction per pair. How
  an unstable pair counts toward H0 is a decision the preregistration has to make
  before a confirmatory run; it is not made in code.
- **D2 depends on the number of nominal replicates.** With K replicates there
  are K(K-1)/2 null pairs; the ratio to the null maximum, not the empirical p,
  is the discriminating quantity at the small K a simulator budget allows,
  and both are reported. The regularisation floor's influence is reported by
  the sensitivity sweep rather than assumed away.
- **Startup-mode domain randomisation is drawn once per scene** from the
  first pair's seed and shared by every arm; it is recorded, not varied.
- **The exact McNemar test needs at least six pairs** to reach alpha 0.05 and
  the H0 result says so when a design cannot.
- **Frontier crossings are fitted, not observed.** A plateau at the target
  quantile identifies an interval; a fit with no monotone support is
  unidentifiable; nothing is extrapolated.

## Engineering

- **Phoenix is a moving sibling.** Ashfall pins the interface, not the commit,
  and records the revision used. A Phoenix change that keeps the interface
  but changes semantics (for example what a reset randomises) would not be
  caught by the pin.
- **The Isaac paths are exercised by one implementation smoke**, whose status
  is in the claims ledger. CPU tests use fakes for every simulator call.
- **Coverage of the legacy analysis modules is unchanged** and they remain
  tied to the archived directory layout.
