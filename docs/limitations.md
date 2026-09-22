# Limitations

Final list at archive (2026-09-22). The pre-archive list, including the limitations of the
earlier Isaac Lab machinery, is kept at
[`legacy/limitations_pre_archive.md`](legacy/limitations_pre_archive.md).

## Scope of all five studies

- **Toys only.** Every result comes from a CPU cart-pole with a low-friction patch. The "real"
  system is a hidden simulator configuration; no physical failure data was used.
- **One hazard family.** Low-friction traverse failure (plus, for the diagnosis toys, a library of
  friction, latency, braking, mass and sensing mechanisms). Other failure families are untested.
- **One baseline policy per toy.** Seeds generalize over training randomness, not over policies.
- **Evolution strategies, not PPO**, trained the replay arms. A gradient-based learner could
  weigh replay states differently.
- **The toys were chosen to be cheap and favourable to the proposed methods.** A NO-GO here means
  the method did not earn a more expensive test. It does not show the method fails everywhere,
  and it says nothing about quadrupeds.

## Per study

- **FBR.** One capsule per world. Reconstruction is not identification: frictions from 0.04 to
  0.24 reproduced the same toy v1 failure, and the hidden friction fell inside the reproduced
  range in only 1 of 6 toy v2 worlds.
- **Precursor.** The optimum at T-1.5 s is a property of this toy's speeds and patch geometry. The
  stage-1 sweep that chose it was not charged to the replay arms, as registered.
- **Recoverability.** Exploratory: it reused outcomes already seen, so it could refute but not
  confirm. Its candidate states were built around time offsets, which may favour the time model,
  and its high-recoverability states are confounded with position past the hazard.
- **FCSI.** 29 instances; three known-mechanism families; unknown mechanisms from a small set. The
  exploratory sparse selector was never preregistered and has no abstention.
- **Active diagnosis.** A small probe family of full slow crossings (the toy policy cannot stand
  still); 12 unknown-mechanism instances from two families. A finer, lower-energy probe space
  could behave differently. No probe was admissible on the latency instances.

## Reproduction

- Three full runs were not repeated during archiving (FBR toy v2, the precursor sweep, FCSI);
  their `--quick` software checks were. See [`../EVIDENCE.md`](../EVIDENCE.md#reproduction-checks).
- The recoverability provenance permutation p varies slightly with Python's hash seed
  (0.070 to 0.078 observed); the gate outcome does not.
