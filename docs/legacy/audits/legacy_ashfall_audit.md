# Legacy Ashfall implementation audit

Audited source snapshots: Ashfall `cf0b81ac3f8343851bf3b02d0373687d796ab0a8`, Phoenix `719ea52543fdf7ac0deb5245b1b13692db4e4ba8`. Findings below describe those snapshots, before the rebuild.

| Question | Classification | Ground truth and evidence |
|---|---|---|
| `failure_fraction` | IMPLEMENTED; TESTED | Ashfall experiment runner writes Phoenix `curriculum.failure_sample_fraction`. `FailureCurriculum.assign` chooses `round(number_of_reset_envs * fraction)` slots and uniform trajectory files. It is a reset fraction, with rounding dependent on reset batch size. |
| PPO effect | IMPLEMENTED; TESTED | `adaptation/fine_tune.py` installs a reset wrapper then calls `OnPolicyRunner.learn`. PPO collects fresh rollouts. No old trajectories enter PPO minibatches. Pool-composition prose describing minibatch dosage is false. |
| Reset row | IMPLEMENTED; TESTED | `reset_bridge.install` defaults to `first`, and `fine_tune.py` passes no options. `load_initial_state` defaults to row 0. Optional onset and onset-minus-k functions exist but are not wired by the training entry point. |
| Synthetic preframes | IMPLEMENTED; TESTED | `synth/generator.py` constructs stable preframes (default 50). Some event flags begin still later because of persistence thresholds. Row 0 is not failure onset. |
| Restored state | IMPLEMENTED; TESTED | Root position, quaternion (xyzw converted to wxyz), joint position and joint velocity. Environment origin added to position. |
| Command | NOT IMPLEMENTED | Reader loads `command_vel`; reset wrapper never sends it to command manager. The environment's independently sampled command remains. |
| Root velocities | NOT IMPLEMENTED in default training | Optional `write_velocity=False` default. The normal environment reset determines velocities. |
| Velocity frames | MISLEADING LEGACY CLAIM | Optional bridge and reconstruct paths send body-frame velocity directly to world-frame Isaac root velocity API. Wrong for rotated base states. A comment acknowledges the bug; tests did not establish the transform. |
| Physical context | NOT IMPLEMENTED | No logged terrain, friction, mass, motor state, latency, contact state or original disturbance is reconstructed by the reset bridge. Generic environment randomization is not inference of a failure mechanism. |
| Reconstruction script | IMPLEMENTED, not reproduction | `replay/reconstruct.py` uses row 0 and zero actions, averages friction variations across the scene and optionally applies mass deltas. No baseline-policy failure-similarity gate. |
| Episode data | IMPLEMENTED; TESTED, no new experiment verified | Current Phoenix `training/evaluate.py` optionally emits v1 episode records and telemetry. Historical n=11 artifacts retain aggregate metrics. README's blanket claim that the current evaluator cannot retain episodes is stale. Scenario identity, independently named seeds and recovered-state outcomes are absent. Terminal telemetry is sampled after auto-reset. |
| Metrics wiring | IMPLEMENTED; TESTED only | Ashfall runner consumes aggregate Phoenix metrics. `FailureAnalyzer` is unit tested, not wired into the historical experiments. `failure_recurrence=min(mode_event_count/episodes,1)` is not recurrence; recovery is event-free steps or an inter-event gap, not recovery of locomotion. |
| Detector evidence | TESTED | 18 synthetic trajectories are authored around detector thresholds. This is fixture coverage and integration correctness, not independent precision/recall validation. |
| n=11 evidence | EXPERIMENTALLY VERIFIED within historical scope | Saved aggregate metrics support paired comparisons of the historical reset-file treatment. They do not establish failure-onset-conditioned repair effectiveness or ineffectiveness. No physical counterexample was reproduced. |
| README conclusion | MISLEADING LEGACY CLAIM | “It does not work”, rejection of the failure-repair hypothesis, “validated contribution” for the detector and claims of a complete failure replay loop exceed the evidence. Non-significance is neither equivalence nor proof of ineffectiveness. |
| Seed guidance | MISLEADING LEGACY CLAIM | Choosing future seeds with low baseline success would condition the experiment on observed outcomes. Difficulty must be calibrated on a separate development set and evaluation seeds frozen independently. |

Baseline CPU validation: Ashfall `python3 -m pytest tests -q`: 173 passed, one pre-existing small-sample t-interval fallback warning. Phoenix baseline test outcome is recorded in the final scientific audit. This audit makes no new simulator or hardware claim.
