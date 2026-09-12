# Controlled low-friction counterexample protocol

Status: protocol and software preparation. No physical Ashfall repair or re-test result is asserted. On the stock GO2, slip and collapse are unsupported phenotypes: there is no calibrated per-foot contact and no validated ground-relative height (`ashfall.ontology.PHENOTYPES`), so a hardware capture can label a tracking stall or an attitude excursion but not a verified slip or collapse.

## Initial question and scope

Use one short, low-speed straight-line locomotion task on a controlled surface. Seek a repeatable tracking/slip event that can be stopped before a fall. A fall is not required for a useful counterexample. Begin with a reference high-friction surface and one securely fixed lower-friction test surface. Record actual material, dimensions, placement, measured slope and surface condition. Do not use loose coverings, liquid lubricants or a surface that can translate under the robot.

## Pre-session requirements

Use Phoenix's normal deployment, observation/action parity, motor-limit and emergency-stop checks. Confirm a tested independent stop operator, clear exclusion area, agreed abort criteria and laboratory-approved support method. Use only an approved quadruped support arrangement that does not entangle feet or materially alter normal contact. Record whether support load influenced the trajectory. Do not release or bypass any deployment safety gate to collect a failure.

Freeze baseline checkpoint hash, command profile, control period, robot configuration, detector/config versions and planned repetition order. Synchronize robot logs with an external video time reference. Confirm that pose, orientation, joint states, root velocities, command and action channels are available with declared frames. Root position from an unverified estimator must be marked estimated; missing position cannot silently become zero.

## Acquisition

1. Perform reference-surface trials at the initial low-speed command. Verify stop and timestamp alignment before challenge trials.
2. Run one operator-supervised challenge at the same command. Change one measured condition at a time within the laboratory-approved envelope.
3. Abort on unsafe attitude/height, unexpected acceleration, surface movement, loss of communication, motor-limit activation or operator concern. Count the intervention as an outcome; do not discard the episode.
4. Retain the complete recording, including stable pre-onset data and the post-event interval. Prefer at least one second before event onset so the 0.5-second reconstruction window is available despite detector latency.
5. Record operator interventions, their timing, disturbance end time, commanded stop and sustained recovery. Keep aborted and nonfailure trials as negatives with labels.
6. Independently review the video and telemetry. Label observed event onset, mechanism confidence and alternatives such as wall blockage, normal gait flight or estimator error. Uncertain slip mechanism remains uncertain.

## Recording and capsule review

Phoenix's ROS trajectory logger records available signals; Ashfall `capture` archives an existing recording and its hash. It does not initiate robot motion. `capsule build` converts labels to versioned capsules. Review policy identity, onset timing, pre-onset row, quaternion ordering, body/world velocity frames, command, contact observations and missing context. Torque/current/IMU/context fields may remain null. Missing reset state blocks simulator reconstruction.

Never put network settings, credentials, local connection instructions or private device identifiers in public data. Release pseudonymous robot/config identity and physically relevant measured parameters. Keep raw-video participant consent and laboratory data rules separate from research artifacts.

## Simulator identification and repair

Search declared unknown friction parameters first, with the baseline checkpoint fixed. Freeze terrain material/contact settings and record every candidate. Compare event mode, latency and measured tracking/attitude channels. A visually similar termination alone is insufficient. Mark failed reconstructions `UNREPRODUCED`, retain them, and do not train or claim repair of them.

After a reproducible candidate is found, freeze basin train/validation/held-out scenarios and the nominal suite. Run the implementation/runtime gates before the A-E comparison. Select training seeds independently of outcomes. Evaluate every candidate on the same frozen target and nominal scenarios. Export only after the explicit regression verdict accepts it.

## Physical re-test

Repeat the original reference and challenge protocol using baseline and accepted candidate with randomized, logged order. Match command, surface, payload, battery operating range and support conditions. Predeclare a fixed repetition count and the effect size of practical interest. Retain all attempts, including aborts, interventions and missing-data episodes; report missing data explicitly.

Primary physical endpoint: matched target event incidence (the fraction of trials in which the event occurs) with confidence intervals; recurrence within a trial (event, sustained recovery, event again) is reported separately, alongside reference-task preservation. Secondary outcomes: tracking, intervention and sustained recovery time in seconds. A simulator frontier shift alone is not a physical repair result. Stop further exposure if the candidate introduces a new hazardous mode, even when the original slip label becomes less frequent.
