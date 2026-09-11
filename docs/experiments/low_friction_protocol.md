# First counterexample experiment

This is a protocol, not a new locomotion result. Start with one reviewed low-friction capsule and a fixed baseline checkpoint. Do not launch the multi-seed campaign until the software and simulator reset/reproduction/episode smoke gates pass.

## Design

Compare A baseline checkpoint, B plain environment fine-tune, C generic domain randomization, D fixed failure-reset fraction and E frontier-guided repair. B-E start from the same baseline and receive equal fresh PPO transition budgets. Log any extra simulator computation used for reproduction or frontier probes separately. D and E should share the corrected pre-onset state/command/velocity adapter so their difference isolates scenario sampling. The original row-0 experiment remains a separate historical treatment; reproduce it only using its archived source version.

Preselect training seeds 11, 23 and 47. Preselect evaluation replicate seeds 1009, 1013 and 1019, with independent scenario-generation seeds. These are protocol constants, not seeds chosen for low baseline performance. Calibrate difficulty and practical power on a disjoint development set, then freeze final support and sample counts before held-out evaluation. The supplied small counts are software/simulator smoke sizes, not a powered research design.

For the first family, use one-dimensional robot-shape dynamic friction with static friction and terrain material fixed. Severity is `1-dynamic_friction`, dimensionless. It is a controlled simulator axis, not an estimate of the physical coefficient. Reproduction searches may add other dimensions only after their adapters, logging and identifiability assumptions are implemented. Record material combination behavior; do not equate robot-shape coefficient with effective contact friction.

## Endpoints and acceptance

Primary endpoints are target-mode recurrence, whole-episode failure/success, probability by severity, R50 and frontier shift. Report all mode changes; replacing a slip with a collapse is not success. Secondary endpoints are return, tracking, intervention rate and sustained recovery seconds with right-censoring.

Example frozen budgets: at least 5 percentage points of target recurrence reduction; no more than 2 percentage points of nominal success degradation; no more than 0.02 m/s tracking error increase; no intervention-rate increase. Test each nominal stratum independently. Use the lower target-improvement interval bound and appropriate one-sided degradation bound for acceptance. These example values require laboratory task justification before registration. An R50 endpoint needs its own physical units and practical effect margin, not a borrowed percentage-point threshold.

Within each training seed, evaluate exactly the same scenario IDs and replicate seeds across policies. Bootstrap scenario clusters rather than independent episodes. Across training seeds, report paired effects with independent-seed intervals. Report all registered seeds and all unsuccessful runs. Do not select the best checkpoint on held-out outcomes. Use validation for checkpoint choice, freeze the choice, and evaluate held-out once. If equivalence matters, register an explicit margin and use the implemented TOST decision; non-significance is insufficient.

## Execution boundary

The CLI sequence and required inputs are in the README. `capture` archives an already recorded Parquet. `reproduce` searches with the original policy. `basin` refuses mock/failed reproduction. `repair` refuses held-out scenarios and requires actual simulator evidence. `evaluate` emits episode records and companion basin observations. `verdict` consumes complete target and nominal episode artifacts under the frozen protocol.

The default nominal suite declares flat, rough, slope, turning, speed and modest push cases. Unsupported backend dimensions remain explicitly unsupported and block acceptance. A narrower first experiment may preregister a smaller supported nominal scope, but its claims must be limited to that scope. The small CPU demonstration does not establish that any of these locomotion tasks passes.
