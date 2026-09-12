# Event capture and independent labeling

The machine-readable side of this procedure now lives in `ashfall.detector_eval`
and the evaluation protocol, dataset composition, metrics and mutation
precondition are specified in [validation_protocol.md](validation_protocol.md).
Labels produced under this document are `label_source="human_review"` there.

The threshold detector captures candidate events for review. It cannot infer a
failure mechanism from tracking error alone. In particular, wall blockage and
low friction can produce the same available telemetry. Contact dropout without
gait phase also confuses normal trot or flight with contact failure, and high
joint velocity alone cannot identify a stumble. An intended crouch can trigger
the collapse threshold. These limitations are exercised explicitly by
`tests/test_detector_labeling.py` and `ashfall.synth.negatives`.

The 18 historical synthetic Parquets provide schema integration, fixture
coverage, and threshold regression checks. They are not an independently labeled
detector evaluation set. No hardware precision, recall, or false positive rate
has been established by these fixtures.

## Review procedure

1. Freeze recorded trajectory bytes and compute their SHA256. Retain synchronized
   video when available and document its synchronization uncertainty privately.
2. Define disjoint windows using zero-based inclusive start/end indices. Include
   successful negatives, not only detector-triggered snippets. Sampling only
   triggers cannot estimate false negative rate or recall.
3. Have a reviewer inspect each window without model predictions. Annotate every
   supported mode and record ambiguity in notes. An empty modes list means a
   reviewed negative. Uncertain or unreviewed windows remain `reviewed: false`.
4. Record labels using `WindowLabel` and `save_labels`. A real reviewer identifier
   can be a study pseudonym; personal details are unnecessary. Independently
   review a subset and report disagreements before freezing evaluation labels.
5. Freeze a detector configuration/version separately from the evaluation set.
   Tune thresholds using separate development trajectories, never evaluation
   labels. Report the split and sampling procedure with results.
6. Run detector predictions for each frozen window and call
   `evaluate_window_predictions`. Report per-mode precision, recall, false
   positive rate, confusion matrices, window count, and trajectory count.

The result uses reviewed windows as its counting unit. Events within a window
are not independent trials. Multiple modes can co-occur, so each mode has a
one-vs-rest confusion matrix rather than an exclusive-class matrix. Overlapping
windows from one trajectory are rejected to prevent double counting; disjoint
windows within a trajectory are still correlated. Confidence intervals should
resample independent trajectories or physical trials. Undefined precision or
recall is reported as null rather than as perfect performance. Hardware,
simulation, and fixture labels must be evaluated in separate reports.

## Label record

Required JSONL fields are `schema_version` (`1.0`), `window_id`,
`trajectory_sha256` (64 lowercase hexadecimal characters), `start_index`,
`end_index`, `modes` (array of taxonomy mode strings), `label_source`
(`hardware_human`, `simulation_human`, or `synthetic_fixture`), `reviewer`,
`reviewed` (boolean), and optional `notes`.

Create a draft record using actual trajectory identity, window bounds, and a
study reviewer ID, with `modes=[]` and `reviewed=False`. Empty draft modes do not
constitute a negative label until review is complete. The evaluator refuses
unreviewed records, mismatched prediction IDs, duplicates, or mixed sources.

## Capsule onset review

`capsules_from_parquet` creates one capsule per contiguous mode-labeled event.
The inherited flag is recorded as `onset_label_source=legacy_failure_flag`.
This is a detector-associated row, not guaranteed disturbance onset. Reviewers
can provide `onset_indices` to explicitly choose the state transition to study.
Disturbance onset, detector emission, and physical failure onset should be
recorded separately in the event descriptor when available. Absence of a flag
produces no automatic capsules; it never silently substitutes row zero.

All original frame indices are retained in capsule JSON. The capsule's pre/post
window determines the reviewed segment, with an inclusive post-failure end.
The default reset targets 0.5 seconds before the labeled onset. A request outside
the reviewed window fails, prompting a new reviewed capsule window rather than
an unnoticed change to the experiment. Physical fields that were not recorded
are null, and missing mandatory reset fields prevent reset eligibility.
