# v3 baseline audit: state of the repository before the causal-repair redesign

Audited source: Ashfall `6601770` (`main` == `research/ashfall-mode-conditioned-adaptation`).
Simulator sibling: go2-phoenix `5783416` on `feat/causal-viability-replication`, the branch that
repository's own `docs/superseded_results.md` names as the source of truth. Phoenix `main`
(`61044e6`) is 86 commits behind that branch and carries deletions the branch deliberately did not
take, so it is not a valid integration target.

Method: every module under `src/ashfall/` and `tests/` read in full, the two prior audits
re-read, the CPU suite and linter run on this machine. Isaac Lab (4.5.22 on Isaac Sim 6.0) and a
GPU are available here; nothing in this audit was established by running them.

## Executed baseline

| check | result |
|---|---|
| `python3 -m pytest -q` | **5 failed, 320 passed** (the README claims 325 pass) |
| `ruff check .` | clean |
| `mypy` | never configured or run |
| CI | one job, Python 3.10, `pytest -m "not sim"` only; no lint, no type check, no matrix |

The five failures are one defect: Phoenix's `InitialState` gained four metadata fields
(`position_frame`, `position_frame_source`, `controller_history`, `environment_parameters`) in
commit `f4ccf08`. `tests/test_delivery.py` asserted set equality with every dataclass field, and
`backends/phoenix.py::_prepare_seed` iterated every dataclass field to build the restore state,
so it raised `AttributeError` on a `CapsuleFrame`. Four backend tests and the contract test failed
together. Nothing on either side declared which fields are the restore contract and which are
metadata about it. Fixed in this pass by `ashfall.backends.phoenix_compat`, which pins both lists
and checks the live sibling against them.

## Defects carried forward from the v2 audit, with disposition

The v2 implementation audit (`docs/audits/v2_implementation_audit.md`) listed roughly forty
findings. This table records which were already closed at `6601770` and which this redesign
owns. Item numbers refer to that document.

| items | status at 6601770 | owned by this pass |
|---|---|---|
| A1 `PhoenixBackend.evaluate` missing | open | yes: backend `evaluate` implemented |
| A3 `create_backend(**config)` splats provenance keys | open | yes: filtered factory |
| A4 verdict requires exact `environment_parameters` and a `nominal_group` field the outcome lacks | open | yes: evidence shape contract |
| A5 no example backend config or repair spec | open | yes: `configs/h0/` |
| B1, B4 dead pre-onset guard, falsified seed row pinned | closed by `8c1374b` (`delivery.py`) | superseded by matched-counterfactual delivery |
| B3 `n_stable=50` hardcoded | open | yes: fixtures parameterised and reclassified |
| B6 no distinctness or direction check | closed (single-feature z-test) | replaced: single feature is no longer the sole criterion |
| C1, C6 zero-filled capture accepted; provenance laundering | closed for `base_pos`; `onset_label_source` still defaults | yes: ontology label sources |
| C3, C4 `contact_loss` undeliverable, nothing records deliverability | closed (`deliverable` on taxonomy) | reframed: excluded from the first study as unsupported |
| D1 BCa sign | closed by `d9e89dd` | guard added so it cannot return |
| D2 `paired_delta` fails open | closed | guard added |
| D4 incidence called recurrence | open | yes: terminology fixed |
| D5, D6 `observed_crossing` from pooled block; plateau reported as a point | open | yes: frontier identification |
| D7 zero-width cluster bootstrap accepted as proof at ceiling | open | yes: degenerate intervals refused |
| D9 held-out selection leaves no trace | open | yes: selection protocol and ledger |
| D10 arms declarative, equal budget never verified | open | yes: compute budget enforcement |
| D11 multiplicity family split per terrain | open | yes: one primary endpoint, Holm on secondaries |
| E1 detector validation evidence not in the repo | open | yes: independent evaluation path |
| E2, E3, E4 threshold-plus-epsilon tests, mutation-blind negatives, coin-flip slip fixture | open | yes: mutation-sensitive evaluation |
| E5, E6 kinematically impossible fixtures, `failure_flag` is not the detector's decision | open | fixtures reclassified; harvesting from physics replaces them |
| E7 CI runs tests only | open | yes |
| E9 invariants enforce provenance, not physics | open | yes: D1 to D3 gates |

## Statements in the public README at 6601770 that this pass must retire or requalify

- "325 tests" (5 fail on a clean checkout against the pinned sibling).
- The taxonomy table's `Deliverable` column reads as a property of a failure mode. Under the
  ontology introduced here, deliverability is a property of an intervention on a backend, and a
  phenotype's observability is a property of a platform; the two were one column.
- "Whether a failure curriculum *delivers* its treatment can be measured. Established, on
  synthetic data." The measurement was a one-feature, own-prefix, 3-sigma test on hand-authored
  trajectories. It remains a useful regression fixture; it is not the delivery criterion.

## What is retained unchanged

- `results/legacy_row0_curriculum/**` is not modified, regenerated or reinterpreted at the file
  level. Its interpretation lives in `docs/legacy/README.md` and the claims ledger.
- The n=11 reproduction from the committed metrics (slippery -0.4155 pp, p=0.726562; rough
  +0.5485 pp, p=0.767578) remains pinned by `tests/test_multiseed_combined.py`.
- Content-addressed artifacts, write-once evidence, exact sign-flip permutation, split discipline.

## Phoenix-side findings made while integrating

Found on this machine (Isaac Lab 4.5.22 on Isaac Sim 6.0, rsl_rl 5.0.1) while
running the H0 smoke. None is fixed in go2-phoenix by this pass; Ashfall works
around the ones it touches.

| finding | status | consequence |
|---|---|---|
| `checkpoints/phoenix-flat-v4/latest.pt` (`model_4999.pt`) is diverged: learned action standard deviation parameter about 49, action maximum about 330 on a canonical standing observation, against about 0.2 to 0.3 and 3.1 for v3b | VERIFIED offline; the policy fell within a second in Isaac Lab | the `NEGATIVE_RESULT.md` table beside it (32 of 32 successes) cannot describe this file as it stands (INFERRED); H0 uses v3b |
| `FrictionScenarioAdapter._get` calls `.clone()` on the material view's return value, which is a `wp.array` on this build even when the robot exposes `root_physx_view` | VERIFIED (`AttributeError` in the probe) | Phoenix's `scenario_bridge` and `fine_tune` with `restore_environment_parameters` would fail here; Ashfall uses its own `MaterialWriter` |
| Articulation data buffers (`root_pos_w` and the rest) are Warp arrays; `np.asarray` on one raises | VERIFIED | any converter must go through the array's own `numpy()`; Phoenix's `state_adapter.as_numpy` does, the converter passed into `snapshot_manager_state` must too |
| None of the flat or rough runner checkpoints carry observation-normalizer statistics, while their train configs set `empirical_normalization: true` | VERIFIED for four checkpoints | rebuilding a runner from the YAML adds an untrained normaliser (a 1% observation shrink); the backend now resolves the flag from the checkpoint |
| v3b was trained in April; `DelayedDCMotor` latency, motor-strength randomisation and `RateLimitedJointPositionAction` were wired in June and are active when `configs/env/flat.yaml` is built today | VERIFIED by probe | nominal behaviour in the H0 smoke is not the trained behaviour; every smoke control rollout was flagged for attitude |
| The harvest script's usage example names flat-v4 with `flat_perturb.yaml`, and the published harvest report does not record its checkpoint; 74 of its 148 terminations left windows under two rows | INFERRED only | if that harvest used flat-v4, an immediately falling policy would explain the short windows; recording the checkpoint hash in the report would settle it |
