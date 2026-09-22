# Evidence

What this archive supports, and what it does not. Status words are strict: **verified** means
executed in this repository and observed, with the artifact or test named. Everything else is
**not verified**, whatever the older documents say about plans.

The detailed claim-by-claim history is [`docs/claims_ledger.md`](docs/claims_ledger.md) (closed
at archive). The project record is [`docs/ARCHIVE.md`](docs/ARCHIVE.md).

## Verified

All of it is CPU toy evidence (`toy_mechanism`): a cart-pole on a low-friction patch, where the
"real" system is a hidden simulator configuration.

| What | Evidence |
|---|---|
| Five NO-GO decisions under preregistered or pre-frozen gates | the five records in [`docs/results/`](docs/results) and their `result.json` / `stage2.json` / `analysis.json` / `eval_result.json` decision fields |
| The frozen artifacts are unchanged since their result commits | 44 SHA-256 hashes listed in the result records, recomputed by `tests/test_frozen_results.py`. 43 files here are byte-identical; one (`results/recoverability/retro_2227de1/analysis.json`) is a sanitized public copy, checked against [`docs/PUBLIC_ARTIFACT_MANIFEST.md`](docs/PUBLIC_ARTIFACT_MANIFEST.md), whose original hash equals the frozen value |
| The result records, preregistrations, analysis plan and transition notes are unchanged | `tests/test_recoverability_history.py`, `tests/test_archive_integrity.py` |
| Each preregistration was committed before its run | commit order in the private research history: `8f49f02` before `08adcff`; `3205f02` before `dfec130`; `2227de1` before `61bb95d`; `1b657c1` before `d49daeb`; `b86006d` before `4186bbf`. Those commits are not published; they are indexed in [`docs/RESEARCH_HISTORY.md`](docs/RESEARCH_HISTORY.md), and the order is attested by the private archive rather than checkable from this repository |
| Study code and statistics behave as specified | `pytest -q` (CPU only, no simulator); count at the archive commit below |
| The reruns listed below reproduce the frozen outputs | [Reproduction checks](#reproduction-checks) |

## Not verified

None of the following was run by any of the five studies, and nothing in this repository
supports a claim about it:

- transfer of any result to Isaac Lab;
- quadruped dynamics;
- cross-simulator behaviour;
- the Unitree GO2, in simulation or on hardware;
- simulator repair on a quadruped;
- policy retraining on a quadruped;
- physical active diagnosis or physical probe safety.

Older documents describe a GO2 simulation study, a hardware protocol and an Isaac Lab FBR adapter.
Those were designs gated behind the toy results. They were never run and will not be; their
documents are marked historical or kept under [`docs/legacy/`](docs/legacy).

Earlier Ashfall designs that predate the five studies did run in Isaac Lab: a Phase-I reset-file
curriculum (n=11 seeds, a null later classified as uninformative because the treatment was never
delivered) and a software smoke of the causal-reproduction machinery. Neither is evidence for or
against any of the five hypotheses. See [`docs/legacy/README.md`](docs/legacy/README.md).

## Reproduction checks

Run for the archive on the private archive branch (code at `4186bbf`; the code in this release
differs from it only in two comments and the version string), writing to a scratch directory, with `OMP_NUM_THREADS=1`. No study was rerun for the
public release itself.

| Study | What was rerun | Outcome |
|---|---|---|
| FBR toy v1 | full run, 150 s wall | `result.json`, `capsule.json`, `baseline.npy` byte-identical to the frozen post-fix run `results/fbr_toy/f3ccc1b_corrected/` |
| FBR toy v2 (the gate) | `--quick` software check only | exit 0; the full gate run was **not** repeated in this pass |
| Precursor sweep | `--quick` stage 1 and stage 2 only | exit 0; the full sweep was **not** repeated in this pass. During the original run the study itself re-derived the six toy v2 capsules byte-identical and reproduced toy v2 arm D exactly (`v2_entry_reproduction.all_equal = true`) |
| Recoverability | full run, 6.5 s wall | `dataset.json` and `recoverability_only.json` byte-identical. `analysis.json` identical except `code_commit` and one value, the provenance permutation p (see below). The public copy of `analysis.json` is sanitized (paths only) |
| FCSI | `--quick` software check only | exit 0; the full 1,058 s run was **not** repeated in this pass |
| Active diagnosis | full `dev` then `eval` | calibration and all 42 evaluation instances identical; only wall-clock fields differ, plus the calibration file hash, which covers its own wall-clock field |

**Known nondeterminism: the recoverability provenance p.** `provenance_permutation` in
`src/ashfall/recoverability/analysis.py` permutes within worlds while iterating over
`set(worlds)`. The iteration order of a set of strings depends on Python's per-process hash seed,
so the random draws are consumed in a different order in each process. Observed: frozen
p = 0.0742; unpinned rerun 0.0704; `PYTHONHASHSEED=0` 0.0740 (twice, identical);
`PYTHONHASHSEED=1` 0.0775. Every other number in `analysis.json` is unaffected, and the gate
condition (p >= 0.05) holds in every run, so the decision does not change. The code is left as it
ran; iterating over `sorted(set(worlds))` would make future reruns deterministic.

## Public release checks

This release was built from the previously public history plus new commits, not from the private
research branches. `tests/test_public_release.py` checks, on every run:

- no commit reachable from the release contains any of the three excluded planning documents
  (skipped on shallow clones, where history is incomplete);
- every file under `results/` is listed in `results/PUBLIC_ARTIFACT_MANIFEST.json` with its public
  SHA-256, and sanitized entries match the frozen original hash in their result record;
- no tracked text file contains an absolute home-directory path, except three legacy files that were
  already public before this release (listed in the manifest);
- every cited commit is either reachable or listed in the research history index.

## Test suite at release

Run on the release branch before tagging (Python 3.10, CPU only):

| Check | Result |
|---|---|
| `pytest -q` | 978 passed. Warnings: only the pre-existing t-table fallback `RuntimeWarning` in `tests/test_multiseed_combined.py::test_t_975_covers_n_up_to_7`, which pytest reports as 1 or 2 depending on the run; with every other warning category raised as an error, 978 still pass |
| `pytest -q -m guard` | 35 passed |
| `ruff check .` | all checks passed |
| `mypy` (typed core, 12 files as configured in `pyproject.toml`) | no issues |
