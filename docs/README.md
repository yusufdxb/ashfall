# Ashfall documentation

The program is archived. Start with [`ARCHIVE.md`](ARCHIVE.md).

## Final record

| Document | Contents |
|---|---|
| [`ARCHIVE.md`](ARCHIVE.md) | objective, the five hypotheses, gates, results, why each stopped, final status |
| [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md) | technical lessons across the five studies |
| [`../EVIDENCE.md`](../EVIDENCE.md) | what is verified, what is not, and the archive reproduction checks |
| [`limitations.md`](limitations.md) | final limitations |
| [`claims_ledger.md`](claims_ledger.md) | the detailed claim-by-claim history (closed) |
| [`data_provenance.md`](data_provenance.md) | evidence bundles, fixture versus scientific data |
| [`PUBLIC_ARTIFACT_MANIFEST.md`](PUBLIC_ARTIFACT_MANIFEST.md) | every published artifact compared with the private archive; the one sanitized copy |
| [`RESEARCH_HISTORY.md`](RESEARCH_HISTORY.md) | the private research commits that the records cite |

## The five studies

Result records and transition notes are frozen: their bytes are pinned by the test suite and they
are never edited. Preregistrations are kept exactly as registered.

| Study | Preregistration or plan | Related-work audit | Result |
|---|---|---|---|
| 1. FBR | [`TOY_V2_PREREGISTRATION.md`](research/TOY_V2_PREREGISTRATION.md) (the toy gate); [`PREREGISTRATION.md`](research/PREREGISTRATION.md) (the GO2 study it gated, never run) | [`RELATED_WORK.md`](research/RELATED_WORK.md) | [`FBR_TOY_NEGATIVE_RESULT.md`](results/FBR_TOY_NEGATIVE_RESULT.md) |
| 2. Precursor | [`PRECURSOR_SWEEP_PREREGISTRATION.md`](research/PRECURSOR_SWEEP_PREREGISTRATION.md) | | [`PRECURSOR_SWEEP_RESULT.md`](results/PRECURSOR_SWEEP_RESULT.md) |
| 3. Recoverability | [`RECOVERABILITY_RETROSPECTIVE_PLAN.md`](research/RECOVERABILITY_RETROSPECTIVE_PLAN.md) (exploratory) | [`RECOVERABILITY_RELATED_WORK.md`](research/RECOVERABILITY_RELATED_WORK.md) | [`RECOVERABILITY_RETROSPECTIVE_RESULT.md`](results/RECOVERABILITY_RETROSPECTIVE_RESULT.md) |
| 4. FCSI | [`FCSI_TOY_PREREGISTRATION.md`](research/FCSI_TOY_PREREGISTRATION.md) | [`FCSI_RELATED_WORK.md`](research/FCSI_RELATED_WORK.md) | [`FCSI_TOY_RESULT.md`](results/FCSI_TOY_RESULT.md) |
| 5. Active diagnosis | [`ACTIVE_DIAGNOSIS_PREREGISTRATION.md`](research/ACTIVE_DIAGNOSIS_PREREGISTRATION.md) | [`ACTIVE_DIAGNOSIS_RELATED_WORK.md`](research/ACTIVE_DIAGNOSIS_RELATED_WORK.md) | [`ACTIVE_DIAGNOSIS_TOY_RESULT.md`](results/ACTIVE_DIAGNOSIS_TOY_RESULT.md) |

Transition notes, each written before the next study, left the earlier verdicts untouched:
[after FBR and precursor](results/ASHFALL_HYPOTHESIS_TRANSITION.md),
[after recoverability](research/FCSI_HYPOTHESIS_TRANSITION.md),
[after FCSI](results/ACTIVE_DIAGNOSIS_HYPOTHESIS_TRANSITION.md).

FBR design records, marked historical: [`RESEARCH_QUESTION.md`](research/RESEARCH_QUESTION.md),
[`FBR_METHOD.md`](research/FBR_METHOD.md) (includes the toy v1 and v2 write-up),
[`EXPERIMENT.md`](research/EXPERIMENT.md).

## Legacy

[`legacy/`](legacy/README.md) holds superseded designs, audits, runbooks, protocols and the
pre-FBR Phase-I record, byte-identical to their originals.
