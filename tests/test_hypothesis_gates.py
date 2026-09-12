"""H0 to H4 as a mechanical gate: nothing downstream runs on a failed or missing prerequisite."""

from __future__ import annotations

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ashfall.ontology import FIRST_STUDY_PHENOTYPES
from ashfall.protocol.hypotheses import (
    HYPOTHESES,
    HYPOTHESIS_IDS,
    VERDICTS,
    HypothesisLedger,
    VerdictRecord,
    hypothesis,
)

EVIDENCE = "a" * 64


@pytest.fixture
def ledger(tmp_path):
    return HypothesisLedger(tmp_path / "hypotheses.jsonl")


def pass_h0(ledger, phenotypes=FIRST_STUDY_PHENOTYPES):
    for p in phenotypes:
        ledger.record_verdict("H0", "PASS", EVIDENCE, "run-h0", phenotype=p)


class TestHierarchy:
    def test_chain_is_linear_and_documented(self):
        assert HYPOTHESIS_IDS == ("H0", "H1", "H2", "H3", "H4")
        assert [h.prerequisite for h in HYPOTHESES] == [None, "H0", "H1", "H2", "H3"]
        assert all(h.statement and h.endpoint for h in HYPOTHESES)
        assert hypothesis("H0").per_phenotype and hypothesis("H1").per_phenotype
        assert not hypothesis("H2").per_phenotype
        with pytest.raises(ValueError, match="unknown hypothesis"):
            hypothesis("H9")

    def test_verdict_record_validation(self):
        with pytest.raises(ValueError, match="per phenotype"):
            VerdictRecord("H0", "PASS", EVIDENCE, "run", "2026-01-01T00:00:00+00:00", None)
        with pytest.raises(ValueError, match="not a per-phenotype"):
            VerdictRecord("H2", "PASS", EVIDENCE, "run", "2026-01-01T00:00:00+00:00", None, "slip")
        with pytest.raises(ValueError, match="unknown verdict"):
            VerdictRecord("H2", "OK", EVIDENCE, "run", "2026-01-01T00:00:00+00:00", None)
        with pytest.raises(ValueError, match="evidence bundle"):
            VerdictRecord("H2", "PASS", EVIDENCE, "", "2026-01-01T00:00:00+00:00", None)


class TestGate:
    def test_h0_is_always_executable(self, ledger):
        assert ledger.can_execute("H0")[0]

    def test_downstream_needs_the_retained_set(self, ledger):
        allowed, reason = ledger.can_execute("H1")
        assert not allowed and "retained phenotype set" in reason
        allowed, reason = ledger.can_execute("H1", retained_phenotypes=())
        assert not allowed and "empty" in reason

    @pytest.mark.guard
    def test_guard_h2_refused_when_any_retained_phenotype_lacks_h0_pass(self, ledger):
        """Reintroduce the defect: run the repair study while delivery is unproven."""
        retained = ("slip", "collapse")
        allowed, reason = ledger.can_execute("H2", retained_phenotypes=retained)
        assert not allowed and "H0 is NOT_RUN" in reason
        ledger.record_verdict("H0", "PASS", EVIDENCE, "run", phenotype="slip")
        ledger.record_verdict("H0", "FAIL", EVIDENCE, "run", phenotype="collapse")
        allowed, reason = ledger.can_execute("H2", retained_phenotypes=retained)
        assert not allowed and "H0 is FAIL for retained phenotype 'collapse'" in reason
        with pytest.raises(PermissionError, match="H2 blocked"):
            ledger.assert_can_execute("H2", retained_phenotypes=retained)
        # Dropping the failed phenotype from the retained set is the only way forward,
        # and H1 must still pass for the retained ones.
        allowed, reason = ledger.can_execute("H2", retained_phenotypes=("slip",))
        assert not allowed and "H1 is NOT_RUN" in reason

    def test_uninformative_is_not_a_pass(self, ledger):
        pass_h0(ledger)
        for p in FIRST_STUDY_PHENOTYPES:
            ledger.record_verdict("H1", "UNINFORMATIVE", EVIDENCE, "run", phenotype=p)
        allowed, reason = ledger.can_execute("H2", retained_phenotypes=FIRST_STUDY_PHENOTYPES)
        assert not allowed and "UNINFORMATIVE" in reason

    def test_full_chain_unlocks_in_order(self, ledger):
        retained = FIRST_STUDY_PHENOTYPES
        pass_h0(ledger)
        assert ledger.can_execute("H1", retained_phenotypes=retained)[0]
        assert not ledger.can_execute("H2", retained_phenotypes=retained)[0]
        for p in retained:
            ledger.record_verdict("H1", "PASS", EVIDENCE, "run", phenotype=p)
        assert ledger.can_execute("H2", retained_phenotypes=retained)[0]
        assert not ledger.can_execute("H3", retained_phenotypes=retained)[0]
        ledger.record_verdict("H2", "PASS", EVIDENCE, "run")
        assert ledger.can_execute("H3", retained_phenotypes=retained)[0]
        ledger.record_verdict("H3", "FAIL", EVIDENCE, "run")
        allowed, reason = ledger.can_execute("H4", retained_phenotypes=retained)
        assert not allowed and "H3 is FAIL" in reason
        assert ledger.summary()["H2"] == "PASS" and ledger.summary()["H0"]["slip"] == "PASS"

    def test_unknown_phenotype_in_retained_set(self, ledger):
        allowed, reason = ledger.can_execute("H1", retained_phenotypes=("tumble",))
        assert not allowed and "unknown" in reason


class TestLedgerPersistence:
    def test_chain_persists_and_detects_tampering(self, tmp_path):
        ledger = HypothesisLedger(tmp_path / "h.jsonl")
        pass_h0(ledger, ("slip",))
        ledger.record_verdict("H1", "PASS", EVIDENCE, "run", phenotype="slip")
        reloaded = HypothesisLedger(ledger.path)
        assert reloaded.records == ledger.records
        assert reloaded.records[1].previous_hash == reloaded.records[0].record_hash
        lines = ledger.path.read_text().splitlines()
        first = json.loads(lines[0])
        first["verdict"] = "FAIL"
        ledger.path.write_text("\n".join([json.dumps(first), lines[1]]) + "\n")
        with pytest.raises(ValueError):
            HypothesisLedger(ledger.path)
        # Deleting a record breaks the chain too.
        ledger.path.write_text(lines[1] + "\n")
        with pytest.raises(ValueError, match="chain broken"):
            HypothesisLedger(ledger.path)

    def test_settled_verdicts_need_an_amendment_to_change(self, ledger):
        ledger.record_verdict("H0", "FAIL", EVIDENCE, "run", phenotype="slip")
        with pytest.raises(ValueError, match="amendment_id"):
            ledger.record_verdict("H0", "PASS", EVIDENCE, "run2", phenotype="slip")
        ledger.record_verdict(
            "H0", "PASS", EVIDENCE, "run2", phenotype="slip", amendment_id="AMD-1"
        )
        assert ledger.verdict_for("H0", "slip") == "PASS"
        with pytest.raises(ValueError, match="already been used"):
            ledger.record_verdict(
                "H0", "FAIL", EVIDENCE, "run3", phenotype="slip", amendment_id="AMD-1"
            )
        # NOT_RUN and UNINFORMATIVE may be superseded without an amendment.
        ledger.record_verdict("H2", "UNINFORMATIVE", EVIDENCE, "run")
        ledger.record_verdict("H2", "PASS", EVIDENCE, "run4")

    @given(
        verdicts=st.lists(
            st.sampled_from([v for v in VERDICTS if v != "PASS"]), min_size=1, max_size=4
        )
    )
    @settings(
        max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_any_non_pass_h0_blocks_everything_downstream(self, tmp_path, verdicts):
        # A fresh ledger per example; the fixture directory is shared across examples.
        name = "h_" + "_".join(v[:1] for v in verdicts) + f"_{len(verdicts)}.jsonl"
        path = tmp_path / name
        if path.exists():
            path.unlink()
        ledger = HypothesisLedger(path)
        phenotypes = FIRST_STUDY_PHENOTYPES[: len(verdicts)]
        for p, v in zip(phenotypes, verdicts):
            ledger.record_verdict("H0", v, EVIDENCE, "run", phenotype=p)
        for hid in ("H1", "H2", "H3", "H4"):
            assert not ledger.can_execute(hid, retained_phenotypes=phenotypes)[0]
