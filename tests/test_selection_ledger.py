"""Validation selection, held-out locking, and verdicts that cannot be edited."""

from __future__ import annotations

import json

import pytest

from ashfall.selection import (
    CandidateCheckpoint,
    HeldOutLedger,
    ImmutableVerdict,
    SelectionError,
    SelectionProtocol,
    SelectionRecord,
    issue_verdict,
    select_candidate,
)

VAL = "1" * 64
HELD = "2" * 64
GIT_A = "3" * 40
GIT_P = "4" * 40


def candidates():
    return [CandidateCheckpoint(("%02x" % i) * 32, i * 100) for i in range(1, 5)]


def protocol(rule="max_validation_metric", tie_break="earliest"):
    return SelectionProtocol(
        rule, None if rule == "last_checkpoint" else "validation_success", tie_break, VAL
    )


class TestSelection:
    def test_rules_and_tie_breaks(self):
        cands = candidates()
        metrics = {c.sha256: v for c, v in zip(cands, (0.5, 0.9, 0.9, 0.7))}
        early = select_candidate(cands, metrics, protocol(), held_out_data_hash=HELD)
        late = select_candidate(
            cands, metrics, protocol(tie_break="latest"), held_out_data_hash=HELD
        )
        assert early.candidate_iteration == 200 and late.candidate_iteration == 300
        low = select_candidate(
            cands, metrics, protocol("min_validation_incidence"), held_out_data_hash=HELD
        )
        assert low.candidate_iteration == 100
        last = select_candidate(cands, None, protocol("last_checkpoint"), held_out_data_hash=HELD)
        assert last.candidate_iteration == 400 and last.metric_values == {}
        assert early.protocol_hash == protocol().protocol_hash
        assert early.record_hash and early.validation_data_hash == VAL

    def test_protocol_hash_changes_with_rule(self):
        assert protocol().protocol_hash != protocol(tie_break="latest").protocol_hash
        with pytest.raises(ValueError, match="reads no metric"):
            SelectionProtocol("last_checkpoint", "x", "earliest", VAL)
        with pytest.raises(ValueError, match="needs a metric"):
            SelectionProtocol("max_validation_metric", None, "earliest", VAL)

    @pytest.mark.guard
    def test_guard_selecting_on_held_out_data_is_refused(self):
        """Reintroduce the defect: point the selection rule at the held-out split."""
        cands = candidates()
        metrics = {c.sha256: 0.5 for c in cands}
        leaky = SelectionProtocol("max_validation_metric", "validation_success", "earliest", HELD)
        with pytest.raises(SelectionError, match="held-out checkpoint screening"):
            select_candidate(cands, metrics, leaky, held_out_data_hash=HELD)
        with pytest.raises(SelectionError, match="leak"):
            select_candidate(cands, metrics, protocol("last_checkpoint"), held_out_data_hash=HELD)

    def test_metrics_must_cover_exactly_the_candidates(self):
        cands = candidates()
        with pytest.raises(SelectionError, match="missing"):
            select_candidate(cands, {cands[0].sha256: 1.0}, protocol(), held_out_data_hash=HELD)
        metrics = {c.sha256: 1.0 for c in cands}
        metrics["f" * 64] = 2.0
        with pytest.raises(SelectionError, match="unknown checkpoints"):
            select_candidate(cands, metrics, protocol(), held_out_data_hash=HELD)
        metrics = {c.sha256: float("nan") for c in cands}
        with pytest.raises(SelectionError, match="non-finite"):
            select_candidate(cands, metrics, protocol(), held_out_data_hash=HELD)
        with pytest.raises(SelectionError, match="no candidate"):
            select_candidate([], {}, protocol(), held_out_data_hash=HELD)

    def test_record_round_trip_and_tamper(self, tmp_path):
        cands = candidates()
        record = select_candidate(
            cands, {c.sha256: 1.0 for c in cands}, protocol(), held_out_data_hash=HELD
        )
        path = record.save(tmp_path / "selection.json")
        assert SelectionRecord.load(path) == record
        data = json.loads(path.read_text())
        data["candidate_iteration"] = 999
        with pytest.raises(ValueError):
            SelectionRecord(**{**data, "considered": tuple(tuple(c) for c in data["considered"])})

    def test_candidate_from_path(self, tmp_path):
        p = tmp_path / "model.pt"
        p.write_bytes(b"weights")
        c = CandidateCheckpoint.from_path(p, 10)
        assert len(c.sha256) == 64 and c.path == str(p)


def selected(cands=None):
    cands = cands or candidates()
    return select_candidate(
        cands, {c.sha256: 1.0 for c in cands}, protocol(), held_out_data_hash=HELD
    )


class TestLedger:
    def test_register_close_and_chain(self, tmp_path):
        ledger = HeldOutLedger(tmp_path / "held_out.jsonl")
        record = selected()
        opened = ledger.register(
            policy_sha256=record.candidate_sha256,
            held_out_data_hash=HELD,
            protocol_hash=record.protocol_hash,
            selection_record=record,
        )
        assert ledger.open_evaluations() == [opened]
        closed = ledger.close(opened, "9" * 64)
        assert closed.previous_hash == opened.entry_hash and ledger.open_evaluations() == []
        reloaded = HeldOutLedger(ledger.path)
        assert reloaded.entries == ledger.entries and reloaded.verify_chain() == 2
        with pytest.raises(SelectionError, match="already closed"):
            ledger.close(opened, "9" * 64)

    def test_tampered_chain_is_detected(self, tmp_path):
        ledger = HeldOutLedger(tmp_path / "l.jsonl")
        record = selected()
        ledger.register(
            policy_sha256=record.candidate_sha256,
            held_out_data_hash=HELD,
            protocol_hash=record.protocol_hash,
            selection_record=record,
        )
        lines = ledger.path.read_text().splitlines()
        entry = json.loads(lines[0])
        entry["policy_sha256"] = "e" * 64
        ledger.path.write_text(json.dumps(entry) + "\n")
        with pytest.raises(ValueError, match="hash does not match"):
            HeldOutLedger(ledger.path)

    @pytest.mark.guard
    def test_guard_repeated_held_out_screening_is_refused(self, tmp_path):
        """Reintroduce the defect: evaluate a second candidate on the same held-out set."""
        ledger = HeldOutLedger(tmp_path / "l.jsonl")
        cands = candidates()
        first = selected(cands)
        ledger.register(
            policy_sha256=first.candidate_sha256,
            held_out_data_hash=HELD,
            protocol_hash=first.protocol_hash,
            selection_record=first,
        )
        # Same candidate again: repeated screening.
        with pytest.raises(SelectionError, match="already been evaluated"):
            ledger.register(
                policy_sha256=first.candidate_sha256,
                held_out_data_hash=HELD,
                protocol_hash=first.protocol_hash,
                selection_record=first,
            )
        # A different candidate under the same protocol and data: refused without amendment.
        other = SelectionRecord(
            cands[3].sha256,
            cands[3].iteration,
            first.protocol_hash,
            VAL,
            first.considered,
            first.metric_values,
            first.selected_at,
        )
        with pytest.raises(SelectionError, match="amendment_id"):
            ledger.register(
                policy_sha256=other.candidate_sha256,
                held_out_data_hash=HELD,
                protocol_hash=other.protocol_hash,
                selection_record=other,
            )
        ledger.register(
            policy_sha256=other.candidate_sha256,
            held_out_data_hash=HELD,
            protocol_hash=other.protocol_hash,
            selection_record=other,
            amendment_id="AMD-2026-01",
        )
        third = SelectionRecord(
            cands[2].sha256,
            cands[2].iteration,
            first.protocol_hash,
            VAL,
            first.considered,
            first.metric_values,
            first.selected_at,
        )
        with pytest.raises(SelectionError, match="already been spent"):
            ledger.register(
                policy_sha256=third.candidate_sha256,
                held_out_data_hash=HELD,
                protocol_hash=third.protocol_hash,
                selection_record=third,
                amendment_id="AMD-2026-01",
            )
        assert len(ledger.evaluations(first.protocol_hash, HELD)) == 2

    def test_register_refuses_mismatched_candidate_and_leaky_selection(self, tmp_path):
        ledger = HeldOutLedger(tmp_path / "l.jsonl")
        record = selected()
        with pytest.raises(SelectionError, match="not the frozen candidate"):
            ledger.register(
                policy_sha256="d" * 64,
                held_out_data_hash=HELD,
                protocol_hash=record.protocol_hash,
                selection_record=record,
            )
        with pytest.raises(SelectionError, match="selected on the held-out data"):
            ledger.register(
                policy_sha256=record.candidate_sha256,
                held_out_data_hash=VAL,
                protocol_hash=record.protocol_hash,
                selection_record=record,
            )
        with pytest.raises(SelectionError, match="do not spend it"):
            ledger.register(
                policy_sha256=record.candidate_sha256,
                held_out_data_hash=HELD,
                protocol_hash=record.protocol_hash,
                selection_record=record,
                amendment_id="unneeded",
            )


class TestImmutableVerdict:
    def verdict(self, tmp_path, status="PASS"):
        ledger = HeldOutLedger(tmp_path / "l.jsonl")
        record = selected()
        opened = ledger.register(
            policy_sha256=record.candidate_sha256,
            held_out_data_hash=HELD,
            protocol_hash=record.protocol_hash,
            selection_record=record,
        )
        verdict = issue_verdict(
            hypothesis="H2",
            status=status,
            protocol=protocol(),
            selection_record=record,
            ledger=ledger,
            opened=opened,
            ashfall_sha=GIT_A,
            phoenix_sha=GIT_P,
            held_out_data_hash=HELD,
            config_hashes={"env": VAL},
            metrics={"incidence": 0.12},
            path=tmp_path / "verdict.json",
        )
        return ledger, record, verdict

    def test_verdict_carries_every_hash_and_closes_the_ledger(self, tmp_path):
        ledger, record, verdict = self.verdict(tmp_path)
        assert verdict.policy_sha256 == record.candidate_sha256
        assert verdict.validation_data_hash == VAL and verdict.held_out_data_hash == HELD
        assert verdict.protocol_hash == protocol().protocol_hash
        assert verdict.selection_record_hash == record.record_hash
        assert verdict.ashfall_sha == GIT_A and verdict.phoenix_sha == GIT_P
        assert ledger.open_evaluations() == []
        assert ledger.entries[-1].verdict_hash == verdict.verdict_hash
        assert ImmutableVerdict.verify(tmp_path / "verdict.json") == verdict

    def test_written_once_and_tamper_detected(self, tmp_path):
        _, _, verdict = self.verdict(tmp_path)
        with pytest.raises(FileExistsError):
            ImmutableVerdict(**{**verdict.to_dict(), "verdict_hash": "", "status": "FAIL"}).write(
                tmp_path / "verdict.json"
            )
        data = json.loads((tmp_path / "verdict.json").read_text())
        data["status"] = "FAIL"
        (tmp_path / "verdict.json").write_text(json.dumps(data))
        with pytest.raises(ValueError, match="edited"):
            ImmutableVerdict.verify(tmp_path / "verdict.json")

    def test_verdict_validation(self):
        with pytest.raises(ValueError, match="unknown verdict status"):
            ImmutableVerdict(
                "H2",
                "MAYBE",
                VAL,
                VAL,
                GIT_A,
                GIT_P,
                VAL,
                VAL,
                HELD,
                {"e": VAL},
                {},
                VAL,
                "2026-01-01T00:00:00+00:00",
            )
        with pytest.raises(ValueError, match="same data"):
            ImmutableVerdict(
                "H2",
                "PASS",
                VAL,
                VAL,
                GIT_A,
                GIT_P,
                VAL,
                HELD,
                HELD,
                {"e": VAL},
                {},
                VAL,
                "2026-01-01T00:00:00+00:00",
            )
        with pytest.raises(ValueError, match="config_hashes"):
            ImmutableVerdict(
                "H2",
                "PASS",
                VAL,
                VAL,
                GIT_A,
                GIT_P,
                VAL,
                VAL,
                HELD,
                {},
                {},
                VAL,
                "2026-01-01T00:00:00+00:00",
            )
