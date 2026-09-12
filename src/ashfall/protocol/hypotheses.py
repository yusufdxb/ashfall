"""The gated hypothesis hierarchy and the ledger that enforces it.

    H0  causal delivery: a physical intervention induces the intended phenotype
        significantly more often than a matched no-intervention counterfactual.
    H1  phenotype fidelity: reproduced failures dynamically resemble
        independently labelled reference failures.
    H2  repair: failure-phenotype-conditioned adaptation reduces held-out
        failure incidence versus equal-compute non-conditioned alternatives.
    H3  generalization: repair generalizes to withheld causes and intensities
        that produce the same phenotype.
    H4  non-degradation: repair does not exceed a preregistered
        nominal-performance degradation margin.

The gate is mechanical. :meth:`HypothesisLedger.can_execute` refuses H(n) when
H(n-1) is not PASS, and refuses H2, H3 and H4 unless H0 is PASS for every
phenotype in the retained set. UNINFORMATIVE is a verdict, not a pass: the
Phase-I result was a correctly executed measurement of a treatment that was
never applied, and that outcome now has a name that cannot unlock anything.

Every verdict is appended to a JSONL ledger whose records form a hash chain,
so a verdict cannot be removed or edited without breaking the chain.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ashfall.ontology import PHENOTYPE_NAMES
from ashfall.provenance import content_hash

VERDICTS: tuple[str, ...] = ("PASS", "FAIL", "NOT_RUN", "UNINFORMATIVE")


@dataclass(frozen=True)
class HypothesisSpec:
    id: str
    name: str
    statement: str
    prerequisite: str | None
    endpoint: str
    per_phenotype: bool

    def to_dict(self) -> dict:
        return asdict(self)


HYPOTHESES: tuple[HypothesisSpec, ...] = (
    HypothesisSpec(
        "H0",
        "causal delivery",
        "A physical intervention induces the intended phenotype significantly more often than "
        "a matched no-intervention counterfactual (same restored state, policy, command, "
        "simulator seed, environment and timing).",
        None,
        "Per phenotype: paired treatment-versus-control phenotype incidence over matched "
        "pairs, exact test on discordant pairs, with D1, D2 and D3 gate rates reported.",
        True,
    ),
    HypothesisSpec(
        "H1",
        "phenotype fidelity",
        "Reproduced failures dynamically resemble independently labelled reference failures "
        "of the same phenotype.",
        "H0",
        "Per phenotype: distance between reproduced and reference trajectories on restorable "
        "channels against the reference-to-reference distribution; detector agreement with "
        "independent labels is reported separately.",
        True,
    ),
    HypothesisSpec(
        "H2",
        "repair",
        "Failure-phenotype-conditioned adaptation reduces held-out failure incidence versus "
        "equal-compute non-conditioned alternatives.",
        "H1",
        "Primary: held-out failure incidence aggregated over the retained phenotype set, "
        "paired by training seed, exact sign-flip permutation; per-phenotype incidences are "
        "secondary with Holm correction.",
        False,
    ),
    HypothesisSpec(
        "H3",
        "generalization",
        "Repair generalizes to withheld causes and intensities that produce the same phenotype.",
        "H2",
        "Held-out incidence on withheld intervention kinds and intensity bins, same pairing "
        "and test as H2.",
        False,
    ),
    HypothesisSpec(
        "H4",
        "non-degradation",
        "Repair does not exceed a preregistered nominal-performance degradation margin.",
        "H3",
        "Per nominal stratum: one-sided bound on success, tracking error and intervention "
        "rate against the registered margin; a degenerate interval cannot pass.",
        False,
    ),
)

HYPOTHESIS_IDS: tuple[str, ...] = tuple(h.id for h in HYPOTHESES)
_BY_ID: Mapping[str, HypothesisSpec] = {h.id: h for h in HYPOTHESES}


def hypothesis(hid: str) -> HypothesisSpec:
    try:
        return _BY_ID[hid]
    except KeyError:
        raise ValueError(f"unknown hypothesis {hid!r}; expected {HYPOTHESIS_IDS}") from None


def _hex64(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    return value


@dataclass(frozen=True)
class VerdictRecord:
    hypothesis: str
    verdict: str
    evidence_hash: str
    bundle_run_id: str
    recorded_at: str
    previous_hash: str | None
    phenotype: str | None = None
    amendment_id: str | None = None
    record_hash: str = ""

    def __post_init__(self):
        spec = hypothesis(self.hypothesis)
        if self.verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {self.verdict!r}; expected {VERDICTS}")
        _hex64(self.evidence_hash, "evidence_hash")
        if not isinstance(self.bundle_run_id, str) or not self.bundle_run_id:
            raise ValueError("a verdict must name the evidence bundle it rests on")
        if spec.per_phenotype and not self.phenotype:
            raise ValueError(f"{self.hypothesis} verdicts are recorded per phenotype")
        if not spec.per_phenotype and self.phenotype:
            raise ValueError(f"{self.hypothesis} is not a per-phenotype hypothesis")
        if self.phenotype is not None and self.phenotype not in PHENOTYPE_NAMES:
            raise ValueError(f"unknown phenotype {self.phenotype!r}")
        if self.previous_hash is not None:
            _hex64(self.previous_hash, "previous_hash")
        datetime.fromisoformat(self.recorded_at)
        expected = self.content_hash()
        if self.record_hash and self.record_hash != expected:
            raise ValueError("verdict record hash does not match its content")
        object.__setattr__(self, "record_hash", expected)

    def content_hash(self) -> str:
        data = asdict(self)
        data.pop("record_hash")
        return content_hash(data)

    def to_dict(self) -> dict:
        return asdict(self)


class HypothesisLedger:
    """Append-only, hash-chained JSONL of hypothesis verdicts."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._records: list[VerdictRecord] = []
        if self.path.exists():
            self._records = self._load()

    def _load(self) -> list[VerdictRecord]:
        records: list[VerdictRecord] = []
        previous: str | None = None
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            record = VerdictRecord(**json.loads(line))
            if record.previous_hash != previous:
                raise ValueError(f"hypothesis ledger chain broken at {record.record_hash[:12]}")
            records.append(record)
            previous = record.record_hash
        return records

    @property
    def records(self) -> tuple[VerdictRecord, ...]:
        return tuple(self._records)

    def verify_chain(self) -> bool:
        self._load() if self.path.exists() else None
        return True

    def current(self, hid: str, phenotype: str | None = None) -> VerdictRecord | None:
        hypothesis(hid)
        matching = [r for r in self._records if r.hypothesis == hid and r.phenotype == phenotype]
        return matching[-1] if matching else None

    def verdict_for(self, hid: str, phenotype: str | None = None) -> str:
        record = self.current(hid, phenotype)
        return "NOT_RUN" if record is None else record.verdict

    def record_verdict(
        self,
        hid: str,
        verdict: str,
        evidence_hash: str,
        bundle_run_id: str,
        *,
        phenotype: str | None = None,
        amendment_id: str | None = None,
    ) -> VerdictRecord:
        """Append a verdict. Changing an existing PASS/FAIL needs an amendment id."""
        existing = self.current(hid, phenotype)
        if existing is not None and existing.verdict in ("PASS", "FAIL") and not amendment_id:
            raise ValueError(
                f"{hid}{'/' + phenotype if phenotype else ''} already holds {existing.verdict}; "
                "re-recording needs a preregistered amendment_id"
            )
        if amendment_id is not None and any(r.amendment_id == amendment_id for r in self._records):
            raise ValueError(f"amendment {amendment_id!r} has already been used")
        record = VerdictRecord(
            hid,
            verdict,
            evidence_hash,
            bundle_run_id,
            datetime.now(timezone.utc).isoformat(),
            self._records[-1].record_hash if self._records else None,
            phenotype,
            amendment_id,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False) + "\n")
        self._records.append(record)
        return record

    def can_execute(
        self, hid: str, *, retained_phenotypes: Iterable[str] | None = None
    ) -> tuple[bool, str]:
        """Whether H(hid) may run now, and why not if it may not."""
        spec = hypothesis(hid)
        if spec.prerequisite is None:
            return True, "H0 is the first gate; nothing precedes it"
        if retained_phenotypes is None:
            return False, (
                f"{hid} needs the retained phenotype set; without it the per-phenotype H0 gate "
                "cannot be checked"
            )
        retained = tuple(retained_phenotypes)
        if not retained:
            return False, f"{hid} cannot run with an empty retained phenotype set"
        unknown = [p for p in retained if p not in PHENOTYPE_NAMES]
        if unknown:
            return False, f"unknown phenotypes in the retained set: {unknown}"
        for phenotype in retained:
            verdict = self.verdict_for("H0", phenotype)
            if verdict != "PASS":
                return False, (
                    f"H0 is {verdict} for retained phenotype {phenotype!r}; drop it from the "
                    f"retained set or deliver it before {hid}"
                )
        prerequisite = hypothesis(spec.prerequisite)
        if prerequisite.per_phenotype:
            for phenotype in retained:
                verdict = self.verdict_for(prerequisite.id, phenotype)
                if verdict != "PASS":
                    return False, (
                        f"{prerequisite.id} is {verdict} for retained phenotype {phenotype!r}, "
                        f"so {hid} cannot execute"
                    )
        else:
            verdict = self.verdict_for(prerequisite.id)
            if verdict != "PASS":
                return False, f"{prerequisite.id} is {verdict}, so {hid} cannot execute"
        return True, f"{spec.prerequisite} PASS for the retained set {list(retained)}"

    def assert_can_execute(self, hid: str, *, retained_phenotypes: Iterable[str] | None = None):
        allowed, reason = self.can_execute(hid, retained_phenotypes=retained_phenotypes)
        if not allowed:
            raise PermissionError(f"{hid} blocked: {reason}")
        return reason

    def summary(self) -> dict:
        out: dict[str, Any] = {}
        for spec in HYPOTHESES:
            if spec.per_phenotype:
                out[spec.id] = {
                    p: self.verdict_for(spec.id, p)
                    for p in PHENOTYPE_NAMES
                    if self.current(spec.id, p) is not None
                }
            else:
                out[spec.id] = self.verdict_for(spec.id)
        return out


__all__ = [
    "HYPOTHESES",
    "HYPOTHESIS_IDS",
    "VERDICTS",
    "HypothesisLedger",
    "HypothesisSpec",
    "VerdictRecord",
    "hypothesis",
]
