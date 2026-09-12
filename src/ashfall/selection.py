"""Checkpoint selection, held-out locking and immutable verdicts.

    training -> candidate checkpoints -> validation selection
             -> freeze candidate SHA256 -> one held-out evaluation
             -> immutable verdict

The v2 audit found that held-out checkpoint selection left no trace: the
verdict function was pure and re-runnable, the candidate id was a free string,
and the validation split existed without a consumer. This module closes that.

* The selection algorithm is part of the preregistered protocol
  (:class:`SelectionProtocol` has a content hash), and selection refuses to
  read validation metrics whose data hash equals the held-out data hash.
* Every held-out evaluation is registered in an append-only, hash-chained
  ledger BEFORE it runs, so an aborted or unfavourable evaluation is on record.
  The ledger refuses to evaluate the same candidate twice and refuses a second
  candidate under the same protocol and held-out data unless a preregistered
  amendment id is supplied, once.
* The verdict carries the hashes it rests on (protocol, Ashfall commit,
  Phoenix commit, policy, validation data, held-out data, configs), is written
  once, and can be re-verified from disk.

Silent repeated screening cannot be made impossible by software alone; what
this module guarantees is that it cannot happen without leaving a record that
contradicts the preregistration.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ashfall.provenance import content_hash, file_hash, write_artifact

SELECTION_RULES: tuple[str, ...] = (
    "max_validation_metric",
    "min_validation_incidence",
    "last_checkpoint",
)
TIE_BREAKS: tuple[str, ...] = ("earliest", "latest")
VERDICT_STATUSES: tuple[str, ...] = ("PASS", "FAIL", "UNINFORMATIVE")


class SelectionError(ValueError):
    """Selection or held-out registration would violate the preregistered protocol."""


def _hex64(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _hex40_or_64(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in (40, 64)
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a git SHA or SHA256 hex digest")
    return value


@dataclass(frozen=True)
class CandidateCheckpoint:
    sha256: str
    iteration: int
    path: str | None = None

    def __post_init__(self):
        _hex64(self.sha256, "sha256")
        if type(self.iteration) is not int or self.iteration < 0:
            raise ValueError("iteration must be a nonnegative integer")

    @classmethod
    def from_path(cls, path: str | Path, iteration: int) -> "CandidateCheckpoint":
        return cls(file_hash(path), iteration, str(path))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SelectionProtocol:
    """The preregistered selection rule. Its hash is part of the protocol hash."""

    rule: str
    metric: str | None
    tie_break: str
    validation_split_hash: str

    def __post_init__(self):
        if self.rule not in SELECTION_RULES:
            raise ValueError(f"unknown selection rule {self.rule!r}; expected {SELECTION_RULES}")
        if self.tie_break not in TIE_BREAKS:
            raise ValueError(f"unknown tie_break {self.tie_break!r}; expected {TIE_BREAKS}")
        if self.rule == "last_checkpoint":
            if self.metric is not None:
                raise ValueError("last_checkpoint reads no metric")
        elif not self.metric or not isinstance(self.metric, str):
            raise ValueError(f"rule {self.rule} needs a metric name")
        _hex64(self.validation_split_hash, "validation_split_hash")

    @property
    def protocol_hash(self) -> str:
        return content_hash(asdict(self))

    def to_dict(self) -> dict:
        return {**asdict(self), "protocol_hash": self.protocol_hash}


@dataclass(frozen=True)
class SelectionRecord:
    candidate_sha256: str
    candidate_iteration: int
    protocol_hash: str
    validation_data_hash: str
    considered: tuple[tuple[str, int], ...]
    metric_values: Mapping[str, float]
    selected_at: str
    record_hash: str = ""

    def __post_init__(self):
        _hex64(self.candidate_sha256, "candidate_sha256")
        _hex64(self.protocol_hash, "protocol_hash")
        _hex64(self.validation_data_hash, "validation_data_hash")
        if type(self.candidate_iteration) is not int or self.candidate_iteration < 0:
            raise ValueError("candidate_iteration must be a nonnegative integer")
        considered = tuple((str(s), int(i)) for s, i in self.considered)
        if not considered or self.candidate_sha256 not in {s for s, _ in considered}:
            raise ValueError("the selected candidate must be among those considered")
        object.__setattr__(self, "considered", considered)
        object.__setattr__(self, "metric_values", dict(sorted(self.metric_values.items())))
        datetime.fromisoformat(self.selected_at)
        expected = self.content_hash()
        if self.record_hash and self.record_hash != expected:
            raise ValueError("selection record hash does not match content")
        object.__setattr__(self, "record_hash", expected)

    def content_hash(self) -> str:
        data = asdict(self)
        data.pop("record_hash")
        return content_hash(data)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        return write_artifact(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "SelectionRecord":
        data = json.loads(Path(path).read_text())
        data["considered"] = tuple(tuple(c) for c in data["considered"])
        return cls(**data)


def select_candidate(
    candidates: Sequence[CandidateCheckpoint],
    validation_metrics: Mapping[str, float] | None,
    protocol: SelectionProtocol,
    *,
    held_out_data_hash: str,
) -> SelectionRecord:
    """Apply the preregistered rule; refuse anything that touches held-out data."""
    _hex64(held_out_data_hash, "held_out_data_hash")
    if protocol.validation_split_hash == held_out_data_hash:
        raise SelectionError(
            "the validation split hash equals the held-out data hash: selecting on it would be "
            "held-out checkpoint screening"
        )
    candidates = list(candidates)
    if not candidates:
        raise SelectionError("no candidate checkpoints")
    shas = [c.sha256 for c in candidates]
    if len(set(shas)) != len(shas):
        raise SelectionError("duplicate candidate checkpoint")
    if len({c.iteration for c in candidates}) != len(candidates):
        raise SelectionError("two candidates share an iteration")
    metrics: dict[str, float] = {}
    if protocol.rule == "last_checkpoint":
        if validation_metrics:
            raise SelectionError(
                "last_checkpoint takes no validation metrics; passing them is a leak"
            )
        chosen = max(candidates, key=lambda c: c.iteration)
    else:
        if validation_metrics is None:
            raise SelectionError(f"rule {protocol.rule} needs validation metrics")
        missing = [c.sha256 for c in candidates if c.sha256 not in validation_metrics]
        if missing:
            raise SelectionError(f"validation metric missing for candidates {missing}")
        extra = sorted(set(validation_metrics) - set(shas))
        if extra:
            raise SelectionError(f"validation metrics for unknown checkpoints {extra}")
        for sha, value in validation_metrics.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise SelectionError(f"non-finite validation metric for {sha}")
            metrics[sha] = float(value)
        sign = 1.0 if protocol.rule == "max_validation_metric" else -1.0
        best = max(sign * metrics[c.sha256] for c in candidates)
        tied = [c for c in candidates if sign * metrics[c.sha256] == best]
        chosen = (min if protocol.tie_break == "earliest" else max)(tied, key=lambda c: c.iteration)
    return SelectionRecord(
        chosen.sha256,
        chosen.iteration,
        protocol.protocol_hash,
        protocol.validation_split_hash,
        tuple((c.sha256, c.iteration) for c in sorted(candidates, key=lambda c: c.iteration)),
        metrics,
        datetime.now(timezone.utc).isoformat(),
    )


@dataclass(frozen=True)
class LedgerEntry:
    kind: str
    policy_sha256: str
    held_out_data_hash: str
    protocol_hash: str
    selection_record_hash: str
    registered_at: str
    previous_hash: str | None
    verdict_hash: str | None = None
    amendment_id: str | None = None
    opened_entry_hash: str | None = None
    entry_hash: str = ""

    def __post_init__(self):
        if self.kind not in ("opened", "closed"):
            raise ValueError("ledger entry kind must be opened or closed")
        for name in (
            "policy_sha256",
            "held_out_data_hash",
            "protocol_hash",
            "selection_record_hash",
        ):
            _hex64(getattr(self, name), name)
        if self.previous_hash is not None:
            _hex64(self.previous_hash, "previous_hash")
        if self.kind == "closed":
            if self.verdict_hash is None or self.opened_entry_hash is None:
                raise ValueError("a closing entry names the verdict and the entry it closes")
            _hex64(self.verdict_hash, "verdict_hash")
            _hex64(self.opened_entry_hash, "opened_entry_hash")
        elif self.verdict_hash is not None or self.opened_entry_hash is not None:
            raise ValueError("an opening entry has no verdict yet")
        datetime.fromisoformat(self.registered_at)
        expected = self.content_hash()
        if self.entry_hash and self.entry_hash != expected:
            raise ValueError("ledger entry hash does not match content")
        object.__setattr__(self, "entry_hash", expected)

    def content_hash(self) -> str:
        data = asdict(self)
        data.pop("entry_hash")
        return content_hash(data)

    def to_dict(self) -> dict:
        return asdict(self)


class HeldOutLedger:
    """Durable, append-only, hash-chained record of every held-out evaluation."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: list[LedgerEntry] = self._load() if self.path.exists() else []

    def _load(self) -> list[LedgerEntry]:
        entries: list[LedgerEntry] = []
        previous: str | None = None
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            entry = LedgerEntry(**json.loads(line))
            if entry.previous_hash != previous:
                raise ValueError(f"held-out ledger chain broken at {entry.entry_hash[:12]}")
            entries.append(entry)
            previous = entry.entry_hash
        return entries

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    def verify_chain(self) -> int:
        """Re-read the file and return the entry count; raises if the chain is broken."""
        return len(self._load()) if self.path.exists() else 0

    def _append(self, entry: LedgerEntry) -> LedgerEntry:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(entry.to_dict(), sort_keys=True, allow_nan=False) + "\n")
        self._entries.append(entry)
        return entry

    def evaluations(self, protocol_hash: str, held_out_data_hash: str) -> list[LedgerEntry]:
        return [
            e
            for e in self._entries
            if e.kind == "opened"
            and e.protocol_hash == protocol_hash
            and e.held_out_data_hash == held_out_data_hash
        ]

    def register(
        self,
        *,
        policy_sha256: str,
        held_out_data_hash: str,
        protocol_hash: str,
        selection_record: SelectionRecord,
        amendment_id: str | None = None,
    ) -> LedgerEntry:
        """Open a held-out evaluation. Call BEFORE running it."""
        _hex64(policy_sha256, "policy_sha256")
        _hex64(held_out_data_hash, "held_out_data_hash")
        _hex64(protocol_hash, "protocol_hash")
        if selection_record.candidate_sha256 != policy_sha256:
            raise SelectionError(
                "the policy offered for held-out evaluation is not the frozen candidate"
            )
        if selection_record.validation_data_hash == held_out_data_hash:
            raise SelectionError("the candidate was selected on the held-out data")
        prior = self.evaluations(protocol_hash, held_out_data_hash)
        if any(e.policy_sha256 == policy_sha256 for e in prior):
            raise SelectionError(
                f"candidate {policy_sha256[:12]} has already been evaluated on this held-out set "
                "under this protocol; a second look is repeated screening"
            )
        if prior:
            if not amendment_id:
                raise SelectionError(
                    f"{len(prior)} candidate(s) already evaluated on this held-out set under this "
                    "protocol; another needs a preregistered amendment_id"
                )
            if any(e.amendment_id == amendment_id for e in self._entries):
                raise SelectionError(f"amendment {amendment_id!r} has already been spent")
        elif amendment_id:
            raise SelectionError(
                "an amendment is not needed for the first evaluation; do not spend it"
            )
        return self._append(
            LedgerEntry(
                "opened",
                policy_sha256,
                held_out_data_hash,
                protocol_hash,
                selection_record.record_hash,
                datetime.now(timezone.utc).isoformat(),
                self._entries[-1].entry_hash if self._entries else None,
                amendment_id=amendment_id,
            )
        )

    def close(self, opened: LedgerEntry, verdict_hash: str) -> LedgerEntry:
        if opened.kind != "opened" or opened not in self._entries:
            raise SelectionError("close needs an opened entry from this ledger")
        if any(
            e.kind == "closed" and e.opened_entry_hash == opened.entry_hash for e in self._entries
        ):
            raise SelectionError("this evaluation is already closed")
        return self._append(
            LedgerEntry(
                "closed",
                opened.policy_sha256,
                opened.held_out_data_hash,
                opened.protocol_hash,
                opened.selection_record_hash,
                datetime.now(timezone.utc).isoformat(),
                self._entries[-1].entry_hash,
                verdict_hash=_hex64(verdict_hash, "verdict_hash"),
                opened_entry_hash=opened.entry_hash,
            )
        )

    def open_evaluations(self) -> list[LedgerEntry]:
        closed = {e.opened_entry_hash for e in self._entries if e.kind == "closed"}
        return [e for e in self._entries if e.kind == "opened" and e.entry_hash not in closed]


@dataclass(frozen=True)
class ImmutableVerdict:
    """A held-out verdict and every hash it rests on."""

    hypothesis: str
    status: str
    protocol_hash: str
    selection_record_hash: str
    ashfall_sha: str
    phoenix_sha: str
    policy_sha256: str
    validation_data_hash: str
    held_out_data_hash: str
    config_hashes: Mapping[str, str]
    metrics: Mapping[str, Any]
    ledger_entry_hash: str
    issued_at: str
    verdict_hash: str = ""

    def __post_init__(self):
        if not self.hypothesis or not isinstance(self.hypothesis, str):
            raise ValueError("verdict needs a hypothesis id")
        if self.status not in VERDICT_STATUSES:
            raise ValueError(f"unknown verdict status {self.status!r}; expected {VERDICT_STATUSES}")
        for name in (
            "protocol_hash",
            "selection_record_hash",
            "policy_sha256",
            "validation_data_hash",
            "held_out_data_hash",
            "ledger_entry_hash",
        ):
            _hex64(getattr(self, name), name)
        _hex40_or_64(self.ashfall_sha, "ashfall_sha")
        _hex40_or_64(self.phoenix_sha, "phoenix_sha")
        if self.validation_data_hash == self.held_out_data_hash:
            raise ValueError("validation and held-out data are the same data")
        if not isinstance(self.config_hashes, Mapping) or not self.config_hashes:
            raise ValueError("config_hashes must name at least one configuration")
        hashes = {str(k): _hex64(v, f"config_hashes[{k!r}]") for k, v in self.config_hashes.items()}
        object.__setattr__(self, "config_hashes", dict(sorted(hashes.items())))
        if not isinstance(self.metrics, Mapping):
            raise ValueError("metrics must be a mapping")
        json.dumps(dict(self.metrics), allow_nan=False)
        object.__setattr__(self, "metrics", dict(self.metrics))
        datetime.fromisoformat(self.issued_at)
        expected = self.content_hash()
        if self.verdict_hash and self.verdict_hash != expected:
            raise ValueError("verdict hash does not match content; the verdict was edited")
        object.__setattr__(self, "verdict_hash", expected)

    def content_hash(self) -> str:
        data = asdict(self)
        data.pop("verdict_hash")
        return content_hash(data)

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        return write_artifact(path, self.to_dict())

    @classmethod
    def verify(cls, path: str | Path) -> "ImmutableVerdict":
        """Re-read a verdict and refuse it if any field was changed after writing."""
        data = json.loads(Path(path).read_text())
        stored = data.get("verdict_hash")
        if not stored:
            raise ValueError("stored verdict carries no hash")
        return cls(**data)


def issue_verdict(
    *,
    hypothesis: str,
    status: str,
    protocol: SelectionProtocol,
    selection_record: SelectionRecord,
    ledger: HeldOutLedger,
    opened: LedgerEntry,
    ashfall_sha: str,
    phoenix_sha: str,
    held_out_data_hash: str,
    config_hashes: Mapping[str, str],
    metrics: Mapping[str, Any],
    path: str | Path,
) -> ImmutableVerdict:
    """Issue and write the verdict, then close the ledger entry with its hash."""
    if opened.policy_sha256 != selection_record.candidate_sha256:
        raise SelectionError("ledger entry and selection record name different candidates")
    if opened.selection_record_hash != selection_record.record_hash:
        raise SelectionError("ledger entry was opened for a different selection record")
    verdict = ImmutableVerdict(
        hypothesis,
        status,
        protocol.protocol_hash,
        selection_record.record_hash,
        ashfall_sha,
        phoenix_sha,
        selection_record.candidate_sha256,
        selection_record.validation_data_hash,
        held_out_data_hash,
        config_hashes,
        metrics,
        opened.entry_hash,
        datetime.now(timezone.utc).isoformat(),
    )
    verdict.write(path)
    ledger.close(opened, verdict.verdict_hash)
    return verdict


__all__ = [
    "SELECTION_RULES",
    "TIE_BREAKS",
    "VERDICT_STATUSES",
    "CandidateCheckpoint",
    "HeldOutLedger",
    "ImmutableVerdict",
    "LedgerEntry",
    "SelectionError",
    "SelectionProtocol",
    "SelectionRecord",
    "issue_verdict",
    "select_candidate",
]
