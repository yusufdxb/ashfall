"""The FBR toy evidence is immutable: every indexed artifact must keep its hash.

The index lives in ``docs/results/FBR_TOY_NEGATIVE_RESULT.md`` and, for the precursor
sweep, in ``docs/results/PRECURSOR_SWEEP_RESULT.md`` once that exists. A later study
writes new directories; it never rewrites these files.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INDEXES = (
    "docs/results/FBR_TOY_NEGATIVE_RESULT.md",
    "docs/results/PRECURSOR_SWEEP_RESULT.md",
    "docs/results/RECOVERABILITY_RETROSPECTIVE_RESULT.md",
    "docs/results/FCSI_TOY_RESULT.md",
    "docs/results/ACTIVE_DIAGNOSIS_TOY_RESULT.md",
)
LINE = re.compile(r"^([0-9a-f]{64})  (results/\S+)$")


def _entries():
    out = []
    for index in INDEXES:
        for line in (ROOT / index).read_text().splitlines():
            m = LINE.match(line.strip())
            if m:
                out.append((m.group(2), m.group(1)))
    return out


def test_index_is_not_empty():
    assert len(_entries()) >= 30


def _sanitized():
    """Public-release copies whose machine-local path text was replaced (see the manifest)."""
    manifest = ROOT / "results/PUBLIC_ARTIFACT_MANIFEST.json"
    if not manifest.exists():
        return {}
    rows = json.loads(manifest.read_text())["artifacts"]
    return {r["path"]: r for r in rows if r["status"] == "SANITIZED_PATHS_ONLY"}


@pytest.mark.parametrize("path,digest", _entries())
def test_artifact_hash_unchanged(path, digest):
    data = (ROOT / path).read_bytes()
    sanitized = _sanitized().get(path)
    if sanitized is not None:
        # The frozen value is the private original; the public copy must be exactly the
        # manifest's sanitized file, and the manifest must name the frozen value as its source.
        assert sanitized["original_sha256"] == digest, f"{path}: manifest does not match the record"
        assert hashlib.sha256(data).hexdigest() == sanitized["public_sha256"], f"{path} changed"
        return
    assert hashlib.sha256(data).hexdigest() == digest, f"{path} changed"
