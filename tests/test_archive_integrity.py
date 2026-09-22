"""The archive stays traceable: frozen records are pinned and every reference resolves.

``test_frozen_results.py`` pins the result artifacts and ``test_recoverability_history.py`` the
earlier records. This file pins the remaining study records (active diagnosis, the FBR GO2
preregistration and the related-work audits) at their bytes as of the last research commit
``4186bbf``, checks that every result record is hash-indexed, and checks that the archive's
documents only point at files and commits that exist.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest
import test_frozen_results
import test_recoverability_history

ROOT = Path(__file__).resolve().parents[1]

FROZEN = {
    "docs/results/ACTIVE_DIAGNOSIS_TOY_RESULT.md": (
        "32d59969786fef6935ac37f6d84e476ddae7daa334389974cdb4514422144b3f"
    ),
    "docs/research/ACTIVE_DIAGNOSIS_PREREGISTRATION.md": (
        "27cfab8e62074b3c3a1b8bb69ff64256667f2ceecc501635cfbb0bdefff8669c"
    ),
    "docs/results/ACTIVE_DIAGNOSIS_HYPOTHESIS_TRANSITION.md": (
        "85d6bcf26001e7914fc8ce1e162ae0dbbf34647873dfb05f2b4799b7ad40eff3"
    ),
    "docs/research/PREREGISTRATION.md": (
        "b17173e473f2d182243b87c9e2ce24862129a721b7cc63e5e1ea7bbf81badadf"
    ),
    "docs/research/RELATED_WORK.md": (
        "f74c66dd662dc7a05d0243925dbf4f0b415c8d9e7d1f38fedd213dff1780c9a6"
    ),
    "docs/research/RECOVERABILITY_RELATED_WORK.md": (
        "b3a75fbf562112bb52ff89ef6ad5e876404251ca2e45912065bf265e8ec934e4"
    ),
    "docs/research/FCSI_RELATED_WORK.md": (
        "b41eab14a5e594c7053e289c6c49a24734a4c8b3a16821a987901ff99c260a53"
    ),
    "docs/research/ACTIVE_DIAGNOSIS_RELATED_WORK.md": (
        "496947837a2d08e0bf99548a920774b695e63109062b64f68e75218a7dcfc1a0"
    ),
}

# The documents that describe the archive; legacy and frozen records keep their original links.
ARCHIVE_DOCS = ("README.md", "EVIDENCE.md", "docs/ARCHIVE.md", "docs/RESEARCH_HISTORY.md")
MD_LINK = re.compile(r"\]\(([^)\s]+)\)")
PATH_REF = re.compile(r"(?<!\[)`((?:docs|results|src|tests|scripts|configs)/[^`\s*<>{]+)`")
COMMIT_REF = re.compile(r"`([0-9a-f]{7})`")


@pytest.mark.parametrize("path,digest", sorted(FROZEN.items()))
def test_study_record_unchanged(path, digest):
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, f"{path} changed"


def test_every_result_record_is_pinned_and_indexed():
    records = {
        str(p.relative_to(ROOT)) for p in (ROOT / "docs/results").glob("*.md")
    }
    pinned = set(FROZEN) | set(test_recoverability_history.FROZEN)
    assert records <= pinned, f"unpinned result records: {sorted(records - pinned)}"
    results = {r for r in records if r.endswith("_RESULT.md")}
    assert results == set(test_frozen_results.INDEXES)


def _checked_markdown():
    out = [ROOT / "README.md", ROOT / "EVIDENCE.md"]
    for p in sorted((ROOT / "docs").rglob("*.md")):
        if "legacy" not in p.relative_to(ROOT / "docs").parts:
            out.append(p)
    return out


@pytest.mark.parametrize(
    "doc", _checked_markdown(), ids=lambda p: str(p.relative_to(ROOT))
)
def test_references_resolve(doc):
    text = doc.read_text()
    missing = []
    for m in MD_LINK.finditer(text):
        target = m.group(1).split("#")[0]
        if target and not re.match(r"[a-z]+:", target):
            if not (doc.parent / target).resolve().exists():
                missing.append(target)
    for m in PATH_REF.finditer(text):
        target = m.group(1).split("::")[0].rstrip(".,;:")
        if not (ROOT / target).exists():
            missing.append(target)
    assert not missing, f"{doc.relative_to(ROOT)} points at missing files: {missing}"


def _cited_commits():
    found = set()
    for name in ARCHIVE_DOCS:
        found.update(COMMIT_REF.findall((ROOT / name).read_text()))
    return sorted(found)


def test_archive_docs_cite_commits():
    assert len(_cited_commits()) >= 15


def _indexed_private_commits():
    """Full hashes of the private research commits listed in docs/RESEARCH_HISTORY.md."""
    text = (ROOT / "docs/RESEARCH_HISTORY.md").read_text()
    return re.findall(r"`([0-9a-f]{40})`", text)


@pytest.mark.parametrize("commit", _cited_commits())
def test_cited_commit_resolves(commit):
    """A cited commit is either in this repository's history or in the research history index."""
    if any(full.startswith(commit) for full in _indexed_private_commits()):
        return
    try:
        out = subprocess.run(
            ["git", "cat-file", "-t", commit], cwd=ROOT, capture_output=True, text=True
        )
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("git not available")
    if out.returncode != 0:
        shallow = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if shallow.returncode != 0 or shallow.stdout.strip() == "true":
            pytest.skip("git history not available (shallow clone or not a repository)")
    assert out.stdout.strip() == "commit", f"{commit} is cited but resolves nowhere"


def test_readme_reports_every_study_as_no_go():
    text = (ROOT / "README.md").read_text()
    assert "status: archived research program" in text.lower()
    for record in test_frozen_results.INDEXES:
        row = next(
            line for line in text.splitlines()
            if line.startswith("| [") and record.split("/")[-1] in line
        )
        assert "NO-GO" in row, row
