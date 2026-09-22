"""The public release is sanitized in its history as well as its tree.

The research branches are kept in a private archive because their history contains internal
planning documents. This release starts from the previously public history and adds the final
archive content as new commits. These tests keep it that way: the excluded documents appear in no
reachable commit, every published result file is accounted for in the public artifact manifest,
and no tracked text carries an absolute home-directory path beyond the three legacy files that
were public before the release.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/PUBLIC_ARTIFACT_MANIFEST.json"

# Excluded from the public release; matched against every path in every reachable commit.
EXCLUDED_NAMES = ("PAPER_OUTLINE", "HARDWARE_DEMO", "CORL_STORY_ANALYSIS")

# The only artifact published as a sanitized copy. Adding to this set is a release decision.
SANITIZED = {"results/recoverability/retro_2227de1/analysis.json"}
PLACEHOLDER = b"<workspace>/"

# Public, byte-for-byte, before this release; see docs/PUBLIC_ARTIFACT_MANIFEST.md.
ALREADY_PUBLIC_WITH_PATHS = {
    "results/legacy_row0_curriculum/REPORT.md",
    "results/legacy_row0_curriculum/multiseed_n11_scale_verdict.md",
    "results/legacy_row0_curriculum/multiseed_scale_ext_2026-06-02_ANALYSIS.md",
}
HOME_PATH = re.compile(rb"/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+")


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError:  # pragma: no cover
        return None


def _require_full_history():
    shallow = _git("rev-parse", "--is-shallow-repository")
    if shallow is None or shallow.returncode != 0 or shallow.stdout.strip() == "true":
        pytest.skip("full git history not available (shallow clone or not a repository)")


def _tracked():
    out = _git("ls-files")
    if out is None or out.returncode != 0:
        pytest.skip("git not available")
    return [p for p in out.stdout.splitlines() if (ROOT / p).is_file()]


def _manifest():
    return json.loads(MANIFEST.read_text())["artifacts"]


def test_excluded_documents_absent_from_all_reachable_history():
    _require_full_history()
    out = _git("log", "HEAD", "--name-only", "--format=", "--no-renames")
    assert out.returncode == 0
    hits = sorted({p for p in out.stdout.splitlines() if any(n in p for n in EXCLUDED_NAMES)})
    assert not hits, f"excluded documents in reachable history: {hits}"


def test_excluded_documents_absent_from_tree():
    hits = [p for p in _tracked() if any(n in p for n in EXCLUDED_NAMES)]
    assert not hits


def test_manifest_covers_every_result_file():
    listed = {r["path"] for r in _manifest()}
    on_disk = {
        p for p in _tracked()
        if p.startswith("results/") and p != "results/PUBLIC_ARTIFACT_MANIFEST.json"
    }
    assert listed == on_disk


@pytest.mark.parametrize("row", _manifest(), ids=lambda r: r["path"])
def test_manifest_hash_matches_file(row):
    data = (ROOT / row["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row["public_sha256"], row["path"]
    if row["status"] == "IDENTICAL":
        assert row["public_sha256"] == row["original_sha256"]
    else:
        assert row["status"] == "SANITIZED_PATHS_ONLY"


def test_sanitized_set_is_exactly_the_declared_one():
    assert {r["path"] for r in _manifest() if r["status"] != "IDENTICAL"} == SANITIZED


@pytest.mark.parametrize("path", sorted(SANITIZED))
def test_sanitized_copy_is_a_pure_prefix_replacement(path):
    row = next(r for r in _manifest() if r["path"] == path)
    data = (ROOT / path).read_bytes()
    assert not HOME_PATH.search(data)
    assert data.count(PLACEHOLDER) == row["occurrences"]
    # Restoring the prefix must give back exactly the original length.
    delta = row["original_prefix_length"] - len(PLACEHOLDER)
    assert len(data) + row["occurrences"] * delta == row["original_size"]


def test_no_home_paths_in_tracked_text():
    offenders = []
    for p in _tracked():
        if p in ALREADY_PUBLIC_WITH_PATHS or p.endswith((".npy", ".png", ".parquet")):
            continue
        if HOME_PATH.search((ROOT / p).read_bytes()):
            offenders.append(p)
    assert not offenders, f"absolute home paths in: {offenders}"


def test_already_public_path_files_are_unchanged_legacy():
    rows = {r["path"]: r for r in _manifest()}
    for p in ALREADY_PUBLIC_WITH_PATHS:
        assert rows[p]["status"] == "IDENTICAL" and rows[p]["public_before_release"], p
