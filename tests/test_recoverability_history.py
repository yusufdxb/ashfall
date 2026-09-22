"""The negative history that motivated the recoverability study stays byte-identical.

``test_frozen_results.py`` protects the result artifacts; this file protects the
documents that state the two NO-GO verdicts and their preregistrations, plus the
frozen toy baseline every later study loads.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FROZEN = {
    "docs/results/FBR_TOY_NEGATIVE_RESULT.md": (
        "0f72d76797005296c0d84d5bfcf2bf28ba855780c0773bddce12388213a6a12f"
    ),
    "docs/results/PRECURSOR_SWEEP_RESULT.md": (
        "6e4c9f07a3e3b04c2fb8817413a23aa5a3b981fa306dc1040f4c897af4a8336e"
    ),
    "docs/research/PRECURSOR_SWEEP_PREREGISTRATION.md": (
        "784c69434b61a1d35a9ee665d1ec249665400878f1948ad9d779054b117ec6f0"
    ),
    "docs/research/TOY_V2_PREREGISTRATION.md": (
        "0cf3e84a1a553de877eea44977f9c43d26dd4ba3a3657200c2a0c07a7b6f85f6"
    ),
    "docs/results/RECOVERABILITY_RETROSPECTIVE_RESULT.md": (
        "6a4b45157b5f633eca0f7562cd7924f78cc0bb061964debaff6f1f0fce99dd93"
    ),
    "docs/research/RECOVERABILITY_RETROSPECTIVE_PLAN.md": (
        "8459a2846174be65cdf5c03cf3fa4ee65a79aba8e5a8430c78be78f554d106b0"
    ),
    "docs/results/ASHFALL_HYPOTHESIS_TRANSITION.md": (
        "6f30cadbd5fc19b351c8f960b5db17238058a6673f92004311f8f89b1e4796cf"
    ),
    "docs/results/FCSI_TOY_RESULT.md": (
        "ca17c8093fb9611589069702990a05aa248a8327c7deb091f07b571bd1242423"
    ),
    "docs/research/FCSI_TOY_PREREGISTRATION.md": (
        "0e7d96f551677f0ef2a3173bf224cf40b4ea8ea7a112d55deb6e6ada6cc19fba"
    ),
    "docs/research/FCSI_HYPOTHESIS_TRANSITION.md": (
        "a5c547306fe743cab0f92d13f1666e53eea99f26c9bbe9861c7bbd6e8c59edcb"
    ),
    "results/fbr_toy_v2/8f49f02/baseline.npy": (
        "2bb2a3bf31bef345bf7b304a5d486c418eda4aeeaedbfb10e88d598b1936e0f3"
    ),
}


@pytest.mark.parametrize("path,digest", sorted(FROZEN.items()))
def test_frozen_document_unchanged(path, digest):
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, f"{path} changed"


def test_transition_note_keeps_both_rejections():
    text = (ROOT / "docs/results/ASHFALL_HYPOTHESIS_TRANSITION.md").read_text()
    assert text.count("REJECTED") >= 2
    assert "NEW hypothesis" in text
    assert "not\npreregistered previously" in text or "not preregistered previously" in text


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    except FileNotFoundError:  # pragma: no cover
        return None


@pytest.mark.parametrize("commit", ["3205f02", "dfec130", "48c8c71", "8f49f02", "08adcff"])
def test_negative_history_commits_reachable(commit):
    out = _git("cat-file", "-t", commit)
    if out is None or out.returncode != 0:
        pytest.skip("git history not available (shallow clone or no git)")
    assert out.stdout.strip() == "commit"


def test_fcsi_transition_keeps_three_rejections():
    text = (ROOT / "docs/research/FCSI_HYPOTHESIS_TRANSITION.md").read_text()
    assert text.count("**REJECTED**") == 3
    assert "NEW hypothesis" in text


@pytest.mark.parametrize("commit", ["61bb95d", "2227de1"])
def test_recoverability_history_commits_reachable(commit):
    out = _git("cat-file", "-t", commit)
    if out is None or out.returncode != 0:
        pytest.skip("git history not available (shallow clone or no git)")
    assert out.stdout.strip() == "commit"


def test_active_diagnosis_transition_keeps_four_no_gos():
    text = (ROOT / "docs/results/ACTIVE_DIAGNOSIS_HYPOTHESIS_TRANSITION.md").read_text()
    assert text.count("**NO-GO**") == 4
    assert "FCSI failed" in text and "new hypothesis" in text


@pytest.mark.parametrize("commit", ["d49daeb", "1b657c1"])
def test_fcsi_history_commits_reachable(commit):
    out = _git("cat-file", "-t", commit)
    if out is None or out.returncode != 0:
        pytest.skip("git history not available (shallow clone or no git)")
    assert out.stdout.strip() == "commit"
