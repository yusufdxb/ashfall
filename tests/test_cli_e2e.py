"""End-to-end CLI runs on the CPU: demo, toy H0 bundle, detector evaluation, env snapshot."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ashfall.cli import main
from ashfall.datasets import DatasetManifest


def test_env_snapshot(tmp_path):
    out = tmp_path / "env.json"
    assert main(["env-snapshot", "--output", str(out)]) == 0
    snapshot = json.loads(out.read_text())
    assert "packages_hash" in snapshot and "simulator_packages" in snapshot


def test_demo_is_mock_only(tmp_path, capsys):
    assert main(["demo", "--output", str(tmp_path / "demo")]) == 0
    result = json.loads((tmp_path / "demo" / "demo.json").read_text())
    assert result["evidence_kind"] == "mock" and result["is_research_result"] is False


def test_h0_toy_writes_a_complete_bundle(tmp_path, capsys):
    spec = {
        "intervention": {
            "kind": "friction_reduction",
            "parameters": [["dynamic_friction", 0.08], ["static_friction", 0.1]],
        },
        "intended_phenotype": "slip",
        "command": [0.6, 0.0, 0.0],
        "n_pairs": 6,
        "replicates_per_pair": 3,
        "horizon_steps": 120,
        "seed_base": 500,
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    code = main(
        [
            "h0",
            "--backend",
            "toy",
            "--spec",
            str(spec_path),
            "--output",
            str(tmp_path / "bundles"),
            "--sensitivity-pairs",
            "1",
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    bundle = tmp_path / "bundles" / summary["run_id"][:16]
    for name in (
        "manifest.json",
        "environment.json",
        "provenance.json",
        "metrics.json",
        "verdict.json",
        "index.json",
        "seed_plan.json",
    ):
        assert (bundle / name).exists(), name
    verdict = json.loads((bundle / "verdict.json").read_text())
    assert verdict["evidence_kind"] == "mock" and verdict["is_research_result"] is False
    assert verdict["verdict"] in ("PASS", "FAIL", "UNINFORMATIVE")
    metrics = json.loads((bundle / "metrics.json").read_text())
    assert len(metrics["per_pair"]) == 6 and len(metrics["d2_sensitivity"]) == 1
    assert metrics["d2_sensitivity"][0]["stable"] in (True, False)
    manifest = DatasetManifest.load(bundle / "harvest")
    assert manifest.kind == "fixture"
    manifest.verify(bundle / "harvest")
    provenance = json.loads((bundle / "provenance.json").read_text())
    assert provenance["ashfall"]["sha"] and provenance["phoenix_absent_reason"]


def test_h0_phoenix_requires_backend_config(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "intervention": {"kind": "none"},
                "intended_phenotype": "slip",
                "n_pairs": 1,
                "replicates_per_pair": 2,
                "horizon_steps": 10,
            }
        )
    )
    with pytest.raises(SystemExit):
        main(["h0", "--backend", "phoenix", "--spec", str(spec_path), "--output", str(tmp_path)])


def test_detector_eval_on_fixture_is_regression_not_validation(tmp_path, capsys):
    assert main(["detector-eval", "--output", str(tmp_path / "det")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["report_kind"] == "fixture_regression"
    assert summary["validation_claim_allowed"] is False
    assert set(summary["mutants_caught"]) >= {
        "always_collapse",
        "always_stumble",
        "always_command_mismatch",
        "no_slip_priority",
    }
    report = json.loads((tmp_path / "det" / "detector_report.json").read_text())
    assert report["report_kind"] == "fixture_regression"
    assert (tmp_path / "det" / "detector_report.md").exists()


def test_console_entry_point_parses():
    proc = subprocess.run(
        [sys.executable, "-m", "ashfall.cli", "h0", "--help"], capture_output=True, text=True
    )
    assert proc.returncode == 0 and "--allow-config-block" in proc.stdout
