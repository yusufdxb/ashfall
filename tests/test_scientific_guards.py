"""Scientific guards: each protected defect has a test that fails when it is reintroduced.

The nine defects the redesign protects against, and where their guards live:

1. row-0 failure seeding                 here, and protocol.arms.SamplerConfig
2. ignored terrain / environment config  here (config_guard)
3. skipped intervention application      tests/test_gates_h0.py
4. phenotype / intervention conflation   here, tests/test_ontology.py, tests/test_protocol_arms.py
5. held-out checkpoint screening         tests/test_selection_ledger.py
6. unequal compute                       tests/test_protocol_arms.py
7. missing provenance                    here
8. the old BCa sign bug                  tests/test_stats_primary.py
9. detector mutation blindness           tests/test_detector_mutation.py

The last test in this file walks every test module and checks that a
``@pytest.mark.guard`` test exists for each of the nine, so removing one is
itself a failure.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from ashfall.config_guard import IgnoredConfigError, assert_config_applied, declared_but_unapplied
from ashfall.experiment.runner import LEGACY_ROW0_TAG, ExperimentRunner, Row0SeedingError
from ashfall.experiment.schema import Condition, ExperimentConfig
from ashfall.ontology import DeliveryVerdict, GateResult, Intervention
from ashfall.provenance import EvidenceBundle, RunProvenance

TESTS = Path(__file__).parent


def _runner(tmp_path):
    phoenix = tmp_path / "phoenix"
    (phoenix / "configs" / "train").mkdir(parents=True)
    (phoenix / "configs" / "train" / "adaptation.yaml").write_text(
        "run: {name: x}\nresume: {path: base.pt}\nenv: {config: configs/env/flat.yaml}\n"
        "curriculum: {failure_sample_fraction: 0.0, trajectory_dir: data}\n"
    )
    return ExperimentRunner(tmp_path / "ashfall", phoenix_root=phoenix), phoenix


@pytest.mark.guard
def test_row0_seeding_is_refused_unless_legacy_reproduction_is_declared(tmp_path):
    """Reintroduce the Phase-I defect: ask the runner for seed_row_strategy='first'."""
    runner, phoenix = _runner(tmp_path)
    (phoenix / "configs" / "train" / "adaptation.yaml").write_text(
        "run: {name: x}\nresume: {path: base.pt}\nenv: {config: configs/env/flat.yaml}\n"
        "curriculum: {failure_sample_fraction: 0.5, trajectory_dir: data, "
        "seed_row_strategy: first}\n"
    )
    config = ExperimentConfig(name="cell", condition=Condition.ADAPTED)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(Row0SeedingError, match="row 0"):
        runner._write_adapt_override(config, run_dir)
    legacy = ExperimentConfig(name="cell", condition=Condition.ADAPTED, tags=[LEGACY_ROW0_TAG])
    path = runner._write_adapt_override(legacy, run_dir)
    assert "seed_row_strategy: first" in path.read_text()


@pytest.mark.guard
def test_declared_but_unapplied_config_is_refused_for_scientific_runs():
    """Reintroduce the terrain defect: a config with a terrain block nobody reads."""
    config = {
        "env": {"task_name": "Isaac-Velocity-Flat-Unitree-Go2-v0"},
        "terrain": {"type": "flat", "friction_patches": {"enabled": True}},
        "domain_randomization": {"enabled": True, "friction_range": [0.3, 1.5]},
    }
    assert "terrain" in declared_but_unapplied(config)
    with pytest.raises(IgnoredConfigError, match="terrain"):
        assert_config_applied(config, purpose="H0 calibration")
    acknowledged = assert_config_applied(config, purpose="H0 calibration", allow=("terrain",))
    assert acknowledged == ["terrain"]
    clean = {k: v for k, v in config.items() if k != "terrain"}
    assert assert_config_applied(clean, purpose="x") == []


def test_pinned_unwired_declaration_matches_live_phoenix():
    """The CPU copy of Phoenix's unwired-section list must not drift from the sibling."""
    pytest.importorskip("phoenix")
    from phoenix.sim_env import go2_env_cfg as live

    from ashfall import config_guard

    assert tuple(live._UNWIRED_TOP_LEVEL) == config_guard.UNWIRED_TOP_LEVEL
    assert tuple(live._UNWIRED_ROBOT_SUB) == config_guard.UNWIRED_ROBOT_SUB
    assert tuple(live._APPLIED_DR_KEYS) == config_guard.APPLIED_DR_KEYS
    assert tuple(live._APPLIED_PERTURBATION_KEYS) == config_guard.APPLIED_PERTURBATION_KEYS


@pytest.mark.guard
def test_conflated_cause_and_effect_cannot_form_a_verdict():
    gates = (
        GateResult("D1_intervention", True),
        GateResult("D2_departure", True),
        GateResult("D3_phenotype", True),
    )
    with pytest.raises(ValueError):
        DeliveryVerdict("friction_reduction", Intervention.friction_reduction(0.2, 0.1), *gates)
    with pytest.raises(ValueError):
        Intervention("slip", (("static_friction", 0.1),))


@pytest.mark.guard
def test_missing_provenance_is_refused_not_blanked(tmp_path):
    sha = "a" * 64
    with pytest.raises(ValueError, match="Phoenix"):
        RunProvenance(
            ashfall={"sha": "1" * 40},
            phoenix=None,
            config_hashes={"c": sha},
            dataset_hashes={},
            policy_hashes={"p": sha},
            seeds={"training": 1},
            simulator_version=None,
            packages_hash=sha,
        )
    bundle = EvidenceBundle.create(tmp_path, {"x": 1})
    bundle.write_environment({"packages_hash": sha, "simulator_packages": {}})
    bundle.write_metrics({})
    bundle.write_verdict({})
    with pytest.raises(ValueError, match="missing"):
        bundle.finalize()


GUARD_PATTERNS = {
    "row-0 failure seeding": r"row0|row_zero|first_row|seed_row_strategy",
    "ignored environment config": r"unapplied|unwired|ignored_config|terrain",
    "skipped intervention application": r"skipped_intervention|not_applied|unverified_intervention",
    "phenotype/intervention conflation": r"conflat",
    "held-out checkpoint screening": r"held_out|heldout|screening",
    "unequal compute": r"unequal|compute|budget",
    "missing provenance": r"provenance",
    "BCa sign bug": r"bca|acceleration",
    "detector mutation blindness": r"mutant|mutation",
}


def _is_guard_marker(node) -> bool:
    if isinstance(node, ast.Attribute) and node.attr == "guard":
        return True
    if isinstance(node, ast.Call):
        return _is_guard_marker(node.func)
    return False


def _guard_test_names() -> list[str]:
    """Tests marked ``guard`` by decorator or by a module-level ``pytestmark``."""
    names = []
    for path in TESTS.glob("test_*.py"):
        tree = ast.parse(path.read_text())
        module_marked = False
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
            ):
                values = (
                    node.value.elts
                    if isinstance(node.value, (ast.List, ast.Tuple))
                    else [node.value]
                )
                module_marked = any(_is_guard_marker(v) for v in values)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                if module_marked or any(_is_guard_marker(d) for d in node.decorator_list):
                    names.append(f"{path.name}::{node.name}")
    return names


def test_every_protected_defect_has_a_guard_test():
    names = _guard_test_names()
    assert len(names) >= 9, names
    missing = [
        defect
        for defect, pattern in GUARD_PATTERNS.items()
        if not any(re.search(pattern, name, re.IGNORECASE) for name in names)
    ]
    assert not missing, f"no @pytest.mark.guard test covers: {missing}\nguards: {names}"


@pytest.mark.guard
def test_every_historical_config_reproduces_row0_explicitly():
    """A Phase-I config without the legacy tag would silently run a different treatment.

    Reintroduce the defect by removing the tag from any historical config: the
    runner would then default to a pre-onset strategy and the "reproduction"
    would not reproduce the archive.
    """
    import yaml

    from ashfall.experiment.runner import load_experiment_config

    root = Path(__file__).resolve().parents[1]
    configs = sorted((root / "configs" / "experiments").glob("*.yaml")) + sorted(
        (root / "configs" / "ablations").rglob("*.yaml")
    )
    assert configs
    for path in configs:
        raw = yaml.safe_load(path.read_text())
        assert LEGACY_ROW0_TAG in raw.get("tags", []), path
        config = load_experiment_config(path)
        assert LEGACY_ROW0_TAG in config.tags, path
        failure_dir = config.curriculum.failure_dir
        assert failure_dir.startswith("data/fixtures/"), (path, failure_dir)
