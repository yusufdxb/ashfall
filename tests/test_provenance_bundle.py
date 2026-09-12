"""Content-addressed runs, complete provenance, and the fixture/scientific split."""

from __future__ import annotations

import json
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ashfall.datasets import (
    DatasetManifest,
    NotScientificData,
    assert_scientific,
    fixture_manifest,
    scientific_manifest,
)
from ashfall.provenance import (
    EvidenceBundle,
    RunProvenance,
    collect_provenance,
    content_hash,
    dataset_hash,
    environment_snapshot,
    file_hash,
)


def git_repo(path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "a.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(path), "add", "a.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "x",
        ],
        check=True,
    )
    return path


SHA = "a" * 64


def provenance(**overrides):
    values = dict(
        ashfall={"sha": "1" * 40, "dirty": False},
        phoenix={"sha": "2" * 40, "dirty": False},
        config_hashes={"env": SHA},
        dataset_hashes={},
        policy_hashes={"baseline": SHA},
        seeds={"training": 11},
        simulator_version="4.5.22",
        packages_hash=SHA,
    )
    values.update(overrides)
    return RunProvenance(**values)


class TestProvenance:
    def test_environment_snapshot_names_simulator_packages_and_hashes_packages(self):
        snapshot = environment_snapshot()
        assert set(snapshot["simulator_packages"]) >= {"isaaclab", "torch", "numpy"}
        assert snapshot["packages_hash"] == content_hash(snapshot["packages"])
        assert "gpu" in snapshot and "cuda" in snapshot

    def test_missing_identity_is_refused_not_blanked(self):
        with pytest.raises(ValueError, match="Ashfall"):
            provenance(ashfall={})
        with pytest.raises(ValueError, match="Phoenix"):
            provenance(phoenix=None)
        provenance(phoenix=None, phoenix_absent_reason="cpu-only analysis")
        with pytest.raises(ValueError, match="policy hash"):
            provenance(policy_hashes={})
        with pytest.raises(ValueError, match="seeds"):
            provenance(seeds={"training": -1})
        with pytest.raises(ValueError, match="SHA256"):
            provenance(config_hashes={"env": "short"})

    def test_collect_reads_repos_and_hashes_inputs(self, tmp_path):
        ashfall = git_repo(tmp_path / "ashfall")
        phoenix = git_repo(tmp_path / "phoenix")
        cfg = tmp_path / "env.yaml"
        cfg.write_text("a: 1\n")
        pol = tmp_path / "policy.pt"
        pol.write_bytes(b"weights")
        data = tmp_path / "data"
        data.mkdir()
        (data / "x.parquet").write_bytes(b"x")
        result = collect_provenance(
            ashfall_repo=ashfall,
            phoenix_repo=phoenix,
            config_paths={"env": cfg},
            dataset_paths={"pool": [data / "x.parquet"]},
            policy_paths={"baseline": pol},
            seeds={"training": 3},
        )
        assert result.config_hashes["env"] == file_hash(cfg)
        assert result.dataset_hashes["pool"] == dataset_hash([data / "x.parquet"])
        assert result.phoenix["sha"] and result.ashfall["sha"]
        assert not result.ashfall["dirty"]
        (ashfall / "a.txt").write_text("edited\n")
        dirty = collect_provenance(
            ashfall_repo=ashfall,
            phoenix_repo=None,
            config_paths={"env": cfg},
            policy_paths={"baseline": pol},
            seeds={"training": 3},
        )
        assert dirty.ashfall["dirty"] and dirty.phoenix_absent_reason

    def test_dataset_hash_is_location_independent_and_refuses_missing(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        for d in (a, b):
            d.mkdir()
            (d / "x.parquet").write_bytes(b"same")
        assert dataset_hash([a / "x.parquet"], root=a) == dataset_hash([b / "x.parquet"], root=b)
        with pytest.raises(FileNotFoundError):
            dataset_hash([a / "missing.parquet"])
        with pytest.raises(ValueError):
            dataset_hash([])


class TestEvidenceBundle:
    def test_layout_identity_and_finalize(self, tmp_path):
        bundle = EvidenceBundle.create(tmp_path, {"arm": "A", "seed": 1})
        again = EvidenceBundle.create(tmp_path / "other", {"seed": 1, "arm": "A"})
        assert bundle.run_id == again.run_id  # key order is not identity
        assert (bundle.root / "manifest.json").exists() and (bundle.root / "logs").is_dir()
        with pytest.raises(ValueError, match="missing"):
            bundle.finalize()
        bundle.write_environment({"python": "3.10", "packages_hash": SHA, "simulator_packages": {}})
        bundle.write_provenance(provenance())
        bundle.write_metrics({"incidence": 0.1})
        bundle.write_verdict({"status": "H0_PASS"})
        index = bundle.finalize()
        verdict = json.loads((bundle.root / "verdict.json").read_text())
        assert verdict["run_id"] == bundle.run_id
        assert set(verdict["rests_on"]) == {
            "manifest.json",
            "environment.json",
            "provenance.json",
            "metrics.json",
        }
        assert set(index) == {
            "manifest.json",
            "environment.json",
            "provenance.json",
            "metrics.json",
            "verdict.json",
        }

    def test_write_once(self, tmp_path):
        bundle = EvidenceBundle.create(tmp_path, {"x": 1})
        bundle.write_metrics({"a": 1})
        bundle.write_metrics({"a": 1})
        with pytest.raises(FileExistsError):
            bundle.write_metrics({"a": 2})

    def test_finalize_refuses_untraceable_provenance(self, tmp_path):
        bundle = EvidenceBundle.create(tmp_path, {"x": 2})
        bundle.write_environment({"packages_hash": SHA, "simulator_packages": {}})
        (bundle.root / "provenance.json").write_text(json.dumps({"ashfall": {"sha": "1"}}))
        bundle.write_metrics({})
        bundle.write_verdict({})
        with pytest.raises(ValueError, match="lacks"):
            bundle.finalize()


def parquet(path):
    pq.write_table(pa.table({"a": [1, 2]}), path)
    return path


class TestDatasets:
    def test_fixture_manifest_cannot_pass_as_scientific(self, tmp_path):
        parquet(tmp_path / "synth_slip_000.parquet")
        manifest = fixture_manifest(tmp_path, "fixtures", generator="test")
        assert not manifest.is_scientific
        with pytest.raises(NotScientificData, match="cannot support a scientific claim"):
            assert_scientific(manifest, purpose="detector evaluation")
        with pytest.raises(NotScientificData):
            DatasetManifest("scientific", "synthetic_generator", manifest.files, "x")

    def test_scientific_manifest_requires_provenance(self, tmp_path):
        parquet(tmp_path / "sim_fall_0000.parquet")
        with pytest.raises(NotScientificData, match="lacks provenance"):
            scientific_manifest(tmp_path, source="simulation", description="d", provenance={})
        manifest = scientific_manifest(
            tmp_path,
            source="simulation",
            description="d",
            provenance=dict(
                policy_id=SHA,
                env_config_hash=SHA,
                simulator_version="4.5.22",
                seeds=[1],
                ashfall_sha="1" * 40,
                phoenix_sha="2" * 40,
            ),
        )
        assert assert_scientific(manifest, purpose="x") is manifest
        manifest.save(tmp_path)
        loaded = DatasetManifest.load(tmp_path)
        assert loaded == manifest
        loaded.verify(tmp_path)
        (tmp_path / "sim_fall_0000.parquet").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="does not match"):
            loaded.verify(tmp_path)

    def test_unclassified_directory_is_unusable(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="unclassified"):
            DatasetManifest.load(tmp_path)

    def test_generator_writes_a_fixture_manifest(self, tmp_path):
        from ashfall.synth.generator import generate_all_failures

        paths = generate_all_failures(tmp_path, n_variants=1, n_stable=20)
        assert len(paths) == 6
        manifest = DatasetManifest.load(tmp_path)
        assert manifest.kind == "fixture" and manifest.provenance["n_stable"] == 20
        assert set(manifest.files) == {p.name for p in paths}
