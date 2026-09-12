"""Canonical content identities, environment capture and evidence bundles.

Every scientific run in Ashfall is content-addressed: its identity is the hash
of its specification, not a timestamp or a directory name. Every run emits one
evidence bundle with a fixed layout so a reader can find, without guessing,
what was run, on what, with which code, and what it concluded::

    <bundle>/
      manifest.json      the specification and the run id derived from it
      environment.json   interpreter, packages, simulator, CUDA and GPU
      provenance.json    repository revisions, config/dataset/policy hashes, seeds
      metrics.json       what was measured
      verdict.json       what was concluded, with the hashes it rests on
      logs/              raw process output

A bundle that cannot name its Ashfall revision, Phoenix revision, policy
hash, config hashes and seed is refused at ``finalize``; missing provenance is
an error, never a blank.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dataset_hash(paths: Iterable[str | Path], *, root: str | Path | None = None) -> str:
    """One identity for a set of files: the hash of their sorted (name, sha256) pairs.

    ``root`` makes the names relative, so the same bytes at another location
    produce the same identity. A missing file raises rather than hashing an
    empty set.
    """
    entries = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"dataset member missing: {path}")
        name = str(path.relative_to(root)) if root is not None else path.name
        entries.append((name, file_hash(path)))
    if not entries:
        raise ValueError("a dataset identity needs at least one file")
    return content_hash(sorted(entries))


def write_artifact(path: str | Path, value: Any) -> Path:
    """Write once; identical reruns are allowed, changed artifacts are refused."""
    path = Path(path)
    payload = canonical_json(value) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != payload:
            raise FileExistsError(f"Refusing to overwrite evidence: {path}")
    else:
        with path.open("x") as stream:
            stream.write(payload)
    return path


def read_artifact(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def git_identity(repo: str | Path) -> dict:
    """SHA plus dirty-tree patch and untracked-file hashes.

    A SHA alone cannot identify an edited run.
    """

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    status = git("status", "--porcelain")
    patch = git("diff", "HEAD", "--binary")
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    return {
        "sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "untracked": {
            p: file_hash(Path(repo) / p) for p in sorted(untracked) if (Path(repo) / p).is_file()
        },
    }


# --------------------------------------------------------------------------- #
# Environment capture
# --------------------------------------------------------------------------- #

#: Packages whose exact version changes what a rollout means.
SIMULATOR_PACKAGES = (
    "isaaclab",
    "isaacsim",
    "isaaclab-rl",
    "isaaclab-tasks",
    "rsl-rl-lib",
    "torch",
    "numpy",
    "scipy",
)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            versions[name.lower()] = dist.version
    return dict(sorted(versions.items()))


def _nvidia_smi(query: str) -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.check_output(
            [exe, f"--query-gpu={query}", "--format=csv,noheader"], text=True, timeout=10
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.strip() or None


def environment_snapshot(*, include_all_packages: bool = True) -> dict:
    """What this interpreter and machine are, as far as it can be read without side effects.

    Torch is not imported here: on a machine with Isaac Lab that import is
    heavy and can initialise CUDA. CUDA and GPU metadata come from
    ``nvidia-smi`` when present. Fields that cannot be read are ``None``.
    """
    packages = _package_versions()
    cuda_from_torch = None
    if "torch" in sys.modules:  # already loaded by the caller; reading is free
        cuda_from_torch = getattr(getattr(sys.modules["torch"], "version", None), "cuda", None)
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "simulator_packages": {name: packages.get(name.lower()) for name in SIMULATOR_PACKAGES},
        "cuda": {
            "driver_version": _nvidia_smi("driver_version"),
            "torch_cuda": cuda_from_torch,
        },
        "gpu": {
            "name": _nvidia_smi("name"),
            "memory_total": _nvidia_smi("memory.total"),
        },
        "packages": packages if include_all_packages else {},
        "packages_hash": content_hash(packages),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# Run provenance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunProvenance:
    """Everything a result must be traceable to. ``None`` is allowed only where marked."""

    ashfall: Mapping[str, Any]
    phoenix: Mapping[str, Any] | None
    config_hashes: Mapping[str, str]
    dataset_hashes: Mapping[str, str]
    policy_hashes: Mapping[str, str]
    seeds: Mapping[str, int]
    simulator_version: str | None
    packages_hash: str
    phoenix_absent_reason: str | None = None

    def __post_init__(self):
        if not self.ashfall or not self.ashfall.get("sha"):
            raise ValueError("Ashfall repository identity is required")
        if self.phoenix is None and not self.phoenix_absent_reason:
            raise ValueError("Phoenix identity is required, or a reason it is absent")
        if self.phoenix is not None and not self.phoenix.get("sha"):
            raise ValueError("Phoenix identity must carry a sha")
        for name in ("config_hashes", "dataset_hashes", "policy_hashes"):
            values = getattr(self, name)
            if not isinstance(values, Mapping):
                raise ValueError(f"{name} must be a mapping")
            for key, value in values.items():
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError(f"{name}[{key!r}] must be a SHA256 hex digest")
        if not self.config_hashes:
            raise ValueError("at least one config hash is required")
        if not self.policy_hashes:
            raise ValueError("at least one policy hash is required")
        if not self.seeds or any(type(v) is not int or v < 0 for v in self.seeds.values()):
            raise ValueError("seeds must be named nonnegative integers")
        if len(self.packages_hash) != 64:
            raise ValueError("packages_hash must be a SHA256 hex digest")

    @property
    def provenance_id(self) -> str:
        return content_hash(asdict(self))

    def to_dict(self) -> dict:
        return {**asdict(self), "provenance_id": self.provenance_id}


def collect_provenance(
    *,
    ashfall_repo: str | Path,
    phoenix_repo: str | Path | None,
    config_paths: Mapping[str, str | Path],
    dataset_paths: Mapping[str, Iterable[str | Path]] | None = None,
    policy_paths: Mapping[str, str | Path],
    seeds: Mapping[str, int],
    environment: Mapping[str, Any] | None = None,
) -> RunProvenance:
    environment = environment or environment_snapshot(include_all_packages=False)
    phoenix = None
    reason = None
    if phoenix_repo is None:
        reason = "no simulator sibling declared for this run"
    else:
        try:
            phoenix = git_identity(phoenix_repo)
        except (subprocess.SubprocessError, OSError) as exc:
            reason = f"phoenix repo not readable: {exc}"
    return RunProvenance(
        ashfall=git_identity(ashfall_repo),
        phoenix=phoenix,
        config_hashes={k: file_hash(v) for k, v in config_paths.items()},
        dataset_hashes={k: dataset_hash(v) for k, v in (dataset_paths or {}).items()},
        policy_hashes={k: file_hash(v) for k, v in policy_paths.items()},
        seeds=dict(seeds),
        simulator_version=environment["simulator_packages"].get("isaaclab"),
        packages_hash=environment["packages_hash"],
        phoenix_absent_reason=reason,
    )


# --------------------------------------------------------------------------- #
# Evidence bundles
# --------------------------------------------------------------------------- #

BUNDLE_FILES = (
    "manifest.json",
    "environment.json",
    "provenance.json",
    "metrics.json",
    "verdict.json",
)


@dataclass(frozen=True)
class ExperimentManifest:
    specification: dict
    created_at: str
    schema_version: str = "3.0.0"

    @property
    def experiment_id(self) -> str:
        # Execution timestamp is deliberately outside deterministic specification identity.
        return content_hash(self.specification)

    def to_dict(self) -> dict:
        return {**asdict(self), "experiment_id": self.experiment_id}

    def save(self, path: str | Path) -> Path:
        return write_artifact(path, self.to_dict())


@dataclass
class EvidenceBundle:
    """A content-addressed run directory with a fixed layout and write-once files."""

    root: Path
    manifest: ExperimentManifest
    written: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls, parent: str | Path, specification: Mapping[str, Any], *, name: str | None = None
    ):
        manifest = ExperimentManifest(dict(specification), datetime.now(timezone.utc).isoformat())
        root = Path(parent) / (name or manifest.experiment_id[:16])
        root.mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        bundle = cls(root, manifest)
        bundle._write("manifest.json", manifest.to_dict())
        return bundle

    @property
    def run_id(self) -> str:
        return self.manifest.experiment_id

    def _write(self, name: str, value: Any) -> Path:
        path = write_artifact(self.root / name, value)
        self.written[name] = file_hash(path)
        return path

    def write_environment(self, snapshot: Mapping[str, Any] | None = None) -> Path:
        return self._write("environment.json", dict(snapshot or environment_snapshot()))

    def write_provenance(self, provenance: RunProvenance) -> Path:
        return self._write("provenance.json", provenance.to_dict())

    def write_metrics(self, metrics: Mapping[str, Any]) -> Path:
        return self._write("metrics.json", dict(metrics))

    def write_verdict(self, verdict: Mapping[str, Any]) -> Path:
        payload = dict(verdict)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault(
            "rests_on", {k: v for k, v in self.written.items() if k != "verdict.json"}
        )
        return self._write("verdict.json", payload)

    def log_path(self, name: str) -> Path:
        return self.root / "logs" / name

    def finalize(self) -> dict:
        """Refuse an incomplete bundle. Returns the file-hash index."""
        missing = [name for name in BUNDLE_FILES if not (self.root / name).exists()]
        if missing:
            raise ValueError(f"evidence bundle {self.root} is missing {missing}")
        provenance = read_artifact(self.root / "provenance.json")
        for key in ("ashfall", "config_hashes", "policy_hashes", "seeds", "packages_hash"):
            if not provenance.get(key):
                raise ValueError(f"provenance.json lacks {key}")
        if provenance.get("phoenix") is None and not provenance.get("phoenix_absent_reason"):
            raise ValueError(
                "provenance.json names neither a Phoenix revision nor why it is absent"
            )
        index = {name: file_hash(self.root / name) for name in BUNDLE_FILES}
        write_artifact(self.root / "index.json", {"run_id": self.run_id, "files": index})
        return index


def build_manifest(
    *,
    ashfall_repo,
    phoenix_repo,
    config,
    checkpoint,
    training_seed,
    scenario_manifest_hash,
    capsule_ids,
    command: list[str],
    simulator_version: str | None = None,
    evaluation_seed: int | None = None,
) -> ExperimentManifest:
    versions: dict[str, str | None] = {}
    for name in ("isaaclab", "rsl-rl-lib", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    spec = dict(
        ashfall=git_identity(ashfall_repo),
        phoenix=git_identity(phoenix_repo),
        config=config,
        config_hash=content_hash(config),
        policy_checkpoint_hash=file_hash(checkpoint),
        training_seed=training_seed,
        evaluation_seed=evaluation_seed,
        evaluation_scenario_manifest_hash=scenario_manifest_hash,
        failure_capsule_ids=sorted(capsule_ids),
        simulator_version=simulator_version,
        versions=versions,
        command=command,
    )
    return ExperimentManifest(spec, datetime.now(timezone.utc).isoformat())
