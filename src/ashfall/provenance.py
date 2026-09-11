"""Canonical content identities and reproducible experiment manifests."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_artifact(path: str | Path, value: Any) -> Path:
    """Write once; identical reruns are allowed, changed artifacts are refused."""
    path = Path(path)
    payload = canonical_json(value) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != payload:
            raise FileExistsError(f"Refusing to overwrite evidence: {path}")
    else:
        with path.open('x') as stream:
            stream.write(payload)
    return path


def git_identity(repo: str | Path) -> dict:
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    # Include tracked patch and untracked content: a SHA alone cannot identify an edited run.
    status = git('status', '--porcelain')
    patch = git('diff', 'HEAD', '--binary')
    untracked = git('ls-files', '--others', '--exclude-standard').splitlines()
    return {'sha': git('rev-parse', 'HEAD'), 'dirty': bool(status),
            'patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
            'untracked': {p: file_hash(Path(repo) / p) for p in sorted(untracked)
                          if (Path(repo) / p).is_file()}}


@dataclass(frozen=True)
class ExperimentManifest:
    specification: dict
    created_at: str
    schema_version: str = '2.0.0'

    @property
    def experiment_id(self) -> str:
        # Execution timestamp is deliberately outside deterministic specification identity.
        return content_hash(self.specification)

    def to_dict(self) -> dict:
        return {**asdict(self), 'experiment_id': self.experiment_id}

    def save(self, path: str | Path) -> Path:
        return write_artifact(path, self.to_dict())


def build_manifest(*, ashfall_repo, phoenix_repo, config, checkpoint,
                   training_seed, scenario_manifest_hash, capsule_ids,
                   command: list[str], simulator_version: str | None = None,
                   evaluation_seed: int | None = None) -> ExperimentManifest:
    versions = {}
    for name in ('isaaclab', 'rsl-rl-lib', 'numpy'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    spec = dict(ashfall=git_identity(ashfall_repo), phoenix=git_identity(phoenix_repo),
                config=config, config_hash=content_hash(config),
                policy_checkpoint_hash=file_hash(checkpoint), training_seed=training_seed,
                evaluation_seed=evaluation_seed,
                evaluation_scenario_manifest_hash=scenario_manifest_hash,
                failure_capsule_ids=sorted(capsule_ids), simulator_version=simulator_version,
                versions=versions, command=command)
    return ExperimentManifest(spec, datetime.now(timezone.utc).isoformat())
