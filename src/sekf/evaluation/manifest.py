"""Experiment manifest (SPEC.md section 15). Owner: Shailen.

A manifest answers "exactly what produced these results?" so a number in a report can be
reproduced or challenged. It records the git commit, the contract version, the dataset and its
hash, the model, every prompt file and its hash, the policy version and thresholds, the run
configuration, the relevant (non-secret) environment settings, and the start and end times.

Per-case outputs, rules fired, memory before/after, tokens, latency and cost are recorded
case by case in the results file (`sekf.evaluation.jsonl_store`); this file only describes the
run as a whole. The manifest's own hash (`manifest_hash`) is what ties a results file to it.

Nothing here reads a secret: the environment record is an allowlist, never a dump.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sekf.contracts import CONTRACT_VERSION, RunConfig, Thresholds

MANIFEST_VERSION = "1"
# Settings that change what a run means. API keys and URLs with credentials are not on this list.
ENV_ALLOWLIST = (
    "SEKF_PROVIDER",
    "SEKF_MEMORY",
    "SEKF_HISTORIAN",
    "SEKF_SYNTHESIZER",
    "SEKF_POLICY",
    "SEKF_LLM_MODEL",
)


class DatasetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    case_count: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)


class GitRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit: str | None
    # True when files differ from the commit: the SHA alone then does NOT identify the code.
    dirty: bool | None


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: str = MANIFEST_VERSION
    run_id: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime | None = None
    git: GitRecord
    contract_version: str
    dataset: DatasetRecord
    systems: list[str] = Field(min_length=1)
    provider: str
    model: str
    policy_version: str
    thresholds: Thresholds
    run_config: RunConfig
    prompts: dict[str, str]
    environment: dict[str, str]
    python_version: str
    constraints_sha256: str | None = None
    notes: str = ""


# ------------------------------------------------------------------------------- hashing


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_files(root: Path, pattern: str) -> dict[str, str]:
    """`{relative posix path: sha256}` for every file under `root` matching `pattern`, sorted."""
    return {
        p.relative_to(root).as_posix(): sha256_bytes(p.read_bytes())
        for p in sorted(root.rglob(pattern))
        if p.is_file()
    }


def hash_dataset(root: Path) -> tuple[str, int]:
    """One hash for a whole dataset folder, plus its case count.

    The hash covers every `*.json` file's relative path and bytes in sorted order, so it changes
    if a case is added, removed, renamed, or edited, and is the same on any machine.
    """
    files = hash_files(root, "*.json")
    digest = hashlib.sha256()
    for rel, file_hash in files.items():
        digest.update(f"{rel}:{file_hash}\n".encode())
    return digest.hexdigest(), len(files)


def hash_prompts(prompts_root: Path) -> dict[str, str]:
    """`{prompt file: sha256}`. Matches `LoadedPrompt.prompt_hash` (sha256 of the file's bytes)."""
    return hash_files(prompts_root, "*.txt")


def manifest_hash(manifest: RunManifest) -> str:
    """Hash of the part of the manifest that defines the experiment (not its timestamps).

    Start and finish times differ between a run and its resume, so they are excluded; everything
    that changes what the numbers mean is included.
    """
    payload = manifest.model_dump(mode="json", exclude={"started_at", "finished_at", "notes"})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(canonical.encode())


# ---------------------------------------------------------------------------------- git


def git_record(repo: Path) -> GitRecord:
    """Current commit and whether the working tree is dirty. `None`s if this is not a git repo."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(  # noqa: S603 - fixed arguments, no shell
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return GitRecord(commit=None, dirty=None)
    status = run("status", "--porcelain")
    return GitRecord(commit=commit.strip(), dirty=None if status is None else bool(status.strip()))


# ----------------------------------------------------------------------------- building


def build_manifest(
    *,
    run_id: str,
    repo_root: Path,
    dataset_name: str,
    dataset_dir: Path,
    systems: Sequence[str],
    provider: str,
    model: str,
    policy_version: str,
    thresholds: Thresholds | None = None,
    run_config: RunConfig | None = None,
    prompts_dir: Path | None = None,
    constraints_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
    started_at: datetime | None = None,
    notes: str = "",
) -> RunManifest:
    """Collect everything that identifies a run. Reads files and git; makes no network call."""
    env = os.environ if environ is None else environ
    data_hash, count = hash_dataset(dataset_dir)
    constraints = constraints_file or (repo_root / "constraints.txt")
    return RunManifest(
        run_id=run_id,
        started_at=started_at or datetime.now(UTC),
        git=git_record(repo_root),
        contract_version=CONTRACT_VERSION,
        dataset=DatasetRecord(
            name=dataset_name,
            path=_relative(dataset_dir, repo_root),
            case_count=count,
            sha256=data_hash,
        ),
        systems=list(systems),
        provider=provider,
        model=model,
        policy_version=policy_version,
        thresholds=thresholds or Thresholds(),
        run_config=run_config or RunConfig(),
        prompts=hash_prompts(prompts_dir or (repo_root / "prompts")),
        environment={k: env[k] for k in ENV_ALLOWLIST if env.get(k)},
        python_version=platform.python_version(),
        constraints_sha256=sha256_bytes(constraints.read_bytes())
        if constraints.is_file()
        else None,
        notes=notes,
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------- persistence


def write_manifest(path: Path, manifest: RunManifest) -> None:
    """Write the manifest as pretty JSON. Overwrites; call again with `finished_at` set to close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)  # never leave a half-written manifest behind


def load_manifest(path: Path) -> RunManifest:
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def finish(manifest: RunManifest, *, at: datetime | None = None) -> RunManifest:
    return manifest.model_copy(update={"finished_at": at or datetime.now(UTC)})


# -------------------------------------------------------------------------- comparison


def differences(saved: RunManifest, current: RunManifest) -> list[str]:
    """Human-readable reasons `current` is not the same experiment as `saved` (empty if equal).

    Use before resuming or replaying: it names the first thing that changed (for example a
    prompt edited since the capture) instead of letting two runs be mixed silently.
    """
    a = saved.model_dump(mode="json", exclude={"started_at", "finished_at", "notes"})
    b = current.model_dump(mode="json", exclude={"started_at", "finished_at", "notes"})
    out: list[str] = []
    for key in sorted(a):
        if a[key] == b[key]:
            continue
        if isinstance(a[key], dict) and isinstance(b[key], dict):
            for sub in sorted(set(a[key]) | set(b[key])):
                if a[key].get(sub) != b[key].get(sub):
                    out.append(
                        f"{key}.{sub}: {_short(a[key].get(sub))} -> {_short(b[key].get(sub))}"
                    )
        else:
            out.append(f"{key}: {_short(a[key])} -> {_short(b[key])}")
    return out


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 24 else text[:12] + "..." + text[-8:]
