"""Reproducible, resumable experiment runs. Owner: Shailen.

`run_experiment` ties the runner to the run manifest and the resumable results file:

    runs/<name>/manifest.json   what was run: git SHA, dataset hash, prompts, Judge, systems
    runs/<name>/results.jsonl   one row per (case, system), fsynced as each finishes
    runs/<name>/summary.json    metrics, bootstrap intervals, council-vs-system differences

Calling it again with the same run directory resumes: finished pairs are skipped and failed
pairs are retried. If the code, data, prompts, Judge, or system list changed since the run
started, it refuses and names what changed, so two experiments are never mixed.

Held-out data is protected. A dataset path containing a `test` directory is refused unless
`confirm_held_out=True`, and a held-out run also requires a clean git tree, so every
official number can be traced to a commit.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from sekf.evaluation.jsonl_store import JsonlResultStore, run_resumable
from sekf.evaluation.manifest import (
    RunManifest,
    build_manifest,
    differences,
    finish,
    git_record,
    load_manifest,
    manifest_hash,
    write_manifest,
)
from sekf.evaluation.runner import (
    ALL_SYSTEMS,
    COUNCIL,
    DERIVED,
    RunContext,
    SystemNotRunnable,
    build_summary,
    case_id,
    check_systems,
    derive_no_policy,
    load_dev_cases,
    run_case,
)
from sekf.judge.fixture import MODEL_ID as FIXTURE_MODEL_ID
from sekf.judge.fixture import PROMPT_VERSION as FIXTURE_VERSION
from sekf.policy.engine import POLICY_VERSION

# FakeJudge writes these literals into its output; it exports no constants.
FAKE_MODEL_ID = "fake/judge"
FAKE_VERSION = "fake-judge-v0"
REPO_ROOT = Path(__file__).resolve().parents[3]


class HeldOutRefused(Exception):
    """A held-out (`test`) run was requested without the explicit confirmation it needs."""


class RunMismatch(Exception):
    """The run directory belongs to a different experiment than the one requested."""


@dataclass(frozen=True)
class ExperimentResult:
    """Paths written by a run and what happened in this call."""

    run_dir: Path
    manifest: RunManifest
    summary: dict[str, Any]
    counts: dict[str, int]


def is_held_out(dataset_dir: Path) -> bool:
    """True when the dataset lives under a directory named `test`."""

    return "test" in dataset_dir.resolve().parts


def judge_identity(judge: str) -> tuple[str, str]:
    """(provider, model) recorded in the manifest for a Judge choice."""

    if judge == "fake":
        return "fake", f"{FAKE_MODEL_ID}@{FAKE_VERSION}"
    if judge == "fixture":
        return "fixture", f"{FIXTURE_MODEL_ID}@{FIXTURE_VERSION}"
    provider = (os.getenv("SEKF_PROVIDER") or "fake").strip().lower()
    return provider, os.getenv("SEKF_LLM_MODEL") or "unset"


def _manifest(
    *,
    run_id: str,
    dataset_dir: Path,
    systems: Sequence[str],
    judge: str,
    repo_root: Path,
    notes: str,
) -> RunManifest:
    provider, model = judge_identity(judge)
    return build_manifest(
        run_id=run_id,
        repo_root=repo_root,
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir,
        systems=list(systems),
        provider=provider,
        model=model,
        policy_version=POLICY_VERSION,
        notes=notes or f"judge={judge}",
    )


def _open_manifest(
    run_dir: Path,
    *,
    dataset_dir: Path,
    systems: Sequence[str],
    judge: str,
    repo_root: Path,
    notes: str,
) -> RunManifest:
    path = run_dir / "manifest.json"
    if path.exists():
        saved = load_manifest(path)
        current = _manifest(
            run_id=saved.run_id,
            dataset_dir=dataset_dir,
            systems=systems,
            judge=judge,
            repo_root=repo_root,
            notes=notes,
        )
        changed = differences(saved, current)
        if changed:
            raise RunMismatch(
                f"{run_dir} is a different experiment; changed: {'; '.join(changed)}. "
                "Use a new run directory, or restore the original code, data and settings."
            )
        return saved
    manifest = _manifest(
        run_id=str(uuid4()),
        dataset_dir=dataset_dir,
        systems=systems,
        judge=judge,
        repo_root=repo_root,
        notes=notes,
    )
    write_manifest(path, manifest)
    return manifest


async def run_experiment(
    *,
    dataset_dir: Path,
    run_dir: Path,
    judge: str = "fixture",
    systems: Sequence[str] = ALL_SYSTEMS,
    limit: int = 0,
    concurrency: int = 1,
    confirm_held_out: bool = False,
    repo_root: Path = REPO_ROOT,
    notes: str = "",
) -> ExperimentResult:
    """Run (or resume) one experiment and write its manifest, results, and summary."""

    check_systems(systems)
    if limit < 0:
        raise ValueError("limit must be >= 0")
    if is_held_out(dataset_dir) and not confirm_held_out:
        raise HeldOutRefused(
            f"{dataset_dir} is held-out data. Freeze prompts and thresholds first, then pass "
            "confirm_held_out (--confirm-held-out). Never tune on these results."
        )
    cases = load_dev_cases(dataset_dir)
    if limit:
        cases = cases[:limit]
    if not cases:
        raise ValueError(f"no labeled cases under {dataset_dir}")

    if is_held_out(dataset_dir) and git_record(repo_root).dirty:
        raise HeldOutRefused("a held-out run needs a clean git tree; commit or stash first")
    manifest = _open_manifest(
        run_dir,
        dataset_dir=dataset_dir,
        systems=systems,
        judge=judge,
        repo_root=repo_root,
        notes=notes,
    )

    store = JsonlResultStore(
        run_dir / "results.jsonl", run_id=manifest.run_id, manifest_hash=manifest_hash(manifest)
    )
    by_id = {case_id(data): data for data in cases}
    ctx = RunContext(judge)
    skipped: dict[str, str] = {}
    run_now = [name for name in systems if name not in DERIVED]
    for name in run_now:
        if name == "single_judge" and (reason := ctx.single_judge_reason()) is not None:
            skipped[name] = reason
    keys = [(cid, name) for cid in by_id for name in run_now if name not in skipped]

    async def work(key: tuple[str, str]) -> dict[str, Any]:
        cid, name = key
        try:
            return await run_case(name, by_id[cid], ctx)
        except SystemNotRunnable as exc:  # recorded as an error row, retried on resume
            raise RuntimeError(str(exc)) from exc

    try:
        counts = await run_resumable(store, keys, work, concurrency=concurrency)
    finally:
        await ctx.aclose()

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in systems}
    for record in store.results():
        if record["case_id"] in by_id and record["system"] in rows:
            rows[record["system"]].append(record["result"])
    for name, source in DERIVED.items():
        if name in rows and source == COUNCIL:
            rows[name] = [derive_no_policy(row) for row in rows[source]]

    summary = build_summary(rows, skipped, n_cases=len(cases), judge=judge)
    summary["run_id"] = manifest.run_id
    summary["dataset"] = manifest.dataset.path
    summary["held_out"] = is_held_out(dataset_dir)
    summary["incomplete_pairs"] = len(store.pending(keys))
    summary["errors"] = len(store.errors())
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if summary["incomplete_pairs"] == 0:
        manifest = finish(manifest)
        write_manifest(run_dir / "manifest.json", manifest)
    return ExperimentResult(run_dir=run_dir, manifest=manifest, summary=summary, counts=counts)
