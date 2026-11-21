"""A results file is bound to its manifest: edit a prompt and the run will not resume."""

import asyncio
from pathlib import Path

import pytest
from tests.unit.test_manifest import build, make_repo

from sekf.evaluation import manifest as mf
from sekf.evaluation.jsonl_store import JsonlResultStore, ManifestMismatch, run_resumable

KEYS = [("a", "policy-v1"), ("b", "policy-v1")]


async def work(_key: tuple[str, str]) -> dict[str, str]:
    return {"ok": "yes"}


def store_for(path: Path, manifest: mf.RunManifest) -> JsonlResultStore:
    return JsonlResultStore(path, run_id=manifest.run_id, manifest_hash=mf.manifest_hash(manifest))


def test_resume_works_when_nothing_relevant_changed(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    results = tmp_path / "results.jsonl"
    asyncio.run(run_resumable(store_for(results, build(repo)), KEYS[:1], work))
    counts = asyncio.run(run_resumable(store_for(results, build(repo)), KEYS, work))
    assert counts["skipped"] == 1 and counts["ran"] == 1


@pytest.mark.parametrize("what", ["prompt", "dataset", "model"])
def test_resume_is_refused_after_the_experiment_changed(tmp_path: Path, what: str) -> None:
    repo = make_repo(tmp_path / "repo")
    results = tmp_path / "results.jsonl"
    asyncio.run(run_resumable(store_for(results, build(repo)), KEYS[:1], work))
    overrides: dict[str, object] = {}
    if what == "prompt":
        (repo / "prompts" / "judge" / "v1.txt").write_text("silently edited prompt")
    elif what == "dataset":
        (repo / "datasets" / "dev" / "a.json").write_text('{"case_id": "a", "x": 1}\n')
    else:
        overrides["model"] = "gpt-4o-mini"
    with pytest.raises(ManifestMismatch):
        store_for(results, build(repo, **overrides))
