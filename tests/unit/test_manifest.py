"""Run manifest: it identifies a run exactly, hashes are stable, and no secret can leak in."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sekf.contracts import CONTRACT_VERSION, RunConfig, Thresholds
from sekf.evaluation import manifest as mf

REPO = Path(__file__).resolve().parents[2]


def make_repo(tmp_path: Path) -> Path:
    """A tiny git repo with a dataset, prompts, and constraints, like the real layout."""
    (tmp_path / "datasets" / "dev").mkdir(parents=True)
    (tmp_path / "datasets" / "dev" / "a.json").write_text('{"case_id": "a"}\n')
    (tmp_path / "datasets" / "dev" / "b.json").write_text('{"case_id": "b"}\n')
    (tmp_path / "prompts" / "judge").mkdir(parents=True)
    (tmp_path / "prompts" / "judge" / "v1.txt").write_text("judge prompt")
    (tmp_path / "constraints.txt").write_text("pydantic==2.0\n")
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
    }
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, env=env)
    return tmp_path


def build(repo: Path, **overrides: object) -> mf.RunManifest:
    kwargs: dict[str, object] = {
        "run_id": "run-1",
        "repo_root": repo,
        "dataset_name": "dev",
        "dataset_dir": repo / "datasets" / "dev",
        "systems": ["policy-v1", "direct_storage"],
        "provider": "fake",
        "model": "fake/none",
        "policy_version": "policy-v1",
        "environ": {},
        "started_at": datetime(2026, 11, 22, tzinfo=UTC),
    }
    return mf.build_manifest(**(kwargs | overrides))  # type: ignore[arg-type]


def test_manifest_records_what_identifies_the_run(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    m = build(repo, thresholds=Thresholds(support_threshold=0.9), run_config=RunConfig(seed=7))
    assert len(m.git.commit or "") == 40 and m.git.dirty is False
    assert m.contract_version == CONTRACT_VERSION
    assert (
        m.dataset.name == "dev" and m.dataset.case_count == 2 and m.dataset.path == "datasets/dev"
    )
    assert m.prompts == {"judge/v1.txt": mf.sha256_bytes(b"judge prompt")}
    assert m.thresholds.support_threshold == 0.9 and m.run_config.seed == 7
    assert m.systems == ["policy-v1", "direct_storage"] and m.policy_version == "policy-v1"
    assert m.constraints_sha256 == mf.sha256_bytes(b"pydantic==2.0\n")


def test_a_dirty_tree_is_flagged_because_the_sha_alone_would_mislead(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "prompts" / "judge" / "v1.txt").write_text("edited after the commit")
    assert build(repo).git.dirty is True


def test_not_a_git_repo_is_recorded_as_unknown_not_a_crash(tmp_path: Path) -> None:
    (tmp_path / "datasets" / "dev").mkdir(parents=True)
    (tmp_path / "datasets" / "dev" / "a.json").write_text("{}")
    m = build(tmp_path)
    assert m.git.commit is None and m.git.dirty is None


def test_dataset_hash_changes_when_any_case_changes_and_is_stable_otherwise(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    data = repo / "datasets" / "dev"
    first = mf.hash_dataset(data)
    assert mf.hash_dataset(data) == first  # stable
    (data / "b.json").write_text('{"case_id": "b", "edited": true}\n')
    assert mf.hash_dataset(data)[0] != first[0]  # edit
    (data / "c.json").write_text("{}")
    assert mf.hash_dataset(data)[1] == 3  # addition is counted
    (data / "c.json").rename(data / "d.json")
    renamed = mf.hash_dataset(data)[0]
    (data / "d.json").rename(data / "c.json")
    assert renamed != mf.hash_dataset(data)[0]  # renames count too


def test_prompt_hash_matches_what_roles_record_in_call_metadata() -> None:
    from sekf.providers import load_role_prompt

    hashes = mf.hash_prompts(REPO / "prompts")
    assert hashes["judge/v1.txt"] == load_role_prompt("judge", "v1").prompt_hash


def test_environment_is_an_allowlist_and_never_contains_a_key(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    secret = "sk-this-must-never-appear-0123456789"
    m = build(
        repo,
        environ={
            "SEKF_PROVIDER": "openai",
            "SEKF_LLM_MODEL": "gpt-4o-mini",
            "SEKF_LLM_API_KEY": secret,
            "SEKF_LLM_BASE_URL": f"https://user:{secret}@example.com",
            "SOMETHING_ELSE": "x",
        },
    )
    assert m.environment == {"SEKF_PROVIDER": "openai", "SEKF_LLM_MODEL": "gpt-4o-mini"}
    assert secret not in m.model_dump_json()


def test_manifest_hash_ignores_times_and_notes_but_not_the_experiment(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    a = build(repo)
    later = a.model_copy(
        update={
            "started_at": datetime(2030, 1, 1, tzinfo=UTC),
            "notes": "second attempt",
            "finished_at": datetime(2030, 1, 2, tzinfo=UTC),
        }
    )
    assert mf.manifest_hash(a) == mf.manifest_hash(later)
    other_model = build(repo, model="gpt-4o-mini")
    other_thresholds = build(repo, thresholds=Thresholds(refute_threshold=0.5))
    assert len({mf.manifest_hash(x) for x in (a, other_model, other_thresholds)}) == 3


def test_round_trip_and_finish(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    m = build(repo)
    path = tmp_path / "out" / "manifest.json"
    mf.write_manifest(path, m)
    assert mf.load_manifest(path) == m
    closed = mf.finish(m, at=datetime(2026, 11, 23, tzinfo=UTC))
    mf.write_manifest(path, closed)
    assert mf.load_manifest(path).finished_at == datetime(2026, 11, 23, tzinfo=UTC)
    assert not list(path.parent.glob("*.tmp"))  # no half-written file left behind
    json.loads(path.read_text())  # valid JSON


def test_differences_name_what_changed_since_the_original_run(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    saved = build(repo)
    assert mf.differences(saved, build(repo)) == []
    (repo / "prompts" / "judge" / "v1.txt").write_text("a changed prompt")
    (repo / "datasets" / "dev" / "a.json").write_text('{"case_id": "a", "x": 1}\n')
    changed = build(repo, model="gpt-4o-mini")
    reasons = "\n".join(mf.differences(saved, changed))
    assert "prompts.judge/v1.txt" in reasons
    assert "dataset.sha256" in reasons
    assert "model:" in reasons


def test_unknown_fields_are_rejected(tmp_path: Path) -> None:
    from pydantic import ValidationError

    data = build(make_repo(tmp_path)).model_dump(mode="json") | {"surprise": 1}
    with pytest.raises(ValidationError):
        mf.RunManifest.model_validate(data)
