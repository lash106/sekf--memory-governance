"""Sprint 1: versioned prompt loader under `prompts/`."""

from pathlib import Path

import pytest

from sekf.providers import load_prompt, load_role_prompt, render_prompt
from sekf.providers.prompts import prompt_version_for, prompts_dir


def test_prompts_dir_points_at_repo_prompts() -> None:
    root = prompts_dir()
    assert root.name == "prompts"
    assert (root / "judge" / "v1.txt").is_file()


def test_load_role_prompt_hashes_judge_v1() -> None:
    loaded = load_role_prompt("judge", "v1")
    assert loaded.relative_path == "judge/v1.txt"
    assert loaded.version == "judge/v1"
    assert loaded.prompt_hash == load_prompt("judge/v1.txt").prompt_hash
    assert len(loaded.prompt_hash) == 64


def test_prompt_version_for_nested_paths() -> None:
    assert prompt_version_for("judge/v1.txt") == "judge/v1"
    assert prompt_version_for("baselines/single_judge/v1.txt") == "baselines/single_judge/v1"


def test_load_prompt_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("judge/does_not_exist.txt")


def test_render_prompt_rejects_missing_and_extra_keys() -> None:
    with pytest.raises(KeyError, match="missing"):
        render_prompt("hello {candidate_json}", evidence_json="[]")
    with pytest.raises(KeyError, match="unused"):
        render_prompt("hello {candidate_json}", candidate_json="{}", extra="nope")


def test_render_prompt_leaves_json_braces_intact() -> None:
    payload = '{"a":1,"b":{"c":2}}'
    rendered = render_prompt("X {candidate_json} Y", candidate_json=payload)
    assert rendered == f"X {payload} Y"
    assert Path(prompts_dir()).exists()
