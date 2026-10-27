"""Judge selection from SEKF_PROVIDER. Offline; no network."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sekf.api.app import build_default_governance
from sekf.errors import ProviderUnavailable
from sekf.judge.compose import build_judge_from_env, load_project_env
from sekf.judge.fake import FakeJudge
from sekf.judge.llm import LLMJudge


def test_default_judge_is_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEKF_PROVIDER", "fake")
    judge = build_judge_from_env()
    assert isinstance(judge, FakeJudge)
    assert isinstance(build_default_governance()._judge, FakeJudge)


def test_openai_provider_selects_llm_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEKF_PROVIDER", "openai")
    monkeypatch.setenv("SEKF_LLM_API_KEY", "test-key")
    judge = build_judge_from_env()
    assert isinstance(judge, LLMJudge)
    wired = build_default_governance()._judge
    assert isinstance(wired, LLMJudge)


def test_unknown_provider_is_not_a_silent_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEKF_PROVIDER", "mystery")
    with pytest.raises(ProviderUnavailable, match="mystery"):
        build_judge_from_env()


def test_dotenv_does_not_override_existing_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SEKF_PROVIDER=openai\nSEKF_LLM_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("SEKF_PROVIDER", "fake")
    monkeypatch.delenv("SEKF_LLM_MODEL", raising=False)
    try:
        load_project_env(env_file)
        assert os.environ["SEKF_PROVIDER"] == "fake"
        assert os.environ["SEKF_LLM_MODEL"] == "from-file"
    finally:
        os.environ.pop("SEKF_LLM_MODEL", None)
