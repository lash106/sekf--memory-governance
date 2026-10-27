"""Choose the Judge implementation from the environment. Owner: Sai Chetan.

`SEKF_PROVIDER=fake` (default) keeps the deterministic FakeJudge used by tests.
`SEKF_PROVIDER=openai` builds `LLMJudge` on `OpenAICompatibleProvider`. Unknown values
fail; there is no silent fallback.
"""

from __future__ import annotations

import os
from pathlib import Path

from sekf.errors import ProviderUnavailable
from sekf.judge.fake import FakeJudge
from sekf.judge.interface import Judge
from sekf.judge.llm.judge import LLMJudge
from sekf.providers.factory import create_provider_from_env

_ENV_LOADED = False


def load_project_env(path: Path | None = None) -> None:
    """Load repo `.env` into the process without overriding variables already set.

    Uvicorn does not load `.env` by itself. Existing variables win so tests and CI stay
    on the values they set explicitly.
    """

    global _ENV_LOADED
    if _ENV_LOADED and path is None:
        return
    env_path = path if path is not None else Path(__file__).resolve().parents[3] / ".env"
    _ENV_LOADED = True
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def build_judge_from_env() -> Judge:
    """Return FakeJudge or LLMJudge according to `SEKF_PROVIDER`."""

    load_project_env()
    name = (os.getenv("SEKF_PROVIDER") or "fake").strip().lower()
    if name in {"", "fake"}:
        return FakeJudge()
    if name in {"openai", "openai_compatible"}:
        return LLMJudge(create_provider_from_env())
    raise ProviderUnavailable(
        f"unknown SEKF_PROVIDER={name!r}; expected 'fake' or 'openai' (no silent fallback)"
    )
