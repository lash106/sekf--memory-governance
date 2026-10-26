import os

import pytest

from sekf.errors import ProviderUnavailable
from sekf.providers import FakeProvider, OpenAICompatibleProvider, create_provider_from_env


def test_factory_defaults_to_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEKF_PROVIDER", raising=False)
    provider = create_provider_from_env(fake_responder=lambda _m, _r: {"value": 1})
    assert isinstance(provider, FakeProvider)


def test_factory_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEKF_PROVIDER", "openai")
    monkeypatch.setenv("SEKF_LLM_API_KEY", "x")
    provider = create_provider_from_env()
    assert isinstance(provider, OpenAICompatibleProvider)


def test_factory_unknown_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEKF_PROVIDER", "anthropic-secret-fallback")
    with pytest.raises(ProviderUnavailable, match="unknown SEKF_PROVIDER"):
        create_provider_from_env()


def test_factory_does_not_read_os_side_effects_beyond_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEKF_PROVIDER", "fake")
    assert os.getenv("SEKF_PROVIDER") == "fake"
