"""The application picks its parts from the environment, and never silently falls back."""

import pytest

from sekf.api import Governance
from sekf.api import app as app_module
from sekf.errors import ConfigurationError
from sekf.historian import DeterministicHistorian, FakeHistorian, LLMHistorian
from sekf.judge import FakeJudge
from sekf.memory import InMemoryRepository, SqlMemoryRepository
from sekf.policy import PolicyEngine
from sekf.synthesizer import DeterministicSynthesizer, FakeSynthesizer, LLMSynthesizer


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore a developer's local .env and start from a blank environment."""
    from sekf.judge import compose

    monkeypatch.setattr(compose, "_ENV_LOADED", True)
    for name in ("SEKF_MEMORY", "SEKF_HISTORIAN", "SEKF_SYNTHESIZER", "SEKF_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SEKF_DB_URL", "sqlite://")  # the default store, in RAM for tests


def test_defaults_need_no_network_and_no_key() -> None:
    gov = app_module.build_default_governance()
    assert isinstance(gov._memory, SqlMemoryRepository)  # the real store, not the test double
    assert isinstance(gov._historian, DeterministicHistorian)
    assert isinstance(gov._judge, FakeJudge)
    assert isinstance(gov._synthesizer, DeterministicSynthesizer)
    assert isinstance(gov._policy, PolicyEngine)


def test_test_double_memory_and_fake_historian_can_be_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEKF_MEMORY", "MEMORY")  # case-insensitive
    monkeypatch.setenv("SEKF_HISTORIAN", "fake")
    monkeypatch.setenv("SEKF_SYNTHESIZER", "fake")
    gov = app_module.build_default_governance()
    assert isinstance(gov._memory, InMemoryRepository)
    assert isinstance(gov._historian, FakeHistorian)
    assert isinstance(gov._synthesizer, FakeSynthesizer)


@pytest.mark.parametrize("name", ["SEKF_MEMORY", "SEKF_HISTORIAN", "SEKF_SYNTHESIZER"])
def test_unknown_values_fail_loudly(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name, "postgres-or-something")
    with pytest.raises(ConfigurationError, match=name):
        app_module.build_default_governance()


def test_app_serves_and_runs_its_lifespan_with_default_wiring() -> None:
    from fastapi.testclient import TestClient

    with TestClient(app_module.create_app()) as client:  # enters and exits the lifespan
        assert client.get("/health").json()["status"] == "ok"


class Owned:
    """A component that owns a resource."""

    def __init__(self, fail: bool = False) -> None:
        self.closed = False
        self._fail = fail

    def close(self) -> None:
        self.closed = True
        if self._fail:
            raise RuntimeError("boom")


class AsyncOwned:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def _governance(**parts: object) -> Governance:
    return Governance(
        memory=parts.get("memory", InMemoryRepository()),  # type: ignore[arg-type]
        historian=parts.get("historian", FakeHistorian()),  # type: ignore[arg-type]
        judge=parts.get("judge", FakeJudge()),  # type: ignore[arg-type]
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
    )


def test_shutdown_closes_every_owned_resource() -> None:
    from fastapi.testclient import TestClient

    memory, judge = Owned(), AsyncOwned()
    gov = _governance(memory=memory, judge=judge)
    with TestClient(app_module.create_app(gov)):
        assert not memory.closed and not judge.closed
    assert memory.closed and judge.closed


def test_a_failing_close_does_not_stop_the_others_and_is_reported() -> None:
    import asyncio

    broken, memory, judge = Owned(fail=True), Owned(), AsyncOwned()
    gov = _governance(historian=broken, memory=memory, judge=judge)
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(gov.aclose())
    assert broken.closed and memory.closed and judge.closed


@pytest.mark.parametrize("name", ["SEKF_HISTORIAN", "SEKF_SYNTHESIZER"])
def test_llm_roles_refuse_to_run_without_a_live_provider(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(name, "llm")  # SEKF_PROVIDER is unset, i.e. "fake"
    with pytest.raises(ConfigurationError, match="SEKF_PROVIDER=openai"):
        app_module.build_default_governance()


def test_llm_roles_are_built_when_a_live_provider_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEKF_PROVIDER", "openai")
    monkeypatch.setenv("SEKF_LLM_API_KEY", "test-key-not-used")  # no request is made
    monkeypatch.setenv("SEKF_HISTORIAN", "llm")
    monkeypatch.setenv("SEKF_SYNTHESIZER", "llm")
    gov = app_module.build_default_governance()
    assert isinstance(gov._historian, LLMHistorian)
    assert isinstance(gov._synthesizer, LLMSynthesizer)


@pytest.mark.parametrize("name", ["SEKF_HISTORIAN", "SEKF_SYNTHESIZER"])
def test_llm_roles_refuse_to_start_without_an_api_key(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this, the app would start and then fail every single request."""
    monkeypatch.setenv("SEKF_PROVIDER", "openai")
    monkeypatch.delenv("SEKF_LLM_API_KEY", raising=False)
    monkeypatch.setenv(name, "llm")
    with pytest.raises(ConfigurationError, match="SEKF_LLM_API_KEY"):
        app_module.build_default_governance()
