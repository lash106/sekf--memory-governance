"""The live smoke script: safe by default, capped, and runnable offline with fakes."""

import pytest
from pydantic import BaseModel
from tests.conftest import NOW

import smoke_llm
from sekf.api import Governance
from sekf.errors import CallUsage, ProviderUnavailable, RoleOutputInvalid
from sekf.historian import DeterministicHistorian
from sekf.judge import FakeJudge
from sekf.memory import SqlMemoryRepository
from sekf.policy import PolicyEngine
from sekf.providers import LLMResult
from sekf.synthesizer import DeterministicSynthesizer


@pytest.fixture(autouse=True)
def no_live_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from sekf.judge import compose

    monkeypatch.setattr(compose, "_ENV_LOADED", True)
    for name in ("SEKF_PROVIDER", "SEKF_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_refuses_to_run_without_a_live_provider_and_sends_nothing() -> None:
    with pytest.raises(SystemExit, match="SEKF_PROVIDER=openai"):
        smoke_llm.main([])


def test_refuses_to_run_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEKF_PROVIDER", "openai")
    with pytest.raises(SystemExit, match="SEKF_LLM_API_KEY"):
        smoke_llm.main([])


def _offline_governance() -> Governance:
    return Governance(
        memory=SqlMemoryRepository.from_url("sqlite://", clock=lambda: NOW),
        historian=DeterministicHistorian(),
        judge=FakeJudge(),
        synthesizer=DeterministicSynthesizer(),
        policy=PolicyEngine(),
        clock=lambda: NOW,
    )


def test_runs_all_five_cases_offline_and_prints_a_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert smoke_llm.main([], governance=_offline_governance()) == 0
    out = capsys.readouterr().out
    for stem in ("shailen-supported-new-01", "shailen-injection-01", "shailen-source-similar-01"):
        assert stem in out
    assert "total: 0 calls, $0.0000; role failures: 0" in out


def test_the_call_limit_stops_the_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert smoke_llm.main(["--max-calls", "0"], governance=_offline_governance()) == 0
    out = capsys.readouterr().out
    assert "limit reached" in out and "shailen-supported-new-01" not in out


# ------------------------------------------------- the limit and the role selection


def test_the_smoke_run_uses_llm_roles_even_when_env_says_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """.env.example sets both roles to `deterministic`; the live script must override that."""
    from sekf.historian import LLMHistorian
    from sekf.judge import LLMJudge
    from sekf.synthesizer import LLMSynthesizer

    monkeypatch.setenv("SEKF_PROVIDER", "openai")
    monkeypatch.setenv("SEKF_LLM_API_KEY", "not-a-real-key")  # nothing is sent
    monkeypatch.setenv("SEKF_HISTORIAN", "deterministic")
    monkeypatch.setenv("SEKF_SYNTHESIZER", "deterministic")
    governance, budget = smoke_llm._live_governance(0.05, 30)
    assert isinstance(governance._historian, LLMHistorian)
    assert isinstance(governance._judge, LLMJudge)
    assert isinstance(governance._synthesizer, LLMSynthesizer)
    # One shared, budgeted provider behind all three roles, so the limit is global.
    assert governance._historian._provider is budget  # type: ignore[attr-defined]
    assert governance._judge._provider is budget  # type: ignore[attr-defined]
    assert governance._synthesizer._provider is budget  # type: ignore[attr-defined]


class _Spender:
    """A provider whose every call costs 2 calls and $0.03, like a call with one repair."""

    async def complete_json(self, **_: object) -> LLMResult:
        return LLMResult(
            parsed=_Answer(), model="m", provider="p", llm_calls=2, estimated_cost_usd=0.03
        )


class _Answer(BaseModel):
    ok: bool = True


async def _call(budgeted: smoke_llm.BudgetedProvider) -> LLMResult:
    return await budgeted.complete_json(
        messages=[], response_model=_Answer, model="m", temperature=0.0, seed=None
    )


async def test_the_call_limit_is_enforced_before_every_provider_call() -> None:
    budgeted = smoke_llm.BudgetedProvider(_Spender(), max_cost=99.0, max_calls=3)  # type: ignore[arg-type]
    await _call(budgeted)  # 2 calls: under the limit
    assert not budgeted.exhausted
    await _call(budgeted)  # 4 calls: this one overshot by one, at most a request's repairs
    assert budgeted.exhausted and budgeted.calls == 4
    with pytest.raises(ProviderUnavailable, match="smoke budget reached"):
        await _call(budgeted)  # refused: no fifth and sixth call is made
    assert budgeted.calls == 4


async def test_the_dollar_limit_is_enforced_when_the_model_is_priced() -> None:
    budgeted = smoke_llm.BudgetedProvider(_Spender(), max_cost=0.05, max_calls=99)  # type: ignore[arg-type]
    await _call(budgeted)
    await _call(budgeted)  # $0.06 >= $0.05
    assert budgeted.exhausted
    with pytest.raises(ProviderUnavailable):
        await _call(budgeted)


async def test_a_failed_call_still_counts_against_the_limit() -> None:
    class _Failing:
        async def complete_json(self, **_: object) -> LLMResult:
            raise RoleOutputInvalid(
                "bad json", usage=CallUsage(llm_calls=3, estimated_cost_usd=0.01)
            )

    budgeted = smoke_llm.BudgetedProvider(_Failing(), max_cost=99.0, max_calls=10)  # type: ignore[arg-type]
    with pytest.raises(RoleOutputInvalid):
        await _call(budgeted)
    assert budgeted.calls == 3 and budgeted.cost == pytest.approx(0.01)
