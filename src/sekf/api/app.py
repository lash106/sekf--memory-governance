"""FastAPI application. Owner: Shailen.

Only `/health` and `/v1/decide` exist so far. Baseline and experiment endpoints
(SPEC.md section 10.3) are added in later sprints.

Which implementation each role uses is chosen from the environment (or `.env`):
`SEKF_MEMORY` (`sql` default, or `memory`), `SEKF_HISTORIAN` and `SEKF_SYNTHESIZER`
(`deterministic` default, `fake`, or `llm`), `SEKF_PROVIDER` (`fake` default, or `openai`; this
selects the Judge and is the model provider the `llm` roles use). `llm` means the rule-based
role plus a model check, and it needs `SEKF_PROVIDER=openai` and a key. Unknown memory/role
values, and `llm` without a live provider, raise `ConfigurationError`; an unknown
`SEKF_PROVIDER` raises `ProviderUnavailable` (from the Judge's own composition code). Nothing
silently falls back.

`sql` is the real store: decisions change claim state and survive restarts (SQLite at
`SEKF_DB_URL`). `memory` is a test double that records decisions but applies no state
transitions, so promoted claims never become active knowledge. Do not use it for demos.

The lifespan calls `Governance.aclose()` on shutdown, which closes every role and the memory
store that owns a resource (`OpenAICompatibleProvider` holds an `httpx.AsyncClient`; the SQL
store holds a connection pool).
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request

from sekf import __version__
from sekf.api.orchestrator import Governance
from sekf.contracts import CONTRACT_VERSION, DecideRequest, DecideResponse
from sekf.errors import ConfigurationError
from sekf.historian import DeterministicHistorian, FakeHistorian, Historian, LLMHistorian
from sekf.judge.compose import build_judge_from_env, load_project_env
from sekf.memory import InMemoryRepository, MemoryRepository, build_memory_repository
from sekf.policy import PolicyEngine, PolicyEngineV2
from sekf.providers import create_provider_from_env
from sekf.synthesizer import (
    DeterministicSynthesizer,
    FakeSynthesizer,
    LLMSynthesizer,
    Synthesizer,
)


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    load_project_env()
    value = (os.getenv(name) or default).strip().lower() or default
    if value not in allowed:
        raise ConfigurationError(
            f"unknown {name}={value!r}; expected one of {sorted(allowed)} (no silent fallback)"
        )
    return value


def build_memory_from_env() -> MemoryRepository:
    """`SEKF_MEMORY=sql` (default, SQLite at `SEKF_DB_URL`) or `memory` (test double)."""
    if _env_choice("SEKF_MEMORY", "sql", {"memory", "sql"}) == "memory":
        return InMemoryRepository()
    return build_memory_repository()


def _require_live_provider(role: str) -> None:
    """An `llm` role needs a real model and a key; fail at startup, not on every request."""
    if (os.getenv("SEKF_PROVIDER") or "fake").strip().lower() not in {
        "openai",
        "openai_compatible",
    }:
        raise ConfigurationError(
            f"SEKF_{role}=llm needs SEKF_PROVIDER=openai (and SEKF_LLM_API_KEY); "
            "no stand-in model is used"
        )
    if not os.getenv("SEKF_LLM_API_KEY"):
        raise ConfigurationError(
            f"SEKF_{role}=llm needs SEKF_LLM_API_KEY, which is not set; every request would fail"
        )


def build_historian_from_env() -> Historian:
    """`SEKF_HISTORIAN`: `deterministic` (default, rule-based), `fake`, or `llm` (rules + model)."""
    choice = _env_choice("SEKF_HISTORIAN", "deterministic", {"deterministic", "fake", "llm"})
    if choice == "fake":
        return FakeHistorian()
    if choice == "llm":
        _require_live_provider("HISTORIAN")
        return LLMHistorian(create_provider_from_env())
    return DeterministicHistorian()


def build_synthesizer_from_env() -> Synthesizer:
    """`SEKF_SYNTHESIZER`: `deterministic` (default), `fake`, or `llm` (rules + model)."""
    choice = _env_choice("SEKF_SYNTHESIZER", "deterministic", {"deterministic", "fake", "llm"})
    if choice == "fake":
        return FakeSynthesizer()
    if choice == "llm":
        _require_live_provider("SYNTHESIZER")
        return LLMSynthesizer(create_provider_from_env())
    return DeterministicSynthesizer()


def build_default_governance() -> Governance:
    """Wire the application from the environment (see the module docstring).

    The defaults need no network and no key: SQLite store, rule-based Historian and Synthesizer,
    fake Judge, deterministic policy.
    """

    memory = build_memory_from_env()
    markdown_dir = os.getenv("SEKF_MARKDOWN_DIR")
    if markdown_dir:
        memory.markdown_root = Path(markdown_dir)
    return Governance(
        memory=memory,
        historian=build_historian_from_env(),
        judge=build_judge_from_env(),
        synthesizer=build_synthesizer_from_env(),
        policy=build_policy_from_env(),
    )


def build_policy_from_env() -> PolicyEngine:
    """`SEKF_POLICY=v1` (default) or `v2`. Any other value is an error."""

    choice = _env_choice("SEKF_POLICY", "v1", {"v1", "policy-v1", "v2", "policy-v2"})
    if choice in {"v2", "policy-v2"}:
        return PolicyEngineV2(judge_temperature=_judge_temperature())
    return PolicyEngine()


def _judge_temperature() -> float:
    """Fitted Judge temperature. Unset means 1.0, which leaves confidence unchanged."""

    raw = os.getenv("SEKF_JUDGE_TEMPERATURE")
    if raw is None or not raw.strip():
        return 1.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(
            "SEKF_JUDGE_TEMPERATURE must be a number > 0; no silent fallback"
        ) from exc
    if value <= 0:
        raise ConfigurationError("SEKF_JUDGE_TEMPERATURE must be > 0; no silent fallback")
    return value


def get_governance(request: Request) -> Governance:
    governance: Governance = request.app.state.governance
    return governance


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Yield the app, then release everything the governance object owns."""

    yield
    governance = getattr(app.state, "governance", None)
    if governance is not None:
        await governance.aclose()


def create_app(governance: Governance | None = None) -> FastAPI:
    app = FastAPI(title="SEKF", version=__version__, lifespan=_lifespan)
    app.state.governance = governance or build_default_governance()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__, "contract_version": CONTRACT_VERSION}

    @app.post("/v1/decide", response_model=DecideResponse)
    async def decide(
        body: DecideRequest, gov: Governance = Depends(get_governance)
    ) -> DecideResponse:
        return await gov.decide(body)

    @app.get("/v1/customers/{subject_key}/brief")
    async def customer_brief(
        subject_key: str, gov: Governance = Depends(get_governance)
    ) -> dict[str, object]:
        return gov.customer_brief(subject_key)

    @app.post("/v1/baselines/{baseline_name}/decide", response_model=DecideResponse)
    async def baseline_decide(
        baseline_name: str, body: DecideRequest, gov: Governance = Depends(get_governance)
    ) -> DecideResponse:
        from sekf.evaluation.systems import baseline_response

        return await baseline_response(baseline_name, body, gov._memory, gov._clock())

    @app.post("/v1/experiments/run")
    async def run_experiment(limit: int = 0) -> dict[str, object]:
        from sekf.evaluation.runner import start_offline_run

        try:
            return await start_offline_run(app, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/experiments/{run_id}")
    async def get_experiment(run_id: str) -> dict[str, object]:
        runs: dict[str, dict[str, object]] = getattr(app.state, "experiments", {})
        found = runs.get(run_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id {run_id}")
        return found

    return app
