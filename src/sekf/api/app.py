"""FastAPI application. Owner: Shailen.

Only `/health` and `/v1/decide` exist so far. Baseline and experiment endpoints
(SPEC.md section 10.3) are added in later sprints.

Which implementation each role uses is chosen from the environment (or `.env`):
`SEKF_MEMORY` (`sql` default, or `memory`), `SEKF_HISTORIAN` (`deterministic` default, or
`fake`), `SEKF_PROVIDER` (`fake` default, or `openai`; this selects the Judge). The Synthesizer
is still the fake until its real implementation lands. Unknown memory/Historian values raise
`ConfigurationError`; an unknown `SEKF_PROVIDER` raises `ProviderUnavailable` (raised by the
Judge's own composition code). Nothing silently falls back.

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

from fastapi import Depends, FastAPI, Request

from sekf import __version__
from sekf.api.orchestrator import Governance
from sekf.contracts import CONTRACT_VERSION, DecideRequest, DecideResponse
from sekf.errors import ConfigurationError
from sekf.historian import DeterministicHistorian, FakeHistorian, Historian
from sekf.judge.compose import build_judge_from_env, load_project_env
from sekf.memory import InMemoryRepository, MemoryRepository, build_memory_repository
from sekf.policy import PolicyEngine
from sekf.synthesizer import FakeSynthesizer


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


def build_historian_from_env() -> Historian:
    """`SEKF_HISTORIAN=deterministic` (default, the rule-based Historian) or `fake`."""
    if _env_choice("SEKF_HISTORIAN", "deterministic", {"deterministic", "fake"}) == "fake":
        return FakeHistorian()
    return DeterministicHistorian()


def build_default_governance() -> Governance:
    """Wire the application from the environment (see the module docstring).

    The defaults need no network and no key: SQLite store, deterministic Historian, fake Judge,
    fake Synthesizer, deterministic policy.
    """

    return Governance(
        memory=build_memory_from_env(),
        historian=build_historian_from_env(),
        judge=build_judge_from_env(),
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
    )


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

    return app
