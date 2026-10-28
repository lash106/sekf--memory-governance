"""FastAPI application. Owner: Shailen.

Only `/health` and `/v1/decide` exist in the skeleton. Baseline and experiment endpoints
(SPEC.md section 10.3) are added in later sprints.

The lifespan closes the Judge's provider on shutdown. `OpenAICompatibleProvider`
owns an `httpx.AsyncClient`; leaving it open leaks sockets across reloads and tests.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request

from sekf import __version__
from sekf.api.orchestrator import Governance
from sekf.contracts import CONTRACT_VERSION, DecideRequest, DecideResponse
from sekf.historian import FakeHistorian
from sekf.judge.compose import build_judge_from_env
from sekf.memory import InMemoryRepository
from sekf.policy import PolicyEngine
from sekf.synthesizer import FakeSynthesizer


def build_default_governance() -> Governance:
    """Wire the walking skeleton. Judge follows `SEKF_PROVIDER`.

    `fake` (default, and every test that does not opt in) uses FakeJudge.
    `openai` uses LLMJudge with the OpenAI-compatible provider. Historian and
    Synthesizer stay fakes until those owners land real roles.
    """

    return Governance(
        memory=InMemoryRepository(),
        historian=FakeHistorian(),
        judge=build_judge_from_env(),
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
    )


def get_governance(request: Request) -> Governance:
    governance: Governance = request.app.state.governance
    return governance


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Yield the app, then close a Judge that owns a live HTTP client."""

    yield
    governance = getattr(app.state, "governance", None)
    judge = getattr(governance, "_judge", None)
    close = getattr(judge, "aclose", None)
    if close is not None:
        await close()


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
