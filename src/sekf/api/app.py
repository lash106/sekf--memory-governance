"""FastAPI application. Owner: Shailen.

Only `/health` and `/v1/decide` exist in the skeleton. Baseline and experiment endpoints
(SPEC.md section 10.3) are added in later sprints.
"""

from fastapi import Depends, FastAPI, Request

from sekf import __version__
from sekf.api.orchestrator import Governance
from sekf.contracts import CONTRACT_VERSION, DecideRequest, DecideResponse
from sekf.historian import FakeHistorian
from sekf.judge import FakeJudge
from sekf.memory import InMemoryRepository
from sekf.policy import PolicyEngine
from sekf.synthesizer import FakeSynthesizer


def build_default_governance() -> Governance:
    """Offline wiring: in-memory store and deterministic fake roles."""
    return Governance(
        memory=InMemoryRepository(),
        historian=FakeHistorian(),
        judge=FakeJudge(),
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
    )


def get_governance(request: Request) -> Governance:
    governance: Governance = request.app.state.governance
    return governance


def create_app(governance: Governance | None = None) -> FastAPI:
    app = FastAPI(title="SEKF", version=__version__)
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
