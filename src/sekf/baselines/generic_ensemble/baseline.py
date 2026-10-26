"""Generic-ensemble baseline as a DecideResponse producer.

Runs FakeHistorian + GenericEnsembleJudge + FakeSynthesizer + PolicyEngine so the
pilot can compare final actions under matched settings. Owner: Sai Chetan.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from sekf.api import Governance
from sekf.baselines.generic_ensemble.ensemble import GenericEnsembleJudge
from sekf.contracts import DecideRequest, DecideResponse
from sekf.historian import FakeHistorian, Historian
from sekf.judge import Judge
from sekf.memory import InMemoryRepository, MemoryRepository
from sekf.policy import DecisionPolicy, PolicyEngine
from sekf.synthesizer import FakeSynthesizer, Synthesizer


def _utcnow() -> datetime:
    return datetime.now(UTC)


class GenericEnsembleBaseline:
    """Council-shaped baseline with three identical Judge members and no specialization."""

    def __init__(
        self,
        member_judge: Judge,
        *,
        memory: MemoryRepository | None = None,
        historian: Historian | None = None,
        synthesizer: Synthesizer | None = None,
        policy: DecisionPolicy | None = None,
        n_members: int = 3,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._governance = Governance(
            memory=memory if memory is not None else InMemoryRepository(),
            historian=historian if historian is not None else FakeHistorian(),
            judge=GenericEnsembleJudge(member_judge, n_members=n_members),
            synthesizer=synthesizer if synthesizer is not None else FakeSynthesizer(),
            policy=policy if policy is not None else PolicyEngine(),
            clock=clock,
        )

    async def decide(self, request: DecideRequest) -> DecideResponse:
        return await self._governance.decide(request)
