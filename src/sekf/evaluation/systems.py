"""Wire the offline harness and baseline HTTP routes to the real baseline packages.

`direct_storage`, `latest_write_wins`, and `static_rag` come from Krishna's packages.
`generic_ensemble` uses FakeJudge. `single_judge` is refused here: it needs a real
provider, and this module will not substitute a different model.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import HTTPException

from sekf.baselines import DirectStorageBaseline, LatestWriteWinsBaseline, StaticRagBaseline
from sekf.baselines.generic_ensemble.baseline import GenericEnsembleBaseline
from sekf.contracts import DecideMode, DecideRequest, DecideResponse
from sekf.judge import FakeJudge
from sekf.memory import MemoryRepository

_STORAGE = {
    "direct_storage": DirectStorageBaseline,
    "latest_write_wins": LatestWriteWinsBaseline,
    "static_rag": StaticRagBaseline,
}
_OFFLINE = set(_STORAGE) | {"generic_ensemble"}


async def baseline_response(
    name: str, request: DecideRequest, memory: MemoryRepository, now: datetime
) -> DecideResponse:
    """One offline baseline decision. Unknown names and single_judge are HTTP errors."""

    if name == "single_judge":
        raise HTTPException(
            status_code=409,
            detail="single_judge needs SEKF_PROVIDER=openai and a key; no substitute model is used",
        )
    if name not in _OFFLINE:
        raise HTTPException(status_code=404, detail=f"unknown baseline {name}")
    if request.mode is DecideMode.COMMIT:
        raise HTTPException(
            status_code=409,
            detail=f"{name} does not write governed memory from this endpoint; use dry_run",
        )
    if name == "generic_ensemble":
        ensemble = GenericEnsembleBaseline(FakeJudge(), memory=memory, clock=lambda: now)
        return await ensemble.decide(request)
    baseline_cls = _STORAGE[name]
    return await baseline_cls(memory, clock=lambda: now).decide(request)
