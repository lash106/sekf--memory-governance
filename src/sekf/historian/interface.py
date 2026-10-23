"""Historian interface (SPEC.md section 10.2). Owner: Mahalakshmi.

The Historian reasons about provenance, time, claim history, and conflicts. It must not decide
evidence entailment and must not mutate memory.
"""

from typing import Protocol

from sekf.contracts import HistorianAssessment, HistorianRequest


class Historian(Protocol):
    async def analyze(self, request: HistorianRequest) -> HistorianAssessment: ...
