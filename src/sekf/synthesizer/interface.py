"""Synthesizer interface (SPEC.md section 10.2). Owner: Mahalakshmi.

Advisory only: it has no access to memory mutation and the policy may only use it to make an
outcome more conservative (the veto step).
"""

from typing import Protocol

from sekf.contracts import SynthesisRequest, SynthesizerRecommendation


class Synthesizer(Protocol):
    async def synthesize(self, request: SynthesisRequest) -> SynthesizerRecommendation: ...
