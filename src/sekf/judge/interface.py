"""Judge interface (SPEC.md section 10.2). Owner: Sai Chetan.

The Judge assesses whether supplied evidence supports the candidate claim. It receives the
claim and evidence only, never gold labels or a final action.
"""

from typing import Protocol

from sekf.contracts import JudgeAssessment, JudgeRequest


class Judge(Protocol):
    async def evaluate(self, request: JudgeRequest) -> JudgeAssessment: ...
