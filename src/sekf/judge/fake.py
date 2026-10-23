"""Deterministic fake Judge for the skeleton and offline tests (no LLM)."""

from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from sekf.contracts import JudgeAssessment, JudgeLabel, JudgeRequest


class FakeJudge:
    """SUPPORTED if any evidence text contains the claim's object text, else INSUFFICIENT."""

    async def evaluate(self, request: JudgeRequest) -> JudgeAssessment:
        cand = request.candidate
        needle = cand.object_text.strip().lower()
        supporting = [e.evidence_id for e in request.evidence if needle in e.content.lower()]
        label = JudgeLabel.SUPPORTED if supporting else JudgeLabel.INSUFFICIENT_EVIDENCE
        return JudgeAssessment(
            assessment_id=uuid5(
                NAMESPACE_URL,
                "judge:"
                + ":".join(
                    [str(cand.claim_id)]
                    + [str(e.evidence_id) + e.content_hash for e in request.evidence]
                ),
            ),
            candidate_claim_id=cand.claim_id,
            label=label,
            confidence=0.9,
            supporting_evidence_ids=supporting,
            missing_information=[] if supporting else ["No evidence mentions the claimed value."],
            rationale="Evidence mentions the claimed value."
            if supporting
            else "No supplied evidence mentions the claimed value.",
            model="fake/judge",
            provider="fake",
            prompt_version="fake-judge-v0",
            prompt_hash=sha256(b"fake-judge-v0").hexdigest(),
            temperature=request.run_config.temperature,
            seed=request.run_config.seed,
            llm_calls=0,
        )
