"""Offline smoke checks for Sprint 1 (LLMJudge) and Sprint 2 (single_judge).

Safe for local runs and Google Colab: no network, no API keys, FakeProvider only.
Run from the repository root:

    python peer_review/SaiChetan/run_offline_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

ROOT = Path(__file__).resolve().parents[2]


def _ensure_import_path() -> None:
    src = str(ROOT / "src")
    root = str(ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_decide_request() -> Any:
    from sekf.contracts import DecideRequest

    payload = json.loads((ROOT / "tests" / "fixtures" / "lab_closing_revise.json").read_text())
    return DecideRequest.model_validate(payload["request"])


def _load_repo() -> Any:
    from sekf.contracts import ExistingClaim
    from sekf.memory import InMemoryRepository

    payload = json.loads((ROOT / "tests" / "fixtures" / "lab_closing_revise.json").read_text())
    return InMemoryRepository(
        [ExistingClaim.model_validate(item) for item in payload["existing_claims"]]
    )


def _supported(_messages: Any, _model: Any) -> dict[str, Any]:
    evidence_id = UUID("33333333-3333-4333-8333-333333333333")
    return {
        "label": "SUPPORTED",
        "confidence": 0.91,
        "supporting_evidence_ids": [str(evidence_id)],
        "refuting_evidence_ids": [],
        "missing_information": [],
        "rationale": "Official schedule supports 18:00.",
    }


def _hallucinated(_messages: Any, _model: Any) -> dict[str, Any]:
    body = _supported(_messages, _model)
    body["supporting_evidence_ids"] = [str(UUID("99999999-9999-4999-8999-999999999999"))]
    return body


def _revise(_messages: Any, _model: Any) -> dict[str, Any]:
    return {
        "recommended_action": "REVISE",
        "confidence": 0.88,
        "rationale": "Newer schedule should supersede 17:00.",
    }


def _invalid(_messages: Any, _model: Any) -> dict[str, Any]:
    return {"recommended_action": "NOT_AN_ACTION", "confidence": 0.1, "rationale": "bad"}


def _boom(_messages: Any, _model: Any) -> dict[str, Any]:
    from sekf.errors import SekfError

    raise SekfError("provider exploded")


async def smoke_sprint1() -> None:
    from sekf.api import Governance
    from sekf.contracts import FinalAction, JudgeLabel, JudgeRequest
    from sekf.errors import RoleOutputInvalid
    from sekf.historian import FakeHistorian
    from sekf.judge import LLMJudge
    from sekf.policy import PolicyEngine
    from sekf.providers import FakeProvider, load_role_prompt
    from sekf.synthesizer import FakeSynthesizer

    print("== Sprint 1: LLMJudge ==")
    request = _load_decide_request()
    evidence_id = UUID("33333333-3333-4333-8333-333333333333")
    judge_req = JudgeRequest(
        candidate=request.candidate,
        evidence=request.evidence,
        run_config=request.run_config,
    )

    prompt = load_role_prompt("judge", "v1")
    assert prompt.version == "judge/v1", prompt.version
    print(f"  prompt ok: {prompt.relative_path} hash={prompt.prompt_hash[:12]}...")

    assessment = await LLMJudge(FakeProvider(_supported)).evaluate(judge_req)
    assert assessment.label is JudgeLabel.SUPPORTED
    assert assessment.supporting_evidence_ids == [evidence_id]
    print(f"  supported path ok: label={assessment.label}")

    try:
        await LLMJudge(FakeProvider(_hallucinated)).evaluate(judge_req)
        raise AssertionError("expected RoleOutputInvalid for hallucinated evidence id")
    except RoleOutputInvalid as exc:
        print(f"  hallucination rejected ok: {exc}")

    governance = Governance(
        memory=_load_repo(),
        historian=FakeHistorian(),
        judge=LLMJudge(FakeProvider(_supported)),
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
    )
    response = await governance.decide(request)
    assert response.decision.final_action is FinalAction.REVISE
    assert response.decision.rule_ids_fired == ["R8"]
    print(f"  governance+FakeHistorian+PolicyEngine ok: {response.decision.final_action}")


async def smoke_sprint2() -> None:
    from sekf.baselines import POLICY_VERSION, SingleJudgeBaseline
    from sekf.contracts import FinalAction
    from sekf.providers import FakeProvider, load_baseline_prompt

    print("== Sprint 2: SingleJudgeBaseline ==")
    request = _load_decide_request()
    repo = _load_repo()

    prompt = load_baseline_prompt("single_judge", "v1")
    assert prompt.version == "baselines/single_judge/v1"
    print(f"  prompt ok: {prompt.relative_path}")

    baseline = SingleJudgeBaseline(FakeProvider(_revise), repo)
    response = await baseline.decide(request)
    assert response.decision.final_action is FinalAction.REVISE
    assert response.decision.policy_version == POLICY_VERSION
    assert response.memory_delta is None
    assert repo.decision_count() == 0
    print(f"  decide ok: action={response.decision.final_action} policy={POLICY_VERSION}")

    bad = await SingleJudgeBaseline(FakeProvider(_invalid)).decide(request)
    assert bad.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    assert bad.decision.rule_ids_fired == ["R1"]
    print("  invalid LLM output -> R1 ok")

    boom = await SingleJudgeBaseline(FakeProvider(_boom)).decide(request)
    assert boom.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    print("  SekfError -> R1 ok")


async def _amain() -> None:
    _ensure_import_path()
    print(f"repo root: {ROOT}")
    await smoke_sprint1()
    await smoke_sprint2()
    print("ALL OFFLINE SMOKE CHECKS PASSED")


def main() -> None:
    """Run as a CLI script or from Jupyter/Colab (which already has an event loop)."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_amain())
        return

    # Notebook kernels already own the loop; asyncio.run() would raise RuntimeError.
    # A short-lived thread gets a fresh loop without needing nest_asyncio.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(lambda: asyncio.run(_amain())).result()


if __name__ == "__main__":
    main()
