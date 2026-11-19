"""Review fixes: declared decision fields, held-out sweeps, cached predictions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.conftest import (
    make_candidate,
    make_context,
    make_existing,
    make_historian,
    make_judge,
    make_synth,
)

from sekf.contracts import DecisionRecord, FinalAction
from sekf.errors import ConfigurationError, ProviderUnavailable
from sekf.evaluation.cached import CachedCase, author_of, load_predictions
from sekf.evaluation.capture import capture_cases, offline_roles
from sekf.evaluation.cv import default_grid, leave_one_author_out, project_cached
from sekf.evaluation.metrics import ScoredCase, efficiency_report, error_category
from sekf.evaluation.retry import with_retries
from sekf.evaluation.runner import load_dev_cases
from sekf.policy import PolicyEngine, PolicyEngineV2


def _cached(author: str, action: str = "PROMOTE") -> CachedCase:
    cand = make_candidate()
    historian = make_historian(cand)
    judge = make_judge(cand, confidence=0.95)
    context = make_context(candidate=cand, historian=historian, judge=judge)
    return CachedCase(
        case_id=f"{author}-1",
        author=author,
        gold_action=action,
        gold_judge="SUPPORTED",
        category="supported_new_claim",
        latency_ms=12.0,
        retrieval_recall=1.0,
        context=context,
    )


def test_author_folders() -> None:
    assert author_of("datasets/dev/krishna/case.json") == "krishna"
    assert author_of("datasets/dev/mahalakshmi/case.json") == "mahalakshmi"
    assert author_of("datasets/dev/shailen/case.json") == "shailen"
    assert author_of("datasets/dev/supported-new-01.json") == "sai_chetan"


def test_decision_fields_are_optional_and_unknown_fields_still_fail() -> None:
    cand = make_candidate().model_copy(update={"metadata": {"risk_tier": "HIGH_RISK"}})
    decision = PolicyEngineV2().decide(
        make_context(candidate=cand, historian=make_historian(cand), judge=make_judge(cand))
    )
    assert decision.risk_tier == "HIGH_RISK"
    assert decision.needs_verification is False
    payload = decision.model_dump(mode="json")
    payload.pop("risk_tier")
    payload.pop("needs_verification")
    payload.pop("compliance_outcome")
    payload.pop("next_evidence_required")
    restored = DecisionRecord.model_validate(payload)
    assert restored.risk_tier is None
    assert restored.needs_verification is False
    payload["not_a_contract_field"] = 1
    with pytest.raises(ValidationError):
        DecisionRecord.model_validate(payload)
    payload.pop("not_a_contract_field")
    payload["risk_tier"] = "EXTREME"
    with pytest.raises(ValidationError):
        DecisionRecord.model_validate(payload)


def test_temperature_is_applied_before_the_support_rule() -> None:
    cand = make_candidate()
    context = make_context(
        candidate=cand,
        historian=make_historian(cand),
        judge=make_judge(cand, confidence=0.85),
    )
    raw = PolicyEngineV2().decide(context)
    cooled = PolicyEngineV2(judge_temperature=5.0).decide(context)
    assert raw.final_action is FinalAction.PROMOTE
    assert raw.policy_version == "policy-v2"
    assert cooled.final_action is FinalAction.KEEP_EPISODIC
    assert cooled.policy_version.startswith("policy-v2@t")
    assert PolicyEngine().decide(context).policy_version == "policy-v1"


def test_compliance_field_does_not_change_the_action() -> None:
    cand = make_candidate().model_copy(
        update={"metadata": {"days_since_purchase": 40, "return_window_days": 30}}
    )
    context = make_context(
        candidate=cand, historian=make_historian(cand), judge=make_judge(cand, confidence=0.95)
    )
    decision = PolicyEngineV2().decide(context)
    assert decision.final_action is FinalAction.PROMOTE
    assert decision.compliance_outcome is not None
    assert "DENY" in decision.compliance_outcome
    assert "R_COMPLIANCE" in decision.rule_ids_fired


def test_quarantine_markdown_names_the_next_evidence(tmp_path: Path) -> None:
    cand = make_candidate()
    historian = make_historian(cand)
    judge = make_judge(cand, confidence=0.95)
    synth = make_synth(cand, historian, judge, FinalAction.QUARANTINE, confidence=0.9)
    context = make_context(
        candidate=cand,
        historian=historian,
        judge=judge,
        synthesizer=synth,
        related_claims=[make_existing()],
    )
    decision = PolicyEngineV2().decide(context)
    assert decision.final_action is FinalAction.QUARANTINE
    assert decision.next_evidence_required
    project_cached(
        [
            CachedCase(
                case_id="q",
                author="krishna",
                gold_action="QUARANTINE",
                gold_judge="SUPPORTED",
                category="conflict",
                latency_ms=1,
                retrieval_recall=0.0,
                context=context,
            )
        ],
        tmp_path,
    )
    text = (tmp_path / "L2_QUARANTINE.md").read_text(encoding="utf-8")
    assert "next_evidence_required:" in text
    assert "failing_clause:" in text


def test_leave_one_author_out_refuses_one_group_and_scores_two() -> None:
    with pytest.raises(ValueError, match="not reported"):
        leave_one_author_out([_cached("krishna")], default_grid())
    report = leave_one_author_out([_cached("krishna"), _cached("shailen")], default_grid())
    assert report["method"] == "leave_one_author_out"
    assert "best" not in report
    assert {fold["held_out_author"] for fold in report["folds"]} == {"krishna", "shailen"}


def test_retrieval_miss_is_not_called_a_policy_error() -> None:
    missed = ScoredCase("PROMOTE", "QUARANTINE", retrieval_recall=0.0, latency_ms=10)
    role = ScoredCase(
        "PROMOTE",
        "PROMOTE",
        gold_judge="SUPPORTED",
        predicted_judge="REFUTED",
        retrieval_recall=1.0,
        latency_ms=30,
    )
    assert error_category(missed) == "retrieval"
    assert error_category(role) == "role"
    report = efficiency_report([missed, role])
    assert report["retrieval_recall_at_k"] == 0.5
    assert report["p95_latency_ms"] == 30
    counts = report["error_categories"]
    assert isinstance(counts, dict)
    assert counts["retrieval"] == 1
    assert counts["policy"] == 0


def test_capture_file_feeds_the_sweep_and_the_judge_report(tmp_path: Path) -> None:
    picked: list[dict] = []
    seen: set[str] = set()
    for data in load_dev_cases():
        author = author_of(str(data["_path"]))
        if author in seen:
            continue
        seen.add(author)
        picked.append(data)
        if len(picked) == 2:
            break
    assert len(picked) == 2
    dest = tmp_path / "predictions.jsonl"
    historian, judge, synthesizer = offline_roles()
    count = asyncio.run(
        capture_cases(picked, dest, historian=historian, judge=judge, synthesizer=synthesizer)
    )
    assert count == 2
    loaded = load_predictions(dest)
    assert {case.author for case in loaded} == seen
    report = leave_one_author_out(loaded, default_grid())
    assert len(report["folds"]) == 2
    from analyze_judge_evidence import load_observations

    observations = load_observations(dest)
    assert len(observations) == 2
    raw = json.loads(dest.read_text(encoding="utf-8").splitlines()[0])
    assert raw["roles"]["judge"]["label"]


def test_rate_limit_retries_and_other_provider_errors_do_not() -> None:
    calls = {"n": 0}

    async def _flaky() -> dict[str, bool]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ProviderUnavailable("LLM rate limited (HTTP 429)")
        return {"ok": True}

    async def _sleep(_delay: float) -> None:
        return None

    assert asyncio.run(with_retries(_flaky, sleep=_sleep, label="case")) == {"ok": True}
    assert calls["n"] == 2

    async def _down() -> dict[str, bool]:
        raise ProviderUnavailable("LLM provider down")

    with pytest.raises(ProviderUnavailable, match="provider down"):
        asyncio.run(with_retries(_down, sleep=_sleep, attempts=3))


def test_live_capture_refuses_without_a_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from capture_live_predictions import main
    from sekf.judge import compose

    monkeypatch.setattr(compose, "_ENV_LOADED", True)
    monkeypatch.setenv("SEKF_PROVIDER", "fake")
    monkeypatch.setenv("SEKF_LLM_API_KEY", "")
    assert main(["--live", "--out", str(tmp_path / "predictions.jsonl")]) == 2


def test_bad_temperature_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from sekf.api.app import build_policy_from_env
    from sekf.judge import compose

    monkeypatch.setattr(compose, "_ENV_LOADED", True)
    monkeypatch.setenv("SEKF_POLICY", "v2")
    monkeypatch.setenv("SEKF_JUDGE_TEMPERATURE", "nope")
    with pytest.raises(ConfigurationError):
        build_policy_from_env()


def test_notebook_mentions_the_offline_scripts() -> None:
    notebook_path = Path(__file__).resolve().parents[2] / "experiments" / "sekf_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "assert_prompts_frozen" in source
    assert "capture_live_predictions.py" in source
    assert "sweep_policy_thresholds.py" in source
    assert "analyze_judge_evidence.py" in source
    assert "run_offline_experiment.py" in source
