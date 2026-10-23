from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError
from tests.conftest import META

from generate_contracts import OUT_DIR, generated_schemas
from sekf.contracts import (
    CandidateClaim,
    DecideRequest,
    EvidenceRecord,
    ExistingClaim,
    FinalAction,
    JudgeAssessment,
)


def candidate_kwargs() -> dict[str, Any]:
    return {
        "claim_id": uuid4(),
        "claim_text": "x",
        "subject_key": "s",
        "predicate_key": "p",
        "object_text": "o",
        "source_id": "src",
        "observed_at": datetime(2026, 10, 1, tzinfo=UTC),
    }


def test_committed_json_schemas_are_current() -> None:
    """If this fails, run `make contracts` and commit the result."""
    for filename, text in generated_schemas().items():
        path = OUT_DIR / filename
        assert path.exists(), f"missing {filename}; run `make contracts`"
        assert path.read_text() == text, f"{filename} is stale; run `make contracts`"
    committed = {p.name for p in OUT_DIR.glob("*.json")}
    assert committed == set(generated_schemas()), "stale schema files; run `make contracts`"


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateClaim(**candidate_kwargs(), surprise="nope")


def test_timestamps_must_be_timezone_aware() -> None:
    kwargs = candidate_kwargs() | {"observed_at": datetime(2026, 10, 1)}
    with pytest.raises(ValidationError):
        CandidateClaim(**kwargs)


def test_validity_window_must_be_ordered() -> None:
    kwargs = candidate_kwargs() | {
        "valid_from": datetime(2026, 10, 2, tzinfo=UTC),
        "valid_to": datetime(2026, 10, 1, tzinfo=UTC),
    }
    with pytest.raises(ValidationError):
        CandidateClaim(**kwargs)


def test_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        JudgeAssessment(
            assessment_id=uuid4(),
            candidate_claim_id=uuid4(),
            label="SUPPORTED",
            confidence=1.5,
            rationale="r",
            **META,
        )


def test_invalid_enum_value_rejected() -> None:
    with pytest.raises(ValidationError):
        JudgeAssessment(
            assessment_id=uuid4(),
            candidate_claim_id=uuid4(),
            label="MAYBE",
            confidence=0.5,
            rationale="r",
            **META,
        )


def test_evidence_hash_is_filled_and_checked() -> None:
    base = {
        "evidence_id": uuid4(),
        "content": "hello",
        "source_id": "s",
        "source_type": "doc",
        "retrieved_at": datetime(2026, 10, 1, tzinfo=UTC),
        "reliability": 0.5,
    }
    ev = EvidenceRecord(**base)
    assert len(ev.content_hash) == 64
    with pytest.raises(ValidationError):
        EvidenceRecord(**base, content_hash="0" * 64)


def test_canonical_action_names_are_exact() -> None:
    assert {a.value for a in FinalAction} == {
        "PROMOTE",
        "ATTACH_EVIDENCE",
        "KEEP_EPISODIC",
        "QUARANTINE",
        "REVISE",
        "REJECT",
        "ESCALATE_FOR_REVIEW",
    }


def test_fixture_validates_against_contracts(lab_fixture: dict[str, Any]) -> None:
    DecideRequest.model_validate(lab_fixture["request"])
    for claim in lab_fixture["existing_claims"]:
        ExistingClaim.model_validate(claim)


def test_role_output_requires_reproducibility_metadata() -> None:
    base = {
        "assessment_id": uuid4(),
        "candidate_claim_id": uuid4(),
        "label": "SUPPORTED",
        "confidence": 0.9,
        "rationale": "r",
    }
    JudgeAssessment(**base, **META)  # complete metadata is accepted
    for missing in ("provider", "prompt_hash", "temperature", "llm_calls"):
        incomplete = {k: v for k, v in META.items() if k != missing}
        with pytest.raises(ValidationError):
            JudgeAssessment(**base, **incomplete)
    with pytest.raises(ValidationError):
        JudgeAssessment(**base, **(META | {"provider": ""}))


def test_commit_bundle_rejects_inconsistent_content() -> None:
    from tests.conftest import make_candidate, make_context, make_historian, make_judge, make_synth

    from sekf.contracts import CommitBundle
    from sekf.policy import PolicyEngine

    cand = make_candidate()
    h, j = make_historian(cand), make_judge(cand)
    s = make_synth(cand, h, j, FinalAction.PROMOTE)
    decision = PolicyEngine().decide(
        make_context(candidate=cand, historian=h, judge=j, synthesizer=s)
    )
    CommitBundle(candidate=cand, decision=decision, historian=h, judge=j, synthesizer=s)  # ok

    other = make_candidate()
    with pytest.raises(ValidationError):  # role output for another candidate
        CommitBundle(
            candidate=cand,
            decision=decision,
            historian=make_historian(other),
            judge=j,
            synthesizer=s,
        )
    with pytest.raises(ValidationError):  # synthesizer references different assessments
        CommitBundle(
            candidate=cand,
            decision=decision,
            historian=h,
            judge=j,
            synthesizer=make_synth(cand, make_historian(cand), j, FinalAction.PROMOTE),
        )
    cited = j.model_copy(update={"supporting_evidence_ids": [uuid4()]})
    with pytest.raises(ValidationError):  # judge cites evidence not in the bundle
        CommitBundle(candidate=cand, decision=decision, historian=h, judge=cited, synthesizer=s)


def test_request_rejects_duplicate_evidence_ids(lab_fixture: dict[str, Any]) -> None:
    request = lab_fixture["request"]
    changed = dict(request["evidence"][0], reliability=0.1)  # same evidence_id, other content
    with pytest.raises(ValidationError, match="duplicate evidence_id"):
        DecideRequest.model_validate({**request, "evidence": [request["evidence"][0], changed]})
