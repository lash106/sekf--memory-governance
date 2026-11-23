"""The fixtures loader and the offline evaluation script."""

from __future__ import annotations

import asyncio

from run_state_transition_eval import SYSTEMS, report, run, to_markdown
from sekf.contracts import ClaimStatus
from sekf.memory.fixtures import build_seeded_store, case_paths, load_cases


def test_load_cases_filters_by_owner_and_split() -> None:
    krishna = load_cases("dev", "krishna")
    assert len(krishna) >= 15 and len(load_cases("dev")) > len(krishna)
    assert len(case_paths("dev", "krishna")) == len(krishna)


def test_seeded_store_holds_exactly_the_initial_memory() -> None:
    case = next(c for c in load_cases("dev", "krishna") if c.case_id.endswith("revise-chain-01"))
    repo = build_seeded_store(case)
    stored = repo.search_related_claims(case.candidate)
    assert {c.claim_id for c in stored} == {c.claim_id for c in case.initial_memory}
    assert sorted(c.status.value for c in stored) == sorted(
        [ClaimStatus.ACTIVE.value] + [ClaimStatus.SUPERSEDED.value] * 2
    )
    assert repo.decision_count() == 0


def test_stores_are_independent_between_cases() -> None:
    case = load_cases("dev", "krishna")[0]
    first, second = build_seeded_store(case), build_seeded_store(case)
    assert first is not second and first.decision_count() == second.decision_count() == 0


def test_script_reports_every_system_on_krishna_cases() -> None:
    cases = load_cases("dev", "krishna")
    data = report(cases, asyncio.run(run(cases)))
    assert set(data["systems"]) == set(SYSTEMS)
    for system in data["systems"].values():
        assert 0.0 <= system["action_accuracy"] <= 1.0
        assert system["delta"]["cases"] == len(cases)
    table = to_markdown(data)
    assert "direct_storage" in table and "council_offline_reference" in table
    # The labels separate the baselines from the governed offline reference.
    rate = {n: s["delta"]["exact_match_rate"] for n, s in data["systems"].items()}
    assert rate["council_offline_reference"] > max(
        rate["direct_storage"], rate["latest_write_wins"], rate["static_rag"]
    )
