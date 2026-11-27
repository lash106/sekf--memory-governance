"""Experiment runner: switchable Judge, SPEC.md 12.2 ablations, manifest + resumable runs."""

import asyncio
import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from sekf.contracts import DecideMode, FinalAction
from sekf.evaluation.ablations import no_policy_action
from sekf.evaluation.experiment import (
    HeldOutRefused,
    RunMismatch,
    is_held_out,
    run_experiment,
)
from sekf.evaluation.metrics import macro_f1, paired_difference_interval
from sekf.evaluation.runner import (
    ALL_SYSTEMS,
    COUNCIL,
    DEFAULT_SYSTEMS,
    NO_POLICY,
    RunContext,
    check_systems,
    derive_no_policy,
    run_case,
    run_dev_cases,
)

DEV = Path(__file__).resolve().parents[2] / "datasets" / "dev"
REVISE_CASE = DEV / "krishna" / "krishna-reliable-newer-correction-01.json"
PROMOTE_CASE = DEV / "krishna" / "krishna-supported-new-01.json"


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    data["_path"] = str(path)
    return data


def _run(system: str, data: dict[str, Any], judge: str = "fixture") -> dict[str, Any]:
    async def go() -> dict[str, Any]:
        ctx = RunContext(judge)
        try:
            return await run_case(system, data, ctx)
        finally:
            await ctx.aclose()

    return asyncio.run(go())


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEKF_PROVIDER", "fake")
    monkeypatch.delenv("SEKF_LLM_API_KEY", raising=False)


# ----------------------------------------------------------------------------- ablations


def test_without_historian_a_correction_is_promoted_instead_of_revising() -> None:
    data = _load(REVISE_CASE)
    full = _run(COUNCIL, data)
    ablated = _run("ablation_no_historian", data)
    assert full["predicted_action"] == "REVISE"
    assert ablated["predicted_action"] == "PROMOTE"  # old value would stay ACTIVE beside it


def test_without_synthesizer_the_veto_cannot_fire() -> None:
    data = copy.deepcopy(_load(PROMOTE_CASE))
    data["candidate"]["extractor_confidence"] = 0.3  # Synthesizer check S3: a strong concern
    full = _run(COUNCIL, data)
    ablated = _run("ablation_no_synthesizer", data)
    assert full["predicted_action"] == "QUARANTINE" and "R_SYN_VETO" in full["rule_ids_fired"]
    assert ablated["predicted_action"] == "PROMOTE"
    assert "R_SYN_VETO" not in ablated["rule_ids_fired"]


def test_without_policy_uses_the_synthesizer_action_and_is_derived_not_rerun() -> None:
    row = _run(COUNCIL, _load(PROMOTE_CASE))
    derived = derive_no_policy({**row, "synthesizer_action": "KEEP_EPISODIC"})
    assert derived["system"] == NO_POLICY
    assert derived["predicted_action"] == "KEEP_EPISODIC"
    assert derived["rule_ids_fired"] == ["ABLATION_NO_POLICY"]
    with pytest.raises(ValueError, match="derived"):
        _run(NO_POLICY, _load(PROMOTE_CASE))


def test_no_policy_action_is_dry_run_only_and_escalates_without_a_synthesizer() -> None:
    class Response:
        def __init__(self, mode: DecideMode) -> None:
            self.mode = mode
            self.synthesizer = None

    assert no_policy_action(Response(DecideMode.DRY_RUN)) is FinalAction.ESCALATE_FOR_REVIEW  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dry-run only"):
        no_policy_action(Response(DecideMode.COMMIT))  # type: ignore[arg-type]


def test_a_derived_system_needs_its_source_and_unknown_names_are_refused() -> None:
    check_systems([COUNCIL, NO_POLICY])
    with pytest.raises(ValueError, match="select policy-v1"):
        check_systems([NO_POLICY])
    with pytest.raises(ValueError, match="unknown system"):
        check_systems(["not-a-system"])


# ----------------------------------------------------------------------------- judges


def test_fixture_judge_can_refute_but_fake_judge_cannot() -> None:
    cases = [_load(p) for p in sorted(DEV.rglob("*.json")) if "gold" in _load(p)]
    labels: dict[str, set[str]] = {}
    for judge in ("fake", "fixture"):
        labels[judge] = {_run(COUNCIL, case, judge)["judge_label"] for case in cases}
    assert "REFUTED" not in labels["fake"]
    assert "REFUTED" in labels["fixture"]


def test_single_judge_is_reported_not_run_without_a_live_model() -> None:
    summary = asyncio.run(run_dev_cases([_load(PROMOTE_CASE)], judge="fixture"))
    entry = next(s for s in summary["systems"] if s["system"] == "single_judge")
    assert entry["status"] == "not_run"
    assert "live model" in entry["reason"]


def test_env_judge_without_a_key_still_refuses_single_judge() -> None:
    summary = asyncio.run(
        run_dev_cases([_load(PROMOTE_CASE)], judge="env", systems=["single_judge"])
    )
    assert summary["systems"][0]["status"] == "not_run"
    assert "SEKF_LLM_API_KEY" in summary["systems"][0]["reason"]


def test_default_systems_keep_the_original_report_shape() -> None:
    summary = asyncio.run(run_dev_cases([_load(PROMOTE_CASE)]))
    assert [s["system"] for s in summary["systems"]][: len(DEFAULT_SYSTEMS) - 1] == [
        name for name in DEFAULT_SYSTEMS if name != "single_judge"
    ]
    assert summary["judge"] == "fake"


# ----------------------------------------------------------------------------- statistics


def test_paired_difference_interval() -> None:
    gold = ["PROMOTE", "REVISE", "REJECT", "PROMOTE"] * 5
    same = paired_difference_interval(gold, gold, gold, macro_f1, n_resamples=200)
    assert (same.point, same.low, same.high) == (0.0, 0.0, 0.0)
    worse = ["PROMOTE"] * len(gold)
    better = paired_difference_interval(gold, gold, worse, macro_f1, n_resamples=200)
    assert better.point > 0 and better.low > 0
    with pytest.raises(ValueError):
        paired_difference_interval(gold, gold, worse[:-1], macro_f1)


# ----------------------------------------------------------------------------- experiments


def _dataset(tmp_path: Path, name: str = "dev") -> Path:
    root = tmp_path / name
    root.mkdir(parents=True)
    for path in (REVISE_CASE, PROMOTE_CASE):
        shutil.copy(path, root / path.name)
    return root


def test_experiment_writes_manifest_results_and_summary_then_resumes(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    run_dir = tmp_path / "run"
    systems = [COUNCIL, NO_POLICY, "latest_write_wins", "single_judge"]

    first = asyncio.run(run_experiment(dataset_dir=dataset, run_dir=run_dir, systems=systems))
    assert first.counts == {"ran": 4, "ok": 4, "error": 0, "skipped": 0}
    for name in ("manifest.json", "results.jsonl", "summary.json"):
        assert (run_dir / name).is_file()
    assert first.manifest.finished_at is not None
    assert first.manifest.model.startswith("fixture/judge@")
    summary = json.loads((run_dir / "summary.json").read_text())
    by_name = {s["system"]: s for s in summary["systems"]}
    assert by_name[NO_POLICY]["n"] == 2  # derived from the council rows, not run
    assert by_name["single_judge"]["status"] == "not_run"
    assert summary["incomplete_pairs"] == 0 and summary["held_out"] is False

    again = asyncio.run(run_experiment(dataset_dir=dataset, run_dir=run_dir, systems=systems))
    assert again.counts["ran"] == 0 and again.counts["skipped"] == 4
    assert again.manifest.run_id == first.manifest.run_id


def test_a_changed_experiment_is_refused_instead_of_mixed(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    run_dir = tmp_path / "run"
    asyncio.run(run_experiment(dataset_dir=dataset, run_dir=run_dir, systems=[COUNCIL]))
    with pytest.raises(RunMismatch, match="provider"):
        asyncio.run(
            run_experiment(dataset_dir=dataset, run_dir=run_dir, systems=[COUNCIL], judge="fake")
        )
    with pytest.raises(RunMismatch, match="systems"):
        asyncio.run(run_experiment(dataset_dir=dataset, run_dir=run_dir, systems=ALL_SYSTEMS))


def test_held_out_data_needs_explicit_confirmation(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, "test")
    assert is_held_out(dataset)
    with pytest.raises(HeldOutRefused, match="confirm-held-out"):
        asyncio.run(run_experiment(dataset_dir=dataset, run_dir=tmp_path / "run"))
