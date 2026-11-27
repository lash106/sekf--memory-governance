"""Experiment runner for labeled case files. Owner: Shailen.

Scores every system on the same cases, with the same Judge, in dry-run mode on a fresh
in-memory store per case (nothing is ever written to a real database):

- Storage baselines from `src/sekf/baselines/`: `direct_storage`, `latest_write_wins`,
  `static_rag`.
- Judge baselines: `generic_ensemble` (three votes from the chosen Judge) and `single_judge`
  (needs a live model; reported as `not_run` otherwise, with no substitute).
- The governed council on policy-v1 (`policy-v1`).
- SPEC.md section 12.2 ablations: `ablation_no_historian`, `ablation_no_synthesizer`,
  `ablation_no_policy` (Synthesizer action used directly; derived from the `policy-v1` row of
  the same case, so it costs nothing extra), `ablation_no_provenance`, `ablation_no_temporal`.

The Judge is chosen once per run (`JUDGE_CHOICES`):

- `fake`: `FakeJudge`. CI stand-in; it never says REFUTED or MIXED_EVIDENCE.
- `fixture`: `FixtureJudge`, Sai Chetan's rule-based lexical Judge. Offline and deterministic,
  and it can refute. It was developed against `datasets/dev`, so its dev scores are optimistic.
- `env`: `build_judge_from_env()`, the live `LLMJudge` when `SEKF_PROVIDER=openai`.

Within one run, each case's Judge output is computed once and reused by every system that
uses the Judge. The comparison is paired (no system wins because of a lucky resample) and a
live run pays for one Judge call per case, not one per system.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from sekf.api.orchestrator import Governance
from sekf.baselines import (
    DirectStorageBaseline,
    LatestWriteWinsBaseline,
    SingleJudgeBaseline,
    StaticRagBaseline,
)
from sekf.baselines.generic_ensemble.baseline import GenericEnsembleBaseline
from sekf.contracts import (
    CandidateClaim,
    DecideMode,
    DecideRequest,
    DecideResponse,
    EvidenceRecord,
    ExistingClaim,
    JudgeAssessment,
    JudgeRequest,
)
from sekf.evaluation.ablations import DisabledSynthesizer, NeutralHistorian, no_policy_action
from sekf.evaluation.metrics import (
    false_promotion_rate,
    macro_f1,
    paired_bootstrap_interval,
    paired_difference_interval,
    percentile,
    promotion_scores,
    revision_accuracy,
    unnecessary_quarantine_rate,
)
from sekf.historian import AblatedHistorian, DeterministicHistorian, Historian
from sekf.judge import FakeJudge, FixtureJudge, Judge, build_judge_from_env
from sekf.memory.in_memory import InMemoryRepository
from sekf.memory.retrieval import hybrid_search, retrieval_recall_at_k
from sekf.policy import PolicyEngine
from sekf.providers import create_provider_from_env
from sekf.synthesizer import DeterministicSynthesizer, Synthesizer

_DEV = Path(__file__).resolve().parents[3] / "datasets" / "dev"

JUDGE_CHOICES = ("fake", "fixture", "env")
COUNCIL = "policy-v1"
NO_POLICY = "ablation_no_policy"
STORAGE_BASELINES = ("direct_storage", "latest_write_wins", "static_rag")
JUDGE_BASELINES = ("generic_ensemble", "single_judge")
ABLATIONS = (
    "ablation_no_historian",
    "ablation_no_synthesizer",
    NO_POLICY,
    "ablation_no_provenance",
    "ablation_no_temporal",
)
ALL_SYSTEMS = (*STORAGE_BASELINES, *JUDGE_BASELINES, COUNCIL, *ABLATIONS)
# What `run_dev_cases` scored before ablations existed; the API route keeps this default.
DEFAULT_SYSTEMS = (*STORAGE_BASELINES, "generic_ensemble", COUNCIL, "single_judge")
# Systems computed from another system's row instead of being run.
DERIVED = {NO_POLICY: COUNCIL}

_STORAGE = {
    "direct_storage": DirectStorageBaseline,
    "latest_write_wins": LatestWriteWinsBaseline,
    "static_rag": StaticRagBaseline,
}


# ----------------------------------------------------------------------------- cases


def load_dev_cases(root: Path | None = None) -> list[dict[str, Any]]:
    """Load labeled case files. Files without a gold object are skipped."""

    base = root or _DEV
    cases: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "gold" in data and "candidate" in data:
            data["_path"] = str(path)
            cases.append(data)
    return cases


def case_id(data: dict[str, Any]) -> str:
    """Stable identifier for one case: its `case_id`, else its file path."""

    return str(data.get("case_id") or data.get("_path"))


def _request(data: dict[str, Any]) -> DecideRequest:
    return DecideRequest(
        candidate=CandidateClaim.model_validate(data["candidate"]),
        evidence=[EvidenceRecord.model_validate(item) for item in data.get("evidence", [])],
        mode=DecideMode.DRY_RUN,
    )


def _memory(data: dict[str, Any]) -> InMemoryRepository:
    claims = [ExistingClaim.model_validate(item) for item in data.get("initial_memory", [])]
    return InMemoryRepository(claims)


# ----------------------------------------------------------------------------- judges


class _MemoJudge:
    """Reuse one Judge answer per claim and evidence set within a run (paired comparison)."""

    def __init__(self, inner: Judge) -> None:
        self._inner = inner
        self._seen: dict[str, JudgeAssessment] = {}

    async def evaluate(self, request: JudgeRequest) -> JudgeAssessment:
        key = request.model_dump_json()
        if key not in self._seen:
            self._seen[key] = await self._inner.evaluate(request)
        return self._seen[key]

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()


def make_judge(choice: str) -> Judge:
    """Build the Judge named by `choice` (one of `JUDGE_CHOICES`). No silent fallback."""

    if choice == "fake":
        return FakeJudge()
    if choice == "fixture":
        return FixtureJudge()
    if choice == "env":
        return build_judge_from_env()
    raise ValueError(f"unknown judge {choice!r}; expected one of {JUDGE_CHOICES}")


def live_provider_configured() -> bool:
    """True when `SEKF_PROVIDER` selects a real model and a key is present."""

    provider = (os.getenv("SEKF_PROVIDER") or "fake").strip().lower()
    return provider in {"openai", "openai_compatible"} and bool(os.getenv("SEKF_LLM_API_KEY"))


class SystemNotRunnable(Exception):
    """A system cannot run in this configuration (for example `single_judge` offline)."""


class RunContext:
    """Everything one run shares across cases: the Judge choice and its cached answers."""

    def __init__(self, judge: str = "fake") -> None:
        if judge not in JUDGE_CHOICES:
            raise ValueError(f"unknown judge {judge!r}; expected one of {JUDGE_CHOICES}")
        self.judge_choice = judge
        self.judge = _MemoJudge(make_judge(judge))
        self._provider: Any = None

    def single_judge_reason(self) -> str | None:
        """Why `single_judge` cannot run here, or None when it can."""

        if self.judge_choice != "env":
            return f"single_judge needs a live model; this run uses judge={self.judge_choice!r}"
        if not live_provider_configured():
            return "needs SEKF_PROVIDER=openai and SEKF_LLM_API_KEY; no substitute model is used"
        return None

    def provider(self) -> Any:
        """The shared live provider for `single_judge`, created on first use."""

        if self._provider is None:
            self._provider = create_provider_from_env()
        return self._provider

    async def aclose(self) -> None:
        """Close HTTP clients held by the Judge or the provider."""

        await self.judge.aclose()
        close = getattr(self._provider, "aclose", None)
        if close is not None:
            await close()


# ----------------------------------------------------------------------------- systems


def _council(
    data: dict[str, Any],
    judge: Judge,
    *,
    historian: Historian | None = None,
    synthesizer: Synthesizer | None = None,
) -> Callable[[DecideRequest], Awaitable[DecideResponse]]:
    governance = Governance(
        memory=_memory(data),
        historian=historian or DeterministicHistorian(),
        judge=judge,
        synthesizer=synthesizer or DeterministicSynthesizer(),
        policy=PolicyEngine(),
    )
    return governance.decide


def _system(name: str, data: dict[str, Any], ctx: RunContext) -> Callable[..., Any]:
    judge = ctx.judge
    if name in _STORAGE:
        return _STORAGE[name](_memory(data)).decide
    if name == "generic_ensemble":
        return GenericEnsembleBaseline(judge, memory=_memory(data)).decide
    if name == "single_judge":
        reason = ctx.single_judge_reason()
        if reason is not None:
            raise SystemNotRunnable(reason)
        return SingleJudgeBaseline(ctx.provider(), memory=_memory(data)).decide
    if name == COUNCIL:
        return _council(data, judge)
    if name == "ablation_no_historian":
        return _council(data, judge, historian=NeutralHistorian())
    if name == "ablation_no_synthesizer":
        return _council(data, judge, synthesizer=DisabledSynthesizer())
    if name == "ablation_no_provenance":
        return _council(
            data,
            judge,
            historian=AblatedHistorian.from_name(DeterministicHistorian(), "provenance"),
        )
    if name == "ablation_no_temporal":
        return _council(
            data, judge, historian=AblatedHistorian.from_name(DeterministicHistorian(), "temporal")
        )
    raise ValueError(f"unknown system {name!r}; expected one of {ALL_SYSTEMS}")


def check_systems(systems: Sequence[str]) -> None:
    """Reject unknown names, and a derived system whose source system is not selected."""

    for name in systems:
        if name not in ALL_SYSTEMS:
            raise ValueError(f"unknown system {name!r}; expected one of {ALL_SYSTEMS}")
        source = DERIVED.get(name)
        if source is not None and source not in systems:
            raise ValueError(f"{name} is derived from {source}; select {source} too")


def _gold_fields(data: dict[str, Any]) -> dict[str, Any]:
    gold = data["gold"]
    return {
        "case_id": case_id(data),
        "gold_action": str(gold["final_action"]),
        "gold_judge": gold.get("judge_label"),
        "gold_conflict": gold.get("conflict_type"),
        "retrieval_recall": _recall(data),
    }


async def run_case(system: str, data: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    """Run one system on one case in dry-run mode and return its result row.

    Raises `SystemNotRunnable` when the system cannot run here (never a substitute).
    """

    if system in DERIVED:
        raise ValueError(f"{system} is derived from {DERIVED[system]}; it is not run")
    decide = _system(system, data, ctx)
    started = time.perf_counter()
    response: DecideResponse = await decide(_request(data))
    latency_ms = int((time.perf_counter() - started) * 1000)
    cost = response.decision.cost
    return {
        **_gold_fields(data),
        "system": system,
        "predicted_action": response.decision.final_action.value,
        "rule_ids_fired": list(response.decision.rule_ids_fired),
        "judge_label": response.judge.label.value if response.judge else None,
        "synthesizer_action": no_policy_action(response).value if system == COUNCIL else None,
        "latency_ms": latency_ms,
        "tokens": cost.input_tokens + cost.output_tokens,
        "llm_calls": cost.llm_calls,
        "cost": cost.estimated_cost_usd,
        "role_errors": [error.role for error in response.role_errors],
    }


def derive_no_policy(council_row: dict[str, Any]) -> dict[str, Any]:
    """The no-policy ablation row: the council's own Synthesizer action, nothing re-run."""

    return {
        **council_row,
        "system": NO_POLICY,
        "predicted_action": council_row["synthesizer_action"],
        "rule_ids_fired": ["ABLATION_NO_POLICY"],
    }


def _recall(data: dict[str, Any]) -> float | None:
    candidate = CandidateClaim.model_validate(data["candidate"])
    claims = [ExistingClaim.model_validate(item) for item in data.get("initial_memory", [])]
    gold = data["gold"]
    delta = gold.get("expected_memory_delta") or {}
    relevant = [str(item) for item in delta.get("superseded_claim_ids", [])]
    if not relevant:
        relevant = [
            str(item["claim_id"])
            for item in data.get("initial_memory", [])
            if item.get("predicate_key") == data["candidate"]["predicate_key"]
        ]
    retrieved = [str(claim.claim_id) for claim in hybrid_search(candidate, claims, k=5)]
    return retrieval_recall_at_k(retrieved, relevant)


# ----------------------------------------------------------------------------- summary


def summarize(name: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """SPEC.md section 14 metrics for one system's rows, with a 95% bootstrap interval."""

    gold = [row["gold_action"] for row in rows]
    pred = [row["predicted_action"] for row in rows]
    precision, recall = promotion_scores(gold, pred)
    interval = paired_bootstrap_interval(gold, pred, macro_f1, n_resamples=1000, seed=0)
    false_rate, false_hits, false_n = false_promotion_rate(
        gold,
        pred,
        gold_judge=[row.get("gold_judge") or "" for row in rows],
        gold_conflict=[row.get("gold_conflict") or "NONE" for row in rows],
    )
    recalls = [row["retrieval_recall"] for row in rows if row["retrieval_recall"] is not None]
    return {
        "system": name,
        "n": len(rows),
        "macro_f1": interval.point,
        "macro_f1_ci95": [interval.low, interval.high],
        "accuracy": (sum(g == p for g, p in zip(gold, pred, strict=True)) / len(rows))
        if rows
        else 0.0,
        "promotion_precision": precision,
        "promotion_recall": recall,
        "revision_accuracy": revision_accuracy(gold, pred),
        "unnecessary_quarantine_rate": unnecessary_quarantine_rate(gold, pred),
        "false_promotion_rate": false_rate,
        "false_promotion_hits": false_hits,
        "false_promotion_support": false_n,
        "retrieval_recall_at_5": (sum(recalls) / len(recalls)) if recalls else None,
        "p95_latency_ms": percentile([float(row["latency_ms"]) for row in rows], 95),
        "tokens": sum(row["tokens"] for row in rows),
        "llm_calls": sum(row["llm_calls"] for row in rows),
        "estimated_cost_usd": sum(row["cost"] for row in rows),
        "role_error_rows": sum(1 for row in rows if row.get("role_errors")),
    }


def compare_to_council(
    rows_by_system: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Paired bootstrap of the macro-F1 difference (council minus system) on shared cases."""

    council = {row["case_id"]: row for row in rows_by_system.get(COUNCIL, [])}
    out: list[dict[str, Any]] = []
    for name, rows in rows_by_system.items():
        if name == COUNCIL or not council:
            continue
        shared = [row for row in rows if row["case_id"] in council]
        if not shared:
            continue
        gold = [row["gold_action"] for row in shared]
        ours = [council[row["case_id"]]["predicted_action"] for row in shared]
        theirs = [row["predicted_action"] for row in shared]
        diff = paired_difference_interval(gold, ours, theirs, macro_f1, n_resamples=1000, seed=0)
        out.append(
            {
                "system": name,
                "n": len(shared),
                "macro_f1_difference": diff.point,
                "ci95": [diff.low, diff.high],
                "council_better": diff.low > 0,
            }
        )
    return out


def not_run(name: str, reason: str) -> dict[str, Any]:
    """Summary entry for a system that was not run, with the reason."""

    return {"system": name, "status": "not_run", "reason": reason}


async def run_dev_cases(
    cases: list[dict[str, Any]],
    *,
    judge: str = "fake",
    systems: Sequence[str] = DEFAULT_SYSTEMS,
) -> dict[str, Any]:
    """Score `cases` (which include their gold object) in memory, without a results file."""

    check_systems(systems)
    ctx = RunContext(judge)
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in systems}
    skipped: dict[str, str] = {}
    try:
        for data in cases:
            for name in systems:
                if name in DERIVED or name in skipped:
                    continue
                try:
                    rows[name].append(await run_case(name, data, ctx))
                except SystemNotRunnable as exc:
                    skipped[name] = str(exc)
    finally:
        await ctx.aclose()
    for name, source in DERIVED.items():
        if name in rows:
            rows[name] = [derive_no_policy(row) for row in rows.get(source, [])]
    return build_summary(rows, skipped, n_cases=len(cases), judge=judge)


def build_summary(
    rows: dict[str, list[dict[str, Any]]],
    skipped: dict[str, str],
    *,
    n_cases: int,
    judge: str,
) -> dict[str, Any]:
    """Assemble the summary object from per-system rows and the systems that could not run."""

    ran = {name: r for name, r in rows.items() if name not in skipped}
    summaries = [summarize(name, r) for name, r in ran.items()]
    summaries += [not_run(name, reason) for name, reason in skipped.items()]
    return {
        "n_cases": n_cases,
        "judge": judge,
        "systems": summaries,
        "council_vs_system": compare_to_council(ran),
    }


async def start_offline_run(app: Any, *, limit: int = 0) -> dict[str, Any]:
    """Run the dev split with the fake Judge, store the summary on the app, and return it."""

    if limit < 0:
        raise ValueError("limit must be >= 0")
    cases = load_dev_cases()
    if limit > 0:
        cases = cases[:limit]
    summary = await run_dev_cases(cases)
    run_id = str(uuid4())
    summary["run_id"] = run_id
    runs = getattr(app.state, "experiments", None)
    if runs is None:
        runs = {}
        app.state.experiments = runs
    runs[run_id] = summary
    return summary
