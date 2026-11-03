"""Live smoke test of the LLM-backed council, with a spending limit.

Needs a real key. Run once after setting `SEKF_LLM_API_KEY` (and `SEKF_PROVIDER=openai`):

    .venv/bin/python scripts/smoke_llm.py --max-cost 0.05 --max-calls 30

It sends the five `datasets/dev/shailen` cases through the model-backed Historian, Judge, and
Synthesizer in dry-run mode and prints, per case, the gold action, the action the system chose,
and the calls and money spent. This is a wiring and cost check, not an evaluation: a
disagreement with the gold label is information, not a failure.

All three roles are LLM-backed here regardless of `SEKF_HISTORIAN` / `SEKF_SYNTHESIZER` in
`.env`, and all share one provider so the limit is global.

How the limit works, exactly: before every provider call, the run checks the totals so far and
refuses the call once either `--max-calls` or `--max-cost` has been reached. A single request
cannot be interrupted once sent, and the provider may repair invalid JSON up to twice, so the
total can overshoot by at most the 3 calls of that last request. The dollar limit needs a
price for the model; for an unpriced model the cost reads $0 and only the call limit is
effective (the script says so). The run never writes to a real database. It exits non-zero if
a role failed for any reason other than reaching the limit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel

from sekf.api.orchestrator import Governance
from sekf.contracts import DecideMode, DecideRequest, ExistingClaim
from sekf.errors import CallUsage, ProviderUnavailable, SekfError
from sekf.historian import LLMHistorian
from sekf.judge import LLMJudge
from sekf.judge.compose import load_project_env
from sekf.memory import SqlMemoryRepository
from sekf.policy import PolicyEngine
from sekf.providers import LLMResult, OpenAICompatibleProvider, StructuredLLMProvider
from sekf.synthesizer import LLMSynthesizer

CASES_DIR = Path(__file__).resolve().parent.parent / "datasets" / "dev" / "shailen"
BUDGET_MARKER = "smoke budget reached"


class BudgetedProvider:
    """Wraps a provider and refuses further calls once the call or dollar limit is reached."""

    def __init__(self, inner: StructuredLLMProvider, *, max_cost: float, max_calls: int) -> None:
        self._inner = inner
        self.max_cost = max_cost
        self.max_calls = max_calls
        self.calls = 0
        self.cost = 0.0

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.max_calls or (self.cost > 0 and self.cost >= self.max_cost)

    def _add(self, usage: CallUsage) -> None:
        self.calls += usage.llm_calls
        self.cost += usage.estimated_cost_usd

    async def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        model: str,
        temperature: float,
        seed: int | None,
    ) -> LLMResult:
        if self.exhausted:
            raise ProviderUnavailable(
                f"{BUDGET_MARKER} ({self.calls} calls, ${self.cost:.4f}); call refused"
            )
        try:
            result = await self._inner.complete_json(
                messages=messages,
                response_model=response_model,
                model=model,
                temperature=temperature,
                seed=seed,
            )
        except SekfError as exc:
            if isinstance(exc.usage, CallUsage):
                self._add(exc.usage)
            raise
        self._add(
            CallUsage(llm_calls=result.llm_calls, estimated_cost_usd=result.estimated_cost_usd)
        )
        return result

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()


def _live_governance(max_cost: float, max_calls: int) -> tuple[Governance, BudgetedProvider]:
    """Build the LLM-backed council. Refuses to run without a live provider and a key."""
    load_project_env()
    if (os.getenv("SEKF_PROVIDER") or "fake").strip().lower() not in {
        "openai",
        "openai_compatible",
    }:
        raise SystemExit(
            "set SEKF_PROVIDER=openai (and SEKF_LLM_API_KEY) to run the live smoke test"
        )
    if not os.getenv("SEKF_LLM_API_KEY"):
        raise SystemExit("SEKF_LLM_API_KEY is not set; nothing was sent")
    live = OpenAICompatibleProvider.from_env()
    model = os.getenv("SEKF_LLM_MODEL") or "gpt-4o-mini"
    if live.rates_for(model) == (0.0, 0.0):
        print(
            f"warning: no price known for model {model!r}; the dollar limit is inactive "
            "and only --max-calls limits the run"
        )
    budget = BudgetedProvider(live, max_cost=max_cost, max_calls=max_calls)
    governance = Governance(
        memory=SqlMemoryRepository.from_url("sqlite://"),  # throwaway, in RAM
        historian=LLMHistorian(budget),
        judge=LLMJudge(budget),
        synthesizer=LLMSynthesizer(budget),
        policy=PolicyEngine(),
    )
    return governance, budget


async def _run(
    governance: Governance, max_cost: float, max_calls: int, budget: BudgetedProvider | None
) -> int:
    spent = 0.0
    calls = 0
    failures = 0
    print(f"{'case':34} {'gold':20} {'chosen':20} {'calls':>5} {'usd':>8}")
    try:
        for path in sorted(CASES_DIR.glob("*.json")):
            stop = (
                (budget.exhausted if budget else False)
                or spent >= max_cost > 0
                or calls >= max_calls
            )
            if stop:
                print(f"-- limit reached (spent ${spent:.4f}, {calls} calls); stopping")
                break
            case = json.loads(path.read_text())
            memory = governance._memory
            if isinstance(memory, SqlMemoryRepository):
                memory.seed_claims(
                    [ExistingClaim.model_validate(c) for c in case["initial_memory"]]
                )
            request = DecideRequest.model_validate(
                {
                    "candidate": case["candidate"],
                    "evidence": case["evidence"],
                    "mode": DecideMode.DRY_RUN,
                }
            )
            response = await governance.decide(request)
            cost = response.decision.cost
            spent += cost.estimated_cost_usd
            calls += cost.llm_calls
            print(
                f"{path.stem:34} {case['gold']['final_action']:20} "
                f"{response.decision.final_action.value:20} {cost.llm_calls:>5} "
                f"{cost.estimated_cost_usd:>8.4f}"
            )
            for err in response.role_errors:
                over_budget = BUDGET_MARKER in err.message
                failures += 0 if over_budget else 1
                print(f"   ! {err.role}: {err.error_type}: {err.message}")
    finally:
        await governance.aclose()
    print(f"total: {calls} calls, ${spent:.4f}; role failures: {failures}")
    if spent == 0.0 and calls:
        print("note: cost is $0 because the model has no price table; calls are still counted")
    return 1 if failures else 0


def main(argv: list[str] | None = None, governance: Governance | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-cost", type=float, default=0.05, help="stop at this many USD")
    parser.add_argument("--max-calls", type=int, default=30, help="stop at this many LLM calls")
    args = parser.parse_args(argv)
    budget = None
    if governance is None:
        governance, budget = _live_governance(args.max_cost, args.max_calls)
    return asyncio.run(_run(governance, args.max_cost, args.max_calls, budget))


if __name__ == "__main__":
    sys.exit(main())
