# Self-Evolving Knowledge Framework for Autonomous AI Agents

## Engineering and Research Specification

**Project acronym:** SEKF
**Team:** Krishna Panjiyar, Mahalakshmi Rajkumar, Sai Chetan Muppalla, Shailen Sutradhar
**Advisor:** Professor Andrew Bond
**Course:** CMPE 295A (Fall 2026), continuing in 295B
**Prototype deadline:** Dec 8, 2026 (end of Sprint 4 in this spec = Sprint 5 in the workbook); final demonstration Dec 9-11
**Specification status:** v1.1

---

## 0. How to Use This Document

- **Humans:** read sections 1-5 and 9 first, then your own module section.
- **AI coding agents:** read `AGENTS.md` first (short), then only the sections of this spec that your task names. The Pydantic models in `src/sekf/contracts/` define the implemented interface. If the implementation and this spec disagree, do not silently pick one: report the mismatch and update the correct artifact (code or spec) through a normal reviewed PR.
- **Understanding the domain is expected to grow while building.** Section 4 contains a worked example. Use it to orient yourself before touching code.
- Open decisions are tracked in section 21. Do not silently resolve them in code.

---

## 1. Purpose

SEKF is a lightweight, independently runnable research prototype that governs whether candidate claims derived from an AI agent's experiences should enter or modify its long-term knowledge.

The system must not trust every extracted claim. It must preserve raw experience, inspect provenance and time, evaluate supplied evidence, identify conflicts, produce an auditable recommendation, and apply deterministic rules before changing validated knowledge.

Primary research question:

> Does a role-specialized Historian-Judge-Synthesizer council improve long-term memory-governance decisions compared with direct storage, latest-write-wins, static retrieval, a single-model judge, and a non-specialized multi-agent ensemble?

The project is not an attempt to reproduce Atlas, train a foundation model, prove objective truth, or build a complete autonomous agent platform.

---

## 2. Research Questions and Hypotheses

### 2.1 Research questions

- **RQ1:** Does the council improve memory-action accuracy over direct storage, latest-write-wins, and a single-model judge?
- **RQ2:** Does role specialization outperform a non-specialized three-judge ensemble when model, evidence, temperature, and total token budget are controlled?
- **RQ3:** Which components contribute most: temporal/provenance analysis (Historian), evidence judgment (Judge), the Synthesizer veto, or the deterministic policy?
- **RQ4:** What latency, model-call, token, and monetary-cost trade-offs result from council review?

### 2.2 Hypotheses

- **H1:** The full system achieves higher promotion precision and revision accuracy than direct storage and latest-write-wins.
- **H2:** The council achieves higher final-action macro-F1 than a single-model judge under a comparable total token budget.
- **H3:** Removing the Historian reduces accuracy most on temporal-validity, source-conflict, and supersession cases.
- **H4:** The Synthesizer veto reduces incorrect promotions/revisions at a measurable cost in unnecessary quarantines.
- **H5:** The council costs more than simpler baselines; the reduction in incorrect promotions is reported against that cost (no a-priori claim that it is justified).

The project is successful even if hypotheses are rejected, provided the experiment is fair, reproducible, and explained.

---

## 3. Scope

### 3.1 Required scope

1. A persistent store for episodes, candidate claims, validated claims, evidence, relationships, role outputs, and decisions.
2. A Historian for provenance, temporal validity, claim history, and conflict analysis.
3. A Judge for evidence support, refutation, insufficiency, and confidence.
4. A Synthesizer that combines Historian and Judge outputs into an advisory recommendation.
5. A deterministic policy engine that makes the final memory action.
6. An orchestration API that runs the proposed system and all baselines through a common interface.
7. A versioned, labeled evaluation dataset and experiment runner.
8. Reproducible metrics for correctness, robustness, latency, token usage, and cost.

### 3.2 Non-goals for the first semester

- Reimplementing Atlas or depending on Atlas at runtime
- NATS, Kubernetes, SLURM, HPC, or NRP Nautilus deployment (see D2 in section 21)
- Training or fine-tuning foundation models
- A production-scale graph database
- Autonomous source-code modification
- Web-scale crawling or live-web fact checking
- Proving objective truth beyond supplied evidence and labels
- A polished end-user chat application
- Self-RAG reproduction (optional stretch; see section 12)

### 3.3 Optional second-semester extensions

- Sequential claim-stream evaluation (memory evolves across many claims; error accumulation, retention, contradiction rate over time)
- Atlas/NATS adapter
- Human-review interface
- Local open models
- Adversarial memory-poisoning evaluation at scale
- Larger public benchmark integration
- Claim extraction from long multi-turn conversations

---

## 4. Worked Example (read this first)

Memory currently holds an **ACTIVE** claim: *"The lab closes at 17:00 on Fridays."* (source: `old-schedule`, valid from 2026-01-01).

A new observation arrives: *"The lab closes at 18:00 on Fridays,"* observed 2026-10-01 from `student-chat`, plus one piece of evidence: an official schedule published 2026-09-28 saying *"Beginning October 1, Friday closing time is 6 PM"* (reliability 0.95).

What each part does:

| Step | Component | Output for this example |
|---|---|---|
| 1 | Memory | Finds the related ACTIVE claim (same `subject_key` + `predicate_key`) |
| 2 | Historian | `temporal_relation=NEWER_UPDATE`, `conflict_type=TEMPORAL_UPDATE`, proposes superseding the 17:00 claim; high provenance quality |
| 3 | Judge | `label=SUPPORTED` (official schedule supports 18:00 from Oct 1), cites the evidence ID |
| 4 | Synthesizer | Recommends `REVISE`, confidence 0.9, no unresolved questions |
| 5 | Policy | Rule `R_REVISE` fires; Synthesizer does not veto; final action `REVISE` |
| 6 | Memory | 17:00 claim becomes `SUPERSEDED` (kept, never deleted); 18:00 claim becomes `ACTIVE` v2 with a `SUPERSEDES` link; audit trace stored |

Variant: if the only evidence were an unofficial chat message (reliability 0.3) the Judge may still say `SUPPORTED`, but the source reliability is below threshold, so the policy fires `R_QUARANTINE` and memory is not changed. This is the core idea of the project: **a newer claim is not automatically a true claim.**

---

## 5. Core Scientific Distinction

The primary experiment begins with **atomic candidate claims** supplied by the evaluation dataset. This isolates memory governance from claim extraction.

1. **Primary governance track:** input is an already-atomic claim. Conclusions about council effectiveness must come from this track.
2. **Secondary end-to-end track:** input is a conversation/document from which claims are extracted. Extraction errors are reported separately and never counted as governance errors.

The system evaluates whether a claim is supported by the *supplied evidence*. It does not guarantee real-world truth.

**Known limitation (state it in the report):** in the primary track `subject_key` and `predicate_key` are supplied, so related-claim lookup is an exact key match. This removes the hard retrieval/canonicalization problem on purpose.

---

## 6. Repository, Branch, and Ownership Model

One shared monorepo (the `Atlas-AI` repository). The implementation is a **modular monolith**: one FastAPI application with independently testable modules. Modules call each other through typed Python interfaces, not HTTP.

### 6.1 Repository layout

```text
.
├── SPEC.md
├── AGENTS.md                   # instructions for any AI coding agent
├── CLAUDE.md                   # Claude Code entry point (imports AGENTS.md)
├── README.md
├── Makefile                    # setup / test / lint / typecheck / run / contracts
├── pyproject.toml
├── .env.example
├── .github/
│   └── workflows/ci.yml
├── contracts/v1/               # JSON Schemas GENERATED from Pydantic (do not hand-edit)
├── prompts/                    # Versioned role prompts
├── datasets/                   # Versioned dev/test cases + annotation guide
├── experiments/                # Manifests; generated results are git-ignored
├── docs/decisions/             # Architecture decision records (ADRs)
├── meetings/
├── src/sekf/
│   ├── api/
│   ├── contracts/              # Pydantic models: THE canonical contracts
│   ├── memory/
│   ├── historian/
│   ├── judge/
│   ├── synthesizer/
│   ├── policy/
│   ├── providers/
│   ├── baselines/
│   └── evaluation/
└── tests/{unit,contract,integration,fixtures}/
```

### 6.2 Module ownership

| Module or path | Primary owner | Responsibility |
|---|---|---|
| `memory/`, migrations, `baselines/direct_storage`, `baselines/latest_write_wins`, `baselines/static_rag` | Krishna Panjiyar | Persistence, state transitions, claim graph, related-claim search, storage and retrieval baselines |
| `historian/`, `synthesizer/`, temporal/provenance cases, annotation guide | Mahalakshmi Rajkumar | Provenance, temporal validity, conflicts, advisory recommendation, temporal/provenance dataset |
| `judge/`, `providers/`, `baselines/single_judge`, `baselines/generic_ensemble`, evidence cases | Sai Chetan Muppalla | Evidence assessment, LLM provider adapters, judge baselines, evidence dataset |
| `policy/`, `api/`, `evaluation/`, CI, Docker, `contracts/` skeleton | Shailen Sutradhar | Deterministic policy, orchestration, public API, experiment harness, metrics, release |
| `src/sekf/contracts/`, `Makefile`, `pyproject.toml`, integration tests | Whole team | Shared interfaces; changes go through the contract-change process |

**Load balance note:** static RAG moved to Krishna (it is retrieval over the shared evidence store and pairs naturally with the memory work). Docker is a Sprint 4 task, not Sprint 0.

### 6.3 Branch workflow (trunk-based)

`main` is protected and always builds and passes tests.

1. Create a short-lived branch from the latest `main`: `<name>/<short-task>` (for example `krishna/memory-schema`).
2. Keep branches to roughly 1-3 days of work. Open a **draft PR early**, mark ready when tests pass.
3. Rebase or merge `main` into your branch at least daily (`git fetch && git rebase origin/main`).
4. Every PR needs one approval from someone other than its author, plus passing CI.
5. PRs that change `src/sekf/contracts/`, shared enums, DB migrations, or policy actions require approval from **every** affected module owner and an ADR.
6. Squash-merge. The PR description names the section 18 task it covers.
7. No force-push to `main`. Delete merged branches.
8. Resolve conflicts with the affected owner; never discard another member's changes to make a merge pass.
9. Conflict hotspots are `pyproject.toml`, `Makefile`, `src/sekf/contracts/`, and `src/sekf/api/` router registration. Change them only in small, fast PRs, and announce in the team chat before and after merging.
10. We do not use GitHub issues. The Sprint backlog in section 18 is the task list; claim a bullet by writing your name next to it in a PR or the team chat so two people do not build the same thing.

Branch protection may require a paid plan for private repositories. If unavailable, enforce the same rules by convention and make CI mandatory by team agreement.

### 6.4 Review pairing

| Author | Required reviewer |
|---|---|
| Krishna | Mahalakshmi |
| Mahalakshmi | Krishna |
| Sai Chetan | Shailen |
| Shailen | Sai Chetan |

At least two people must be able to run and debug the whole application. Experiment execution must never be left to one person in the final week.

### 6.5 Canonical contracts

Pydantic models in `src/sekf/contracts/` are canonical. `contracts/v1/*.json` is generated by `make contracts`; CI fails if generated files are stale. Example payloads live in `tests/fixtures/`.

A contract change requires: an ADR in `docs/decisions/`; updated models, regenerated schemas, and updated fixtures; passing unit, contract, and integration tests; approval from every affected owner; a version bump if backward compatibility breaks.

---

## 7. Technology Constraints

- Python 3.12, FastAPI, Pydantic v2
- SQLAlchemy 2; Alembic for migrations (introduced in Sprint 1; `create_all` is acceptable in Sprint 0)
- SQLite for the prototype
- `httpx` for provider calls
- `pytest`, `pytest-asyncio`, coverage
- Ruff (format + lint), and `mypy` or `pyright`
- `python -m venv` + `pip` for environments (`make setup`); `constraints.txt` (from `make lock`) pins exact versions for reproducible installs
- Docker in Sprint 4
- One provider-neutral LLM adapter for OpenAI-compatible JSON responses (the concrete provider is open decision D1)

No LangChain, graph database, GPU, or hosted vector database in the core. Add a library only for a demonstrated need, recorded in an ADR if it is a dependency of more than one module.

Every LLM-dependent component must have a deterministic **fake provider** so CI and tests run offline with no credentials.

---

## 8. System Architecture

```text
Atomic candidate claim + evidence + current memory state
                         |
                         v
                 Governance API
                  /           \
                 v             v
          Historian Module   Judge Module        (run concurrently)
                 \             /
                  v           v
              Synthesizer Module                 (advisory)
                        |
                        v
             Deterministic Policy Engine         (sole authority)
                        |
                        v
                  Memory Module                  (transactional)
                        |
                        v
           Decision + updated state + audit trace
```

Only the policy engine may authorize a memory mutation; the memory module applies it transactionally. The Synthesizer cannot write memory and cannot upgrade an outcome (see the veto rule in 9.4).

---

## 9. Contracts, State, and Policy

All IDs are UUID strings. All timestamps are UTC RFC 3339. Unknown fields are rejected in v1 models.

### 9.1 Data contracts

Field-level definitions live in `src/sekf/contracts/`. Required objects and their key fields:

- **CandidateClaim:** `claim_id`, `claim_text`, `subject_key`, `predicate_key`, `object_text`, `source_id`, `observed_at`, `valid_from`, `valid_to`, `extractor_confidence`, `metadata`.
- **EvidenceRecord:** `evidence_id`, `content`, `source_id`, `source_type`, `published_at`, `retrieved_at`, `reliability` (0-1), `content_hash`, `metadata`.
- **ExistingClaim:** `claim_id`, `claim_text`, `subject_key`, `predicate_key`, `object_text`, `status`, `version`, `valid_from`, `valid_to`, `confidence`, `created_at`, `updated_at`, `supersedes_claim_id`, `evidence_ids`.
- **HistorianAssessment:** `assessment_id`, `candidate_claim_id`, `related_claim_ids`, `provenance_quality` (HIGH/MEDIUM/LOW/UNKNOWN), `temporal_relation`, `conflict_type`, `supersedes_claim_id`, `source_reliability_score`, `confidence`, `rationale`, plus call metadata.
- **JudgeAssessment:** `assessment_id`, `candidate_claim_id`, `label`, `confidence`, `supporting_evidence_ids`, `refuting_evidence_ids`, `missing_information`, `rationale`, plus call metadata.
- **SynthesizerRecommendation:** `recommendation_id`, `candidate_claim_id`, `recommended_action`, `confidence`, `historian_assessment_id`, `judge_assessment_id`, `rationale`, `unresolved_questions`, plus call metadata.
- **DecisionRecord:** `decision_id`, `candidate_claim_id`, `final_action`, `policy_version`, `thresholds`, `rule_ids_fired`, `previous_claim_ids`, `resulting_claim_id`, assessment/recommendation IDs, `explanation`, `created_at`, cost and latency summary.

"Call metadata" = `model`, `provider`, `prompt_version`, `prompt_hash`, `temperature`, `seed`, `llm_calls`, `input_tokens`, `output_tokens`, `latency_ms`, `estimated_cost_usd`. `provider`, `prompt_hash`, `temperature`, and `llm_calls` are required (a role that omits them fails validation); `seed` may be null. `llm_calls` is the number of real provider calls including repair attempts, copied from the provider result; deterministic or fake roles report 0. Cost-normalized comparisons use this field.

Enums:

```text
temporal_relation: NO_RELATED_CLAIM SAME_PERIOD OLDER_INFORMATION NEWER_UPDATE
                   OVERLAPPING_VALIDITY NON_OVERLAPPING_VALIDITY UNKNOWN
conflict_type:     NONE DUPLICATE PARAPHRASE VALUE_CONFLICT TEMPORAL_UPDATE
                   SOURCE_CONFLICT UNKNOWN
judge label:       SUPPORTED REFUTED INSUFFICIENT_EVIDENCE MIXED_EVIDENCE INVALID_INPUT
claim status:      CANDIDATE ACTIVE QUARANTINED SUPERSEDED REJECTED
relationship:      SUPPORTS CONTRADICTS SUPERSEDES DERIVED_FROM VALID_DURING DUPLICATES
final action:      PROMOTE ATTACH_EVIDENCE KEEP_EPISODIC QUARANTINE REVISE REJECT
                   ESCALATE_FOR_REVIEW
```

Vocabulary note: the workbook names the actions promote, retain as episodic, quarantine, conflict, revise, reject. Here "conflict" is not a separate action: a conflicted claim is `QUARANTINE`d and the conflict is recorded in the Historian's `conflict_type` and a `CONTRADICTS` relationship. `ATTACH_EVIDENCE` and `ESCALATE_FOR_REVIEW` are additions. Use these exact names; do not invent alternatives.

Action semantics:

| Action | Effect on memory |
|---|---|
| `PROMOTE` | New claim becomes `ACTIVE` |
| `ATTACH_EVIDENCE` | Duplicate/paraphrase of an ACTIVE claim: add evidence link, no new version |
| `KEEP_EPISODIC` | Stored as episode/candidate only; not validated knowledge |
| `QUARANTINE` | Claim stored with status `QUARANTINED`; ACTIVE knowledge unchanged |
| `REVISE` | Old claim `SUPERSEDED` (kept), new claim `ACTIVE` with `SUPERSEDES` link |
| `REJECT` | Claim stored with status `REJECTED` (kept for audit) |
| `ESCALATE_FOR_REVIEW` | Claim stays `CANDIDATE`, flagged in a review queue |

All mutations preserve previous versions. Nothing is ever deleted.

### 9.2 Roles

- **Historian** decides provenance, time, history, and conflict. It must not decide evidence entailment.
- **Judge** decides evidence support only. It must not see the expected label, the memory state's final action, or the Historian's output.
- **Synthesizer** summarizes both and recommends an action. Advisory only.

### 9.3 Default thresholds (configurable, recorded with every decision)

```text
support_threshold                = 0.80
refute_threshold                 = 0.80
historian_confidence_threshold   = 0.70
source_reliability_threshold     = 0.60
synthesizer_veto_threshold       = 0.80
```

### 9.4 Deterministic policy v1

Step A computes a `rule_action` using rules in priority order (first match wins):

| ID | Condition | Action |
|---|---|---|
| R1 | Any role output invalid / schema-noncompliant / judge `INVALID_INPUT` | `ESCALATE_FOR_REVIEW` |
| R2 | Judge `REFUTED`, confidence >= refute_threshold | `REJECT` |
| R3 | Judge `INSUFFICIENT_EVIDENCE` | `KEEP_EPISODIC` |
| R4 | Judge `MIXED_EVIDENCE`, or `REFUTED` below threshold | `QUARANTINE` |
| R5 | Judge `SUPPORTED` below support_threshold | `KEEP_EPISODIC` |
| R6 | Historian confidence below threshold, or conflict_type `UNKNOWN` | `ESCALATE_FOR_REVIEW` |
| R7 | Supported duplicate/paraphrase of an ACTIVE claim | `ATTACH_EVIDENCE` |
| R8 | Supported newer update with a clearly identified superseded claim, Historian confidence and source reliability above thresholds | `REVISE` |
| R9 | Unresolved conflict (source conflict, value conflict without supersession, overlapping validity) or source reliability below threshold | `QUARANTINE` |
| R10 | Supported new claim, no unresolved conflict, adequate source reliability | `PROMOTE` |
| R11 | Anything else | `ESCALATE_FOR_REVIEW` |

Step B, **Synthesizer veto** (rule `R_SYN_VETO`): if `rule_action` is `PROMOTE`, `REVISE`, or `ATTACH_EVIDENCE`, and the Synthesizer recommends a different action with confidence >= synthesizer_veto_threshold, the final action becomes `QUARANTINE`. The Synthesizer can only make an outcome *more conservative*; it can never turn a non-mutating action into a mutating one. This gives the Synthesizer a measurable effect, so the "without Synthesizer" ablation is meaningful.

All fired rule IDs, the pre-veto `rule_action`, and the thresholds are stored in the DecisionRecord.

---

## 10. Module and API Interfaces

### 10.1 Memory

```python
class MemoryRepository(Protocol):
    def search_related_claims(self, claim: CandidateClaim) -> list[ExistingClaim]: ...
    def get_claim(self, claim_id: UUID) -> ExistingClaim | None: ...
    def get_claim_history(self, claim_id: UUID) -> list[ExistingClaim]: ...
    def get_bundle(self, decision_id: UUID) -> CommitBundle | None: ...  # answers replays
    def apply_decision(self, bundle: CommitBundle) -> MemoryDelta: ...
    def get_audit_trace(self, candidate_claim_id: UUID) -> AuditTrace: ...
```

Reads are side-effect free. `apply_decision` is the only write. It persists a `CommitBundle` (episode, candidate, evidence, role outputs, decision, and resulting state changes) in one transaction, all or nothing, idempotent by `decision_id`. Nothing is written before the policy has decided, so a crash leaves memory untouched and a retry is safe. The bundle also carries `memory_before` (the related claims the roles saw), `role_errors` (typed `RoleFailure` entries) and the `run_config`, so a committed escalation keeps the actual failure. An inconsistent bundle raises `CommitRejected` and writes nothing, and `CommitBundle` itself refuses to be built if its parts contradict each other. `decision_id` is derived from the stable request (full candidate, full evidence records, thresholds, policy version, the whole run configuration including `timeout_s`, optional `idempotency_key`; NOT the memory snapshot, which a committed decision itself changes), never from role outputs; `DecisionRecord.request_hash` stores that fingerprint. In commit mode the ID is computed before any role runs and a retry is answered from the stored bundle (no LLM calls, and the client sees the decision actually committed). Reusing an ID with a different `request_hash` is rejected, and a commit is rejected as stale if the current related-claim set differs in any way from `memory_before` (a claim changed, removed, or newly added); the caller retries with a fresh read. `apply_decision` recomputes `request_hash` and `decision_id` from the bundle rather than trusting them, and `get_audit_trace` returns one entry per decision with the bundle and the `MemoryDelta` it produced. Repositories store and return copies (claims as well as audit records), so callers cannot alter stored memory or audit history. To re-evaluate the same candidate against newer memory, or to force a deliberate re-run, pass a new `idempotency_key`. In `dry_run` mode no write happens. See `docs/decisions/0001-commit-bundle-and-call-metadata.md`.

### 10.2 Roles and policy

```python
class Historian(Protocol):
    async def analyze(self, request: HistorianRequest) -> HistorianAssessment: ...


class Judge(Protocol):
    async def evaluate(self, request: JudgeRequest) -> JudgeAssessment: ...


class Synthesizer(Protocol):
    async def synthesize(self, request: SynthesisRequest) -> SynthesizerRecommendation: ...


class DecisionPolicy(Protocol):
    def identify(
        self, context: PolicyContext
    ) -> tuple[UUID, str]: ...  # (decision_id, request_hash)
    def decide(self, context: PolicyContext) -> DecisionRecord: ...
```

The generic-ensemble baseline calls the same non-specialized Judge interface three times and aggregates labels by a documented rule (majority; ties go to `INSUFFICIENT_EVIDENCE`).

### 10.3 Public API

- `GET /health`
- `POST /v1/decide` (modes: `dry_run` default, `commit`)
- `POST /v1/baselines/{baseline_name}/decide`
- `POST /v1/experiments/run`
- `GET /v1/experiments/{run_id}`

Baseline names: `direct_storage`, `latest_write_wins`, `static_rag`, `single_judge`, `generic_ensemble`.

Every decision response includes: final action, role outputs, rules fired, memory changes, latency, token counts, and estimated cost.

---

## 11. LLM Provider Interface

```python
class StructuredLLMProvider(Protocol):
    async def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        model: str,
        temperature: float,
        seed: int | None,
    ) -> LLMResult: ...
```

Requirements: strict Pydantic validation; at most two repair attempts; failure recorded after repair exhaustion and surfaced as a typed error (policy rule R1 then escalates); concise rationales only; prompt version, model id, temperature, seed, tokens, latency, and cost recorded; API keys only from environment variables; deterministic fake provider for all tests. Prompts live in `prompts/` with versions and hashes.

---

## 12. Baselines and Ablations

### 12.1 Required baselines

1. **direct_storage:** promote every valid candidate.
2. **latest_write_wins:** replace the related ACTIVE claim whenever the candidate is newer.
3. **static_rag:** retrieve evidence and answer queries; never modifies governed memory. Reported separately where outputs are not comparable to memory actions.
4. **single_judge:** one model receives all context and recommends the final memory action. This is the closest analogue of LLM-managed memory systems such as Mem0 (ADD/UPDATE/DELETE/NOOP); cite those systems in the report as related work and describe this baseline as "Mem0-style".
5. **generic_ensemble:** three identical judge calls vote; no role specialization.

Direct storage and latest-write-wins use no LLM and are expected to be weak; the informative comparisons are single_judge, generic_ensemble, and the council.

Optional stretch: Self-RAG official implementation, or a clearly labeled adapted version (promised in the workbook; drop only with advisor agreement, see D2).

### 12.2 Required ablations

- Full system without Historian
- Full system without Synthesizer (no veto step)
- Full system without deterministic policy (Synthesizer action used directly; **dry-run only**, never writes to the canonical database)
- Full system without provenance fields
- Full system without temporal fields

### 12.3 Fairness controls

Same base model (unless model is the variable), same evidence, same candidate and memory state, temperature 0, recorded token limits, comparable total token budgets or explicit cost-normalized analysis, identical held-out cases.

---

## 13. Evaluation Dataset

### 13.1 Case schema

`case_id`, `category`, initial memory state, candidate claim, evidence records, **gold Judge label**, **gold Historian labels** (`temporal_relation`, `conflict_type`), **gold final action**, expected memory delta, annotation notes, split (`dev`/`test`), annotator IDs.

### 13.2 Label independence (important)

Gold final actions are assigned by annotators following the **annotation guide** (`datasets/ANNOTATION_GUIDE.md`, owned by Mahalakshmi), which defines each action in plain language ("should this belong in long-term knowledge, given this evidence and memory?"). Annotators must **not** derive gold actions by running or reading the policy rules. Role-level gold labels are scored separately from final actions, so role errors and policy errors can be separated.

### 13.3 Required categories

Supported new claim; unsupported claim; refuted claim; duplicate/paraphrase; reliable newer correction; unreliable newer contradiction; non-overlapping temporal validity; overlapping temporal conflict; conflicting sources with unequal reliability; conflicting sources with similar reliability; insufficient evidence; malformed input; prompt-injection / memory-poisoning attempt.

### 13.4 Targets and process

- Sprint 0: every member writes 5 cases from the template (20 total)
- Sprint 1: at least 24 integration fixtures
- Sprint 2: at least 60 labeled `dev` cases (pilot)
- Sprint 3: grow toward 120-200 reviewed cases; create the frozen `test` split
- Prompts and thresholds are tuned on `dev` only. The `test` split is frozen after the first official run.
- Every `test` case is labeled independently by two members; disagreements are adjudicated and logged.
- With about 60 cases, confidence intervals will be wide; say so explicitly.

---

## 14. Metrics

**Primary:** final-action macro-F1; promotion precision and recall; revision accuracy; contradiction-detection F1.

**Secondary:** provenance completeness; temporal-reasoning accuracy; unnecessary quarantine rate; escalation rate; Judge label accuracy; confidence calibration (Brier or ECE); a small downstream QA accuracy set.

**Efficiency:** median and p95 latency; LLM calls, tokens, and cost per case; storage growth per accepted claim.

**Analysis rules:** raw counts alongside percentages; 95% paired-bootstrap confidence intervals; per-case outputs preserved; consistency is not a synonym for correctness; errors are categorized as extraction, retrieval, role, policy, or storage errors.

---

## 15. Audit and Reproducibility

Every experiment run records: git commit SHA, contract version, dataset version and hash, model provider and exact id, prompt versions and hashes, policy version and thresholds, temperature and seed, start/end timestamps, per-case role outputs, rules fired, memory state before/after, tokens, latency, cost. No secrets or private conversations are committed.

---

## 16. Testing

Each owned module has: unit tests for deterministic logic; schema validation tests; error and timeout tests; fake-provider tests (no network); contract tests against canonical fixtures.

Integration tests: happy path; supported promotion; refuted rejection; insufficient-evidence retention; temporal revision; source-conflict quarantine; duplicate evidence attachment; malformed role output escalation; Synthesizer veto; idempotent decision replay; timeout and graceful failure. Plus a `/health` test and (Sprint 4) a Docker smoke test.

Target at least 80% line coverage for policy, storage, aggregation, and metric code.

---

## 17. Definition of Done

1. Acceptance criteria met.
2. Tests included and passing locally and in CI.
3. `make lint` and `make typecheck` pass.
4. Public interfaces have docstrings and an example.
5. Errors explicit; no silent fallback that changes scientific behavior.
6. Configuration documented in `.env.example`, no secrets.
7. Another member approved the PR.
8. Fixtures and docs updated.

---

## 18. Plan

Dates align with the submitted workbook sprints, but this spec adds a Sprint 0, so numbers differ: spec Sprint 0 = workbook Sprint 1, spec Sprint 1 = workbook Sprint 2, spec Sprint 2 = workbook Sprint 3, spec Sprint 3 = workbook Sprint 4, spec Sprint 4 = workbook Sprint 5, spec Sprint 5 = workbook Sprint 6. Use the spec numbering in code, PRs, and chat; use dates when talking to the advisor. Tasks below are the task list. Sprint 4 includes Thanksgiving week, so plan lighter work there.

### Sprint 0 — Walking skeleton (Oct 9-13; workbook Sprint 1)

**Goal:** one fixture traverses every module end to end offline with fake roles, so everyone then works against real, stable interfaces.

- **Shailen (critical path, target merged by Oct 11):** `pyproject.toml`, `Makefile`, CI; Pydantic contracts and enums; Protocol stubs for Memory, Historian, Judge, Synthesizer, Policy; fake provider; `/health`; `/v1/decide` calling stub modules; `make contracts`; one example fixture.
- **Krishna (no contract dependency yet):** draft the DB schema and an ADR for the claim/version/relationship model; spike SQLAlchemy models against it; align with Shailen once contracts land.
- **Mahalakshmi:** pure functions for interval overlap, temporal ordering, and provenance scoring (plain Python, no LLM, no contract imports); draft `ANNOTATION_GUIDE.md`.
- **Sai Chetan:** resolve D1 (LLM provider, budget, rate limits) and confirm working API access; draft the Judge prompt; write the provider retry/repair design.
- **Everyone:** write 5 dataset cases from the template; read the Sprint 0 reading list (Self-RAG, Generative Agents, FActScore, Mem0 paper or docs).

**Milestone:** skeleton on `main`, CI green, one fixture flows through all stubs; D1 decided; advisor confirmation sent (D2).

### Sprint 1 — Working council and memory mutations (Oct 14-27)

**Goal:** five end-to-end governed decisions with a real LLM, plus an offline equivalent for CI.

- **Krishna:** SQLAlchemy models and Alembic migrations; repository methods; `search_related_claims`; transactional, idempotent `apply_decision` for all actions; version preservation; rollback tests.
- **Mahalakshmi:** Historian v1 (deterministic temporal logic + LLM assessment); Synthesizer v1 and prompt; duplicate / temporal-update / value-conflict / source-conflict detection; validate cited claim IDs; 8+ temporal/provenance fixtures.
- **Sai Chetan:** provider adapter (real + fake) with repair and typed errors; Judge v1 and prompt; evidence-ID validation; token/latency/cost recording; 12+ evidence fixtures.
- **Shailen:** policy engine with thresholds, rule IDs, and Synthesizer veto; concurrent Historian + Judge; `dry_run` and `commit` modes; cost aggregation; end-to-end tests.

**Milestone:** demo of promote, reject, retain, temporal revision, and source-conflict quarantine; at least 24 integration fixtures.

### Sprint 2 — Baselines, dataset, pilot (Oct 28-Nov 10)

- **Krishna:** `direct_storage`, `latest_write_wins`, `static_rag`; reusable initial-memory fixtures; expected-versus-actual memory-delta comparison; 15+ state-transition cases.
- **Mahalakshmi:** 20+ temporal/provenance/source-conflict cases; temporal accuracy and provenance completeness metrics; independently label a teammate's cases and log disagreements; Historian-only ablation on `dev`.
- **Sai Chetan:** `single_judge` and `generic_ensemble`; 20+ supported/refuted/mixed/insufficient cases; Judge accuracy and contradiction F1; matched-settings enforcement.
- **Shailen:** dataset loader/validator; experiment manifest; resumable JSONL runner; primary metrics; run the first pilot on 60+ `dev` cases across all systems.

**Milestone:** pilot results table (proposed system vs. direct, latest-write-wins, single judge, generic ensemble) from saved result files. Workbook Sprint 3 (pilot) ends Nov 10.

### Sprint 3 — Main experiments and error analysis (Nov 11-24)

- Grow dataset toward 120+; build and freeze the `test` split with double labeling.
- Run required ablations; paired bootstrap confidence intervals.
- Error analysis by category: Krishna (storage/state), Mahalakshmi (provenance/time/conflict), Sai Chetan (Judge/evidence), Shailen (policy/veto/cost).
- Freeze prompts and thresholds before the first official `test` run.

**Milestone:** first official held-out run archived with manifest.

### Sprint 4 — Hardening and demonstration (Nov 25-Dec 8)

- Fix defects found in error analysis; Docker build and one-command startup; pinned dependencies; clean-environment verification.
- Immutable results to tables and figures; reproducibility guide.
- Each owner freezes their module and documents known limitations.
- Tag `v0.1.0`; rehearse the demo (section 19). Shailen coordinates but does not solely present.

**Milestone:** reviewer can clone `v0.1.0`, start the app, submit a claim with evidence, see Historian/Judge/Synthesizer/policy outputs, inspect memory and audit trace, run the pilot, and reproduce the summary from saved results.

### Sprint 5 — Demonstration (Dec 9-11)

Each member verifies their module, resolves final defects, and presents their part.

### Semester 2 (295B) outline

Repeat across seeds and (if affordable) a second model; sequential claim-stream track; complete ablations and statistics; failure-mode analysis; optional Atlas adapter, human-review UI, local model, larger adversarial set; final report/thesis and archive.

---

## 19. Demonstration Script

Three scenarios: (1) supported promotion; (2) temporal revision preserving the old version; (3) conflict quarantine from a newer but unreliable source. For each, show initial memory, candidate and evidence, Historian, Judge, Synthesizer, rules fired, final action, updated memory history, and latency/calls/tokens/cost. Finish with one pilot-results table.

---

## 20. Success Criteria

**By Dec 8:** one reproducible monorepo; strict contracts; end-to-end decisions with transactional updates; offline test mode; all required baselines implemented; 60+ reviewed `dev` cases and a frozen `test` split started; reproducible metrics and audit traces.

**End of semester 2:** repeated cost-aware experiments; complete ablations and error analysis; tested reproducibility package; a clear answer to each research question including negative or mixed findings.

---

## 21. Open Decisions and Risks

### Open decisions

| ID | Decision | Owner | Due |
|---|---|---|---|
| D1 | LLM provider, exact model, budget source, rate limits, caching of dev responses | Sai Chetan | Oct 13 |
| D2 | Written advisor confirmation of: API-based LLM (no HPC/Nautilus deployment), Self-RAG optional, role split in this spec. The Oct 6 meeting notes assign cluster deployment, which this spec drops. | Shailen | Oct 13 |
| D3 | Branch protection availability on the repository plan | Shailen | Oct 11 |

### Risks

| Risk | Mitigation |
|---|---|
| LLM access or budget failure | Fake provider for CI; one low-cost model; cached dev responses; fixed dataset |
| Merge conflicts / drift | Trunk-based short branches; daily rebase; contract tests; walking skeleton first |
| Council wins only by using more calls | Generic ensemble; cost/token-normalized analysis |
| Labels subjective or circular | Annotation guide independent of policy; two annotators; adjudication; role-level labels; frozen test split |
| Extraction hides governance quality | Atomic-claim primary track |
| Invalid JSON | Strict validation, two repairs, explicit escalation |
| False knowledge corrupts memory | Dry-run default; deterministic policy; quarantine; immutable history; transactional writes |
| Weak novelty versus LLM memory managers | Cite Mem0 / Zep / A-MEM / Letta-style systems; emphasize temporal/provenance governance and controlled evaluation |
| Schedule too ambitious | Self-RAG, Atlas adapter, graph DB, UI stay optional |
| One person becomes the integration bottleneck | Backup reviewers; shared experiment execution; contracts landed early; load rebalanced in 6.2 |
| Prompt tuning leaks test labels | Dev/test split; freeze before the official run |

---

## 22. Instructions for AI Coding Agents

See `AGENTS.md` for the operational version. The rules in short:

1. Read `AGENTS.md`, then only the spec sections your task names.
2. Work only within your assigned module; use documented interfaces or test doubles for others.
3. Never change contracts or enums without an ADR and owner approval.
4. Prefer deterministic code for state transitions, thresholds, metrics, and aggregation.
5. All LLM calls go through the provider interface; keep fake-provider support.
6. Add or update tests with every behavior change.
7. No secrets in code, fixtures, logs, or commits.
8. Record scientifically relevant configuration; never silently fall back to another model, prompt, policy, or evidence set.
9. Never tune against `test` labels.
10. Preserve old claim versions and audit records.
11. Return explicit typed errors; no silent failures.
12. Keep PRs small. The policy engine is the final authority for memory mutation.
