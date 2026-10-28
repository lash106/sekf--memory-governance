# SEKF: Self-Evolving Knowledge Framework

A research prototype for governing what AI agents store in long-term memory.

SEKF evaluates candidate claims using three specialized roles:

- Historian: analyzes time, provenance, and conflicts.
- Judge: evaluates evidence support.
- Synthesizer: combines the role outputs and makes an advisory recommendation.
- Policy Engine: deterministically makes the final memory action.

The system compares this approach with simpler baselines:

- Direct storage
- Latest-write-wins
- Static RAG
- Single-model judge
- Generic ensemble

The project evaluates whether role specialization improves memory-governance decisions while preserving evidence, history, and auditability.


## Repository Structure

```text
src/sekf/
├── api/
├── contracts/
├── memory/
├── historian/
├── judge/
├── synthesizer/
├── policy/
├── providers/
├── baselines/
└── evaluation/

datasets/       Evaluation cases and annotation guide
experiments/    Experiment configurations and results
prompts/        Versioned role prompts
tests/          Unit, contract, and integration tests
SPEC.md         Full project specification
AGENTS.md       Instructions for AI coding agents
```

## Quick start

```bash
make setup      # venv + dependencies (Python 3.12)
make test       # offline tests, fake roles
make run        # API on http://localhost:8000  (GET /health, POST /v1/decide)
```

### Configuration

`make setup` copies `.env.example` to `.env` (git-ignored). The defaults need no network and no key. They run the real rule-based Historian, the real policy and the real SQLite memory store (`./data/sekf.sqlite3`, git-ignored), with a fake Judge and a fake Synthesizer.

| Variable | Values | Meaning |
|---|---|---|
| `SEKF_MEMORY` | `sql` (default), `memory` | `sql` persists to SQLite at `SEKF_DB_URL`; `memory` is a test double that applies no claim changes |
| `SEKF_HISTORIAN` | `deterministic` (default), `fake` | rule-based Historian or the canned fake |
| `SEKF_PROVIDER` | `fake` (default), `openai` | fake Judge, or the LLM Judge (needs `SEKF_LLM_*`) |

An unknown value stops the app with an error (`ConfigurationError`, or `ProviderUnavailable` for `SEKF_PROVIDER`); nothing falls back silently.

Send the worked-example request (`SPEC.md` section 4) to a running server. The first call promotes the claim and a repeat is answered from storage as a replay; the REVISE outcome needs the older 17:00 claim in memory first (see `tests/integration/test_sprint1_milestone.py`):

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' \
  -d "$(python3 -c "import json;print(json.dumps(json.load(open('tests/fixtures/lab_closing_revise.json'))['request']))")"
```

Design and team workflow: see `SPEC.md`. AI-agent rules: `AGENTS.md`.
