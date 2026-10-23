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

Design and team workflow: see `SPEC.md`. AI-agent rules: `AGENTS.md`.
