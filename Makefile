PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: lock setup test lint format typecheck contracts run clean

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install $(if $(wildcard constraints.txt),-c constraints.txt) -e ".[dev]"
	@test -f .env || cp .env.example .env

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff format --check .
	$(BIN)/ruff check .

format:
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck:
	$(BIN)/mypy

# Regenerate contracts/v1/*.json from the Pydantic models (the canonical contracts).
contracts:
	$(BIN)/python scripts/generate_contracts.py

run:
	$(BIN)/uvicorn sekf.api.app:create_app --factory --reload --port 8000

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov

# Refresh exact dependency versions after adding or upgrading a dependency.
lock:
	$(BIN)/pip freeze --exclude-editable > constraints.txt
