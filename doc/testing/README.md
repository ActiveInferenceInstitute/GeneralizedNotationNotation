# GNN Testing Guide

The repository uses `pytest` for unit and integration coverage. Tests live primarily
under `src/tests/`, with module-focused subdirectories and root-level tests for
cross-cutting behavior. The pytest configuration and markers are defined in
`pyproject.toml`; this guide does not duplicate a fictional test-tree inventory or
hard-code pass counts.

## Fast feedback

Run the documentation and parser contracts first when changing documentation or GNN
syntax:

```bash
uv run --extra dev python -m pytest \
  src/tests/test_doc_contracts.py \
  src/tests/test_docs_audit.py \
  src/tests/gnn/ \
  -q
```

The documentation checks themselves are:

```bash
uv run --extra dev python doc/development/docs_audit.py \
  --strict --check-anchors --no-write
uv run --extra dev python scripts/check_doc_contracts.py --strict
uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict
uv run --extra dev python scripts/check_maintained_doc_terms.py --strict
uv run --extra dev python scripts/check_repo_terminology.py --strict
```

## Standard test suite

Use the repository's command of record. The two Ollama tests are explicitly ignored
unless a local Ollama daemon and the configured test model are available:

```bash
uv sync --extra dev
uv run --extra dev python -m pytest src/tests/ -q --tb=short \
  --ignore=src/tests/llm/test_llm_ollama.py \
  --ignore=src/tests/llm/test_llm_ollama_integration.py
```

The suite includes environment-sensitive Julia, GUI, renderer, and pipeline tests.
Run the full command in CI-like environments; use a focused path when iterating on a
single module.

Useful focused commands:

```bash
# Parser and schema behavior.
uv run --extra dev python -m pytest src/tests/gnn/ -q

# Renderer and executor contracts.
uv run --extra dev python -m pytest src/tests/render/ src/tests/execute/ -q

# Pipeline integration.
uv run --extra dev python -m pytest src/tests/pipeline/ -q

# Collect without running.
uv run --extra dev python -m pytest src/tests/ --collect-only -q
```

## Markers and optional tests

The project declares `slow`, `integration`, `unit`, `uv`, `xfail`, `pipeline`, and
`mcp` markers in `pyproject.toml`. Use marker expressions only with markers declared
there, for example:

```bash
uv run --extra dev python -m pytest src/tests/ -m "not slow" -q
uv run --extra dev python -m pytest src/tests/ -m "not pipeline and not mcp" -q
```

When testing LLM behavior, set the provider environment variables in a local ignored
`.env` file and never commit credentials. Ollama-specific tests remain opt-in because
they require a running local service.

## Coverage and quality

```bash
uv run --extra dev python -m pytest src/tests/ \
  --cov=src --cov-report=term-missing \
  --ignore=src/tests/llm/test_llm_ollama.py \
  --ignore=src/tests/llm/test_llm_ollama_integration.py
uv run ruff check src scripts
uv run ruff format --check src scripts
uv run mypy src --show-error-codes
```

The project coverage floor is configured in `pyproject.toml`. Do not infer a coverage
or pass guarantee from a partial local run; report the command and environment with
any measured result.

## Documentation and example policy

- GNN examples intended for validation must include all five enforced sections.
- Commands in maintained docs must match the live `--help` output of the referenced
  CLI or script.
- Generated artifacts under `output/` are evidence, not maintained fixtures.
- Tests should assert user-visible behavior such as exit codes, files, summaries, and
  parsed results rather than private implementation details.

For the CI job split and required checks, see
[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml). For test-module
orientation, see [`src/tests/README.md`](../../src/tests/README.md).
