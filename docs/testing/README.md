# GNN Testing Guide

The repository uses `pytest` for unit and integration coverage. Tests live primarily
under `tests/`, with module-focused subdirectories and root-level tests for
cross-cutting behavior. The pytest configuration and markers are defined in
`pyproject.toml`; this guide does not duplicate a fictional test-tree inventory or
hard-code pass counts.

## Fast feedback

Run the documentation and parser contracts first when changing documentation or GNN
syntax:

```bash
uv run --frozen --no-sync python -m pytest \
  tests/docs/test_doc_contracts.py \
  tests/docs/test_docs_audit.py \
  tests/gnn/ \
  -q
```

The documentation checks themselves are:

```bash
uv run --frozen --no-sync python docs/development/docs_audit.py \
  --strict --check-anchors --no-write
uv run --frozen --no-sync python scripts/check_doc_contracts.py --strict
uv run --frozen --no-sync python scripts/check_gnn_doc_patterns.py --strict
uv run --frozen --no-sync python scripts/check_maintained_doc_terms.py --strict
uv run --frozen --no-sync python scripts/check_repo_terminology.py --strict
```

## Standard test suite

The ordinary source profile uses an explicit marker selection:

```bash
uv sync --frozen --extra dev --python 3.14
uv run --frozen --no-sync python -m pytest tests/ -q --tb=short -rsx \
  -m 'not ollama and not pipeline and not mcp and not env_heavy and not toolchain'
```

Pipeline, MCP, native tools, GUI and all-extras tests retain separate workflow
profiles. Use the [workflow roster](../../.github/workflows/README.md) for their
exact selections and runtimes. A raw full-suite invocation requires those
prerequisites; skipping them locally does not qualify a release.

Useful focused commands:

```bash
# Parser and schema behavior.
uv run --frozen --no-sync python -m pytest tests/gnn/ -q

# Renderer and executor contracts.
uv run --frozen --no-sync python -m pytest tests/render/ tests/execute/ -q

# Pipeline integration.
uv run --frozen --no-sync python -m pytest tests/pipeline/ -q

# Collect without running.
uv run --frozen --no-sync python -m pytest tests/ --collect-only -q
```

## Markers and optional tests

[pyproject.toml](../../pyproject.toml) owns declared markers, including ordinary,
pipeline, MCP, Ollama, environment-heavy and native toolchain classifications.
Use marker expressions from the matching workflow; unregistered markers fail
collection under strict marker checking.

When testing LLM behavior, set the provider environment variables in a local ignored
`.env` file and never commit credentials. Ollama-specific tests remain opt-in because
they require a running local service.

## Coverage and quality

Use `just test-cov`, `just lint`, `just format-check` and `just typecheck`
for the current local recipes. The ordinary core floor is configured in
[pyproject.toml](../../pyproject.toml). Comprehensive native coverage requires
[strict independent reports](../development/native_comprehensive_coverage.md)
per configured interpreter. [Installed-platform qualification](../development/installed_platform_acceptance.md)
uses ordinary wheel installations outside the checkout. Partial local tests,
imports and cross-environment unions cannot substitute for these requirements.

## Documentation and example policy

- GNN examples intended for validation must include all five enforced sections.
- Commands in maintained docs must match the live `--help` output of the referenced
  CLI or script.
- Generated artifacts under `output/` are evidence, not maintained fixtures.
- Tests should assert user-visible behavior such as exit codes, files, summaries, and
  parsed results rather than private implementation details.

For the CI job split and required checks, see
[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml). For test-module
orientation, see [`tests/README.md`](../../tests/README.md).
