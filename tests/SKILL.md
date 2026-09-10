---
name: gnn-test-suite
description: GNN comprehensive test suite execution and management. Use when running tests, writing new test cases, checking coverage, debugging test failures, or validating pipeline correctness across all 25 GNN modules.
---

# GNN Test Suite (Step 2)

## Purpose

Executes the comprehensive test suite across all GNN pipeline modules. Manages test discovery, execution, coverage reporting, and result aggregation.

## Key Commands

```bash
# Run all tests via pipeline step
python src/2_tests.py --comprehensive

# Run tests directly with pytest
uv run --extra dev python -m pytest tests/ -v

# Module-specific tests
uv run --extra dev python -m pytest tests/gnn/ -v
uv run --extra dev python -m pytest tests/render/ -v
uv run --extra dev python -m pytest tests/export/ -v

# Check coverage
pytest --cov=src --cov-report=term-missing

# Run with specific markers
uv run --extra dev python -m pytest tests/ -v -m "not slow"

# Quick smoke test
uv run --extra dev python -m pytest tests/ -x -q --tb=short
```

## Test Organization

```
tests/
├── gnn/                       # GNN parsing tests
├── render/                    # Code generation tests
├── export/                    # Export format tests
├── visualization/             # Visualization tests
├── pipeline/                  # Pipeline integration tests
└── conftest.py                # Shared fixtures
```

## Writing New Tests

- Place tests in `tests/<module>/test_{module}_*.py`
- Use real methods only in production code
- Follow existing patterns: fixtures, parametrize, clear assertions
- Target >80% coverage per module

## Current Status

- **1,522+ tests** passing
- **100% pipeline success rate**
- All 25 steps validated


## MCP Tools

This module defines MCP tools in `mcp.py` (`run_all_tests`,
`run_integration_tests`, `run_unit_tests`) with working handlers, but the
`tests` package is deliberately excluded from MCP auto-discovery
(`discovery_excluded_dirs={"tests"}` in `src/mcp/mcp.py`), so these tools are
not registered on the live GNN MCP server.

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification
- [../../pytest.ini](../../pytest.ini) — Pytest configuration


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
