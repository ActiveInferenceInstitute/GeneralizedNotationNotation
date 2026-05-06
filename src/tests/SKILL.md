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
pytest src/tests/ -v

# Module-specific tests
pytest src/tests/test_gnn_*.py -v
pytest src/tests/test_render_*.py -v
pytest src/tests/test_export_*.py -v

# Check coverage
pytest --cov=src --cov-report=term-missing

# Run with specific markers
pytest src/tests/ -v -m "not slow"

# Quick smoke test
pytest src/tests/ -x -q --tb=short
```

## Test Organization

```
src/tests/
├── test_gnn_*.py              # GNN parsing tests
├── test_render_*.py           # Code generation tests
├── test_export_*.py           # Export format tests
├── test_visualization_*.py    # Visualization tests
├── test_pipeline_*.py         # Pipeline integration tests
└── conftest.py                # Shared fixtures
```

## Writing New Tests

- Place tests in `src/tests/test_{module}_*.py`
- Use real methods only — **no mocks in production code**
- Follow existing patterns: fixtures, parametrize, clear assertions
- Target >80% coverage per module

## Current Status

- **1,522+ tests** passing
- **100% pipeline success rate**
- All 25 steps validated


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `run_all_tests`
- `run_integration_tests`
- `run_unit_tests`

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
