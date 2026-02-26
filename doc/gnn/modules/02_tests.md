# Step 2: Tests — Test Suite Execution

**Script**: `src/2_tests.py` (86 lines, thin orchestrator)  
**Module**: `src/tests/` (91 test files, 734+ functions, 24 categories)  
**Last Updated**: February 24, 2026

## Overview

Step 2 orchestrates the GNN test suite. It follows the **Thin Orchestrator** pattern: the script itself is 86 lines and delegates all logic to `tests.run_tests()` in `src/tests/runner.py` (618 lines).

## Usage

```bash
# Fast tests (default) — 1-3 min
python src/2_tests.py --fast-only --verbose

# Comprehensive — 5-15 min (all 91 test files)
python src/2_tests.py --comprehensive --verbose

# Skip tests entirely (e.g., for rapid iteration)
SKIP_TESTS_IN_PIPELINE=1 python src/main.py
```

## Architecture

```
src/2_tests.py                          ← thin orchestrator (86 lines)
    └── tests.run_tests()               ← runner.py (618 lines)
            ├── run_fast_pipeline_tests()   ← test_runner_modes.py
            ├── run_comprehensive_tests()   ← test_runner_modes.py
            └── run_fast_reliable_tests()   ← test_runner_modes.py (fallback)
```

Auto-fallback: if fast mode collects 0 tests, it automatically escalates to comprehensive.

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--fast-only` | flag | ✅ | Run only fast tests |
| `--comprehensive` | flag | ❌ | Run all 91 test files |

## Environment Variables

| Variable | Effect |
|----------|--------|
| `SKIP_TESTS_IN_PIPELINE` | Skip test execution entirely (returns True) |
| `FAST_TESTS_TIMEOUT` | Override timeout (default: 600 seconds) |

## Test Suite at a Glance

| Metric | Value |
|--------|-------|
| Test files | 91 |
| Test functions | 1,522 |
| Test categories | 24 (in `categories.py`) |
| Pytest markers | 21 (in `conftest.py`) |
| MCP audit tests | 125 (in `test_mcp_audit.py`) |
| Execution modes | 3 (fast, comprehensive, reliable) |

## Output

Pipeline step output directory: `output/2_tests_output/`

```
output/2_tests_output/
├── test_execution_report.json       ← structured results
├── pytest_stdout.txt                ← raw pytest stdout  
├── pytest_stderr.txt                ← raw pytest stderr  
└── htmlcov/                         ← coverage report (if --cov enabled)
```

## MCP Tools

Step 2 registers 3 MCP tools (via `src/tests/mcp.py`):

| Tool | Description |
|------|-------------|
| `process_tests` | Run the GNN test suite |
| `get_test_results` | Retrieve the latest test execution results |
| `get_tests_module_info` | Return tests module version and capabilities |

## Source Code Connections

| Component | File | Key Function |
|-----------|------|-------------|
| Thin orchestrator | [2_tests.py](#placeholder) | `_test_runner_wrapper()` |
| Core runner | [tests/runner.py](#placeholder) | `run_tests()` |
| Category definitions | [tests/categories.py](#placeholder) | `MODULAR_TEST_CATEGORIES` |
| Fixtures & markers | [tests/conftest.py](#placeholder) | `PYTEST_MARKERS`, fixtures |
| MCP tools | [tests/mcp.py](#placeholder) | `register_tools()` |
| MCP audit | [tests/test_mcp_audit.py](#placeholder) | `TestMCPModuleDiscovery`, `TestMCPDomainTools` |

## Deep Dive

See the full testing documentation hub: **[doc/gnn/testing/README.md](../testing/README.md)**
