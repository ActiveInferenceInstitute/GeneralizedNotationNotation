# Step 2: Tests — Test Suite Execution

**Script**: `src/2_tests.py` (thin orchestrator, ~85 lines)  
**Module**: `src/tests/` (120+ `test_*.py` modules; item counts vary with markers and ignores)  
**Last Updated**: 2026-03-24

## Overview

Step 2 orchestrates the GNN test suite. It follows the **Thin Orchestrator** pattern: the script delegates to `tests.run_tests()` in [`src/tests/runner.py`](../../../src/tests/runner.py) (~600 lines).

`pyproject.toml` enables **`--strict-markers`**. Test modules must only use markers that are registered (in `pyproject.toml` or via pytest plugins). Unregistered markers (for example `anyio` without `pytest-anyio`) cause **collection errors** before any tests run.

## Usage

```bash
# Fast tests (default) — ~2–4 min typical
python src/2_tests.py --fast-only --verbose

# Comprehensive — longer; all discovered tests
python src/2_tests.py --comprehensive --verbose

# Skip tests entirely (e.g., for rapid iteration)
SKIP_TESTS_IN_PIPELINE=1 python src/main.py
```

## Architecture

```
src/2_tests.py                          ← thin orchestrator
    └── tests.run_tests()               ← runner.py
            ├── run_fast_pipeline_tests()   ← test_runner_modes.py
            ├── run_comprehensive_tests()   ← test_runner_modes.py
            └── run_fast_reliable_tests()   ← test_runner_modes.py (alternative)
```

Auto-escalation: if the fast run collects **0** tests, `run_tests(..., auto_fallback=True)` (default) retries with comprehensive mode.

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--fast-only` | flag | ✅ | Run only fast tests |
| `--comprehensive` | flag | ❌ | Run full discovered suite |

## Environment Variables

| Variable | Effect |
|----------|--------|
| `SKIP_TESTS_IN_PIPELINE` | Skip test execution entirely (returns True) |
| `FAST_TESTS_TIMEOUT` | Override timeout (default: 600 seconds) |

## Test Suite at a Glance

| Metric | Value |
|--------|-------|
| Test modules | `find src/tests -maxdepth 1 -name 'test_*.py' \| wc -l` |
| Items collected | `uv run pytest src/tests/ --collect-only -q` |
| Categories | Defined in `categories.py` / runner (counts drift) |
| Pytest markers | Registered in `conftest.py` and `pyproject.toml` |
| Execution modes | 3 (fast, comprehensive, reliable) |

## Output

Pipeline step output directory: `output/2_tests_output/`

**Fast pipeline mode** ([`run_fast_pipeline_tests`](../../../src/tests/test_runner_modes.py)) writes:

- `test_execution_report.json` (under `output/2_tests_output/`) — `execution_summary` includes `tests_run`, `tests_passed`, `tests_failed`, `tests_skipped`, `tests_errors`, `collection_errors`, `exit_code`, `command`, `timestamp`. Counts are parsed from pytest’s final summary line via [`parse_test_statistics`](../../../src/tests/infrastructure/utils.py) (also exposes legacy-compatible keys `total`, `passed`, `failed`, `skipped`, `errors`).
- [`pytest_reliable_output.txt`](../../../src/tests/test_runner_modes.py) — captured stdout/stderr from the subprocess.
- [`test_execution_report.md`](../../../src/tests/test_runner_modes.py) — human-readable summary (from [`flatten_pipeline_test_summary`](../../../src/tests/infrastructure/report_generator.py) + [`generate_markdown_report`](../../../src/tests/infrastructure/report_generator.py)).
- `pytest.log` — when using repo [`pytest.ini`](../../../pytest.ini), `log_file` is `output/2_tests_output/pytest.log` (created when pytest runs with that config).

**Comprehensive / modular runner** paths may also emit `pytest_stdout.txt`, `pytest_stderr.txt`, and per-category outputs under subfolders (see [`test_runner_modular.py`](../../../src/tests/test_runner_modular.py), [`tests/runner.py`](../../../src/tests/runner.py)).

`htmlcov/` appears when pytest is run with `--cov` (not the default for fast pipeline step 2).

## MCP Tools

Step 2 registers 3 MCP tools (via `src/tests/mcp.py`):

| Tool | Description |
|------|-------------|
| `process_tests` | Run the GNN test suite |
| `get_test_results` | Retrieve the latest test execution results |
| `get_tests_module_info` | Return tests module version and capabilities |

## Source Code Connections

| Component | File | Key Function |
|-----------|------|--------------|
| Thin orchestrator | [2_tests.py](../../../src/2_tests.py) | `_test_runner_wrapper()` |
| Core runner | [tests/runner.py](../../../src/tests/runner.py) | `run_tests()` |
| Category definitions | [tests/categories.py](../../../src/tests/categories.py) | `MODULAR_TEST_CATEGORIES` |
| Fixtures & markers | [tests/conftest.py](../../../src/tests/conftest.py) | `PYTEST_MARKERS`, fixtures |
| MCP tools | [tests/mcp.py](../../../src/tests/mcp.py) | `register_tools()` |
| MCP audit | [tests/test_mcp_audit.py](../../../src/tests/test_mcp_audit.py) | `TestMCPModuleDiscovery`, `TestMCPDomainTools` |
| Fast runner + parsing | [tests/test_runner_modes.py](../../../src/tests/test_runner_modes.py) | `run_fast_pipeline_tests()` |
| Stats / coverage helpers | [tests/infrastructure/utils.py](../../../src/tests/infrastructure/utils.py) | `parse_test_statistics`, `parse_coverage_statistics` |
| Parsing unit tests | [tests/test_infrastructure_utils_statistics.py](../../../src/tests/test_infrastructure_utils_statistics.py) | `parse_test_statistics`, `flatten_pipeline_test_summary` |

## Deep Dive

See the full testing documentation hub: **[doc/gnn/testing/README.md](../testing/README.md)**
