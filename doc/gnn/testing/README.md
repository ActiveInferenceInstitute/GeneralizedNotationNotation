# GNN Test Suite Hub

**Last Updated**: February 24, 2026  
**Source of truth**: [`src/tests/TEST_SUITE_SUMMARY.md`](../../../src/tests/TEST_SUITE_SUMMARY.md) · [`src/tests/categories.py`](../../../src/tests/categories.py)  
**Total test files**: 91 · **Total test functions**: 1,522 · **Test categories**: 24

## Architecture

```
src/tests/
├── conftest.py              ← 601 lines: 21 pytest markers, all shared fixtures
├── categories.py            ← 293 lines: 24 MODULAR_TEST_CATEGORIES with timeouts
├── runner.py                ← 618 lines: run_tests(), 3 execution modes
├── __init__.py              ← 286 lines: public API, TEST_CATEGORIES, PYTEST_MARKERS
├── mcp.py                   ← MCP tool registration (3 tools for the tests module)
├── infrastructure/
│   ├── test_runner.py       ← TestRunner class, ResourceMonitor, TestExecutionConfig
│   ├── resource_monitor.py  ← Memory + CPU tracking (psutil, optional)
│   ├── report_generator.py  ← JSON / Markdown test reports
│   ├── test_config.py       ← TestExecutionConfig dataclass
│   └── utils.py             ← build_pytest_command(), check_test_dependencies()
├── helpers/
│   └── render_recovery.py   ← helpers for render-step recovery testing
├── test_data/               ← representative GNN files used as fixtures
├── test_mcp_audit.py        ← 203-test MCP audit suite
├── test_pipeline_*.py       ← 15 pipeline orchestration files
├── test_*_overall.py        ← per-module system tests
└── test_execute_pymdp_*.py  ← 8 PyMDP-specific execute tests
```

## Execution Modes

Three modes are implemented in `runner.py` and exposed by `2_tests.py`:

| Mode | Flag | Duration | What Runs |
|------|------|----------|-----------|
| **Fast** | `--fast-only` (default) | 1–3 min | `run_fast_pipeline_tests()` — essential path only |
| **Comprehensive** | `--comprehensive` | 5–15 min | `run_comprehensive_tests()` — all 91 files |
| **Reliable** | auto-fallback | ~90 sec | `run_fast_reliable_tests()` — critical path fallback |

Auto-fallback: if fast mode collects 0 tests, mode automatically escalates to comprehensive.

## Module Coverage Matrix (from `categories.py`)

| Category | Test Files | Timeout | Parallel |
|----------|-----------|---------|---------|
| `gnn` | 5 | 120s | ✅ |
| `render` | 4 | 120s | ✅ |
| `mcp` | 5 | 120s | ✅ |
| `audio` | 4 | 120s | ✅ |
| `visualization` | 8 | 120s | ✅ |
| `pipeline` | 15 | 1800s | ❌ |
| `export` | 1 | 90s | ✅ |
| `execute` | 9 | 300s | ✅ |
| `llm` | 3 | 120s | ✅ |
| `ontology` | 1 | 90s | ✅ |
| `website` | 1 | 90s | ✅ |
| `report` | 4 | 90s | ✅ |
| `environment` | 5 | 90s | ✅ |
| `type_checker` | 1 | 90s | ✅ |
| `validation` | 1 | 90s | ✅ |
| `model_registry` | 1 | 90s | ✅ |
| `analysis` | 1 | 120s | ✅ |
| `integration` | 2 | 120s | ✅ |
| `security` | 1 | 90s | ✅ |
| `research` | 1 | 90s | ✅ |
| `ml_integration` | 1 | 120s | ✅ |
| `advanced_visualization` | 1 | 120s | ✅ |
| `gui` | 2 | 90s | ✅ |
| `comprehensive` | 11 | 300s | ❌ |

**Total: 91 test files across 24 categories**

## Running Tests

```bash
# Fast mode (default) — ~1-3 min
PYTHONPATH=src python src/2_tests.py --fast-only --verbose

# Comprehensive — ~5-15 min  
PYTHONPATH=src python src/2_tests.py --comprehensive --verbose

# Direct pytest — specific category
PYTHONPATH=src python -m pytest src/tests/test_gnn_overall.py -v

# By marker
PYTHONPATH=src python -m pytest src/tests/ -m "fast" -q
PYTHONPATH=src python -m pytest src/tests/ -m "not slow" -q

# MCP audit only
PYTHONPATH=src python -m pytest src/tests/test_mcp_audit.py -v

# With coverage
PYTHONPATH=src python -m pytest src/tests/ --cov=src --cov-report=html
```

## Output Structure

```
output/2_tests_output/
├── test_execution_report.json       ← structured results (TestRunner.generate_report())
├── pytest_stdout.txt                ← raw pytest stdout
├── pytest_stderr.txt                ← raw pytest stderr
├── coverage.json                    ← coverage data (if --cov enabled)
└── htmlcov/                         ← HTML coverage report
```

## Pipeline Integration

`src/2_tests.py` is a **thin orchestrator** (86 lines). It:

1. Reads env vars `SKIP_TESTS_IN_PIPELINE` (skip entirely) and `FAST_TESTS_TIMEOUT`
2. Delegates all execution to `tests.run_tests()` in `runner.py`
3. Returns a bool success flag to `src/main.py`

## See Also

- [test_patterns.md](test_patterns.md) — No-mock policy, fixtures, markers, assertion style
- [mcp_audit.md](mcp_audit.md) — The 5-layer MCP audit framework anatomy
- [modules/02_tests.md](../modules/02_tests.md) — Pipeline Step 2 documentation
- [`src/tests/TEST_SUITE_SUMMARY.md`](../../../src/tests/TEST_SUITE_SUMMARY.md) — canonical test suite summary (with full module coverage matrix)
- [REPO_COHERENCE_CHECK.md](../REPO_COHERENCE_CHECK.md) — quality gates and standards
