# Infrastructure Tests Agent

## Overview

This directory owns tests for the test-suite infrastructure itself: environment validation (Python, dependencies, system, `uv`), JAX + PyMDP stack validation, coverage meta-tests, and the custom test runner with its output/coverage parsing helpers.

## Module Structure

- **Environment validation** — `test_environment_python.py`, `test_environment_dependencies.py`, `test_environment_system.py`, `test_environment_integration.py`, `test_environment_overall.py`, `test_uv_environment.py`
- **Stack validation** — `test_jax_pymdp_stack_validation.py`
- **Coverage meta-tests** — `test_coverage_assessment.py`, `test_coverage_gap_infrastructure.py`, `test_coverage_overall.py`
- **Runner** — `test_runner.py` (canonical `TestRunner`; `tests.runner` re-exports it, do not add a second copy)
- **Statistics/parsing** — `test_infrastructure_utils_statistics.py`

Supporting non-test modules: `utils.py` (pytest command building, output parsing, dependency checks), `test_config.py` (`TestExecutionConfig` / `TestExecutionResult`), `report_generator.py` (markdown/timeout/error report generation), `resource_monitor.py` (memory/CPU monitoring).

## Purpose

- Validate that the environment the suite runs in is real and complete before other tests depend on it.
- Keep the custom runner and its parsing helpers pinned to observable behavior.
- Do not place production implementation logic here.

## Verification

```bash
uv run --extra dev python -m pytest tests/infrastructure -q
```

## Parent Module

See [tests/AGENTS.md](../AGENTS.md) for the overall test architecture.

**Version**: 3.2.0
