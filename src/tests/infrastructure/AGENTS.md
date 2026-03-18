# Tests Infrastructure - Agent Scaffolding

## Module Overview

**Purpose**: Provide reusable infrastructure for running pytest suites with resource monitoring and structured reporting.

**Parent**: `src/tests/`

This folder is not a pipeline step; it is a support library used by the test runner and test tooling.

---

## Public API

`src/tests/infrastructure/__init__.py` re-exports the public surface:

- **Configuration**: `TestExecutionConfig`, `TestExecutionResult`
- **Monitoring**: `ResourceMonitor`, `PSUTIL_AVAILABLE`
- **Runner**: `TestRunner`
- **Reports**: `generate_markdown_report`, `generate_fallback_report`, `generate_timeout_report`, `generate_error_report`
- **Utilities**: `check_test_dependencies`, `build_pytest_command`, `extract_collection_errors`, `parse_test_statistics`, `parse_coverage_statistics`

---

## Key files

- `test_config.py`: dataclasses defining test configuration and result shape
- `resource_monitor.py`: CPU/memory monitoring (optional `psutil`)
- `test_runner.py`: `TestRunner` implementation
- `report_generator.py`: report assembly (markdown + recovery reports)
- `utils.py`: pytest command building and output parsing helpers

---

## Testing guidance

This module should be validated by:

- unit tests for parsing helpers (collection errors, stats extraction)
- smoke tests for `TestRunner` command construction (without requiring long-running executions)

