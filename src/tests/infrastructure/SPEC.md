# Test Infrastructure — Technical Specification

**Version**: 1.6.0

## Test Runner

- Custom pytest runner (`test_runner.py`)
- Progress tracking with visual indicators
- Resource monitoring (memory, CPU) via `resource_monitor.py`
- Markdown/fallback/timeout/error report generation via `report_generator.py`

## Test Configuration

- `test_config.py` provides the `TestExecutionConfig` dataclass (timeout, max failures, parallel, coverage, markers, memory/CPU limits) and the `TestExecutionResult` dataclass.

## Utility Functions

- `utils.py` — `check_test_dependencies()`, `build_pytest_command()`, `extract_collection_errors()`, `parse_test_statistics()`, `parse_coverage_statistics()`

