# Test Infrastructure

Core test runner, configuration, monitoring, and reporting infrastructure.

## Components

| File | Purpose |
|------|---------|
| `test_runner.py` | Custom pytest runner with progress tracking |
| `utils.py` | Shared parsing/error-collection utilities |
| `report_generator.py` | Markdown/fallback/timeout/error report generation |
| `resource_monitor.py` | Memory/CPU monitoring during tests |
| `test_config.py` | `TestExecutionConfig` / `TestExecutionResult` |

## See Also

- [Parent: tests/README.md](../README.md)
