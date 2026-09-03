# Test Infrastructure Sub-module

## Overview

Core test infrastructure providing the test runner, configuration, resource monitoring, report generation, and shared utilities for the GNN test suite.

## Architecture

```
infrastructure/
├── __init__.py             # Infrastructure exports
├── test_runner.py          # Custom test runner with progress tracking
├── test_config.py          # TestExecutionConfig / TestExecutionResult
├── report_generator.py     # Markdown/fallback/timeout/error report generators
├── resource_monitor.py     # Memory and CPU monitoring during tests
└── utils.py                # Shared parsing/error-collection utilities
```

## Key Components

- **`TestRunner`** — Custom pytest runner with progress bars, resource tracking, and structured output.
- **`TestExecutionConfig`** (`test_config.py`) — Test execution configuration and result container.
- **Report generation** (`report_generator.py`) — `generate_markdown_report()`, `generate_fallback_report()`, `generate_timeout_report()`, `generate_error_report()`, and `flatten_pipeline_test_summary()`.
- **`ResourceMonitor`** — Tracks memory and CPU usage during test execution for performance regression detection.

## Parent Module

See [tests/AGENTS.md](../AGENTS.md) for the overall test architecture.

**Version**: 3.2.0
