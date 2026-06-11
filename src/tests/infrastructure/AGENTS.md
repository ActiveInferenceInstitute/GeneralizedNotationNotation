# Test Infrastructure Sub-module

## Overview

Core test infrastructure providing the test runner, configuration, resource monitoring, report generation, and shared utilities for the GNN test suite.

## Architecture

```
infrastructure/
├── __init__.py             # Infrastructure exports (62 lines)
├── test_runner.py          # Custom test runner with progress tracking (296 lines)
├── test_config.py          # Test configuration and environment detection (41 lines)
├── report_generator.py     # Test report generation (HTML, JSON) (130 lines)
├── resource_monitor.py     # Memory and CPU monitoring during tests (99 lines)
└── utils.py                # Shared infrastructure utilities (365 lines)
```

## Key Components

- **`TestRunner`** — Custom pytest runner with progress bars, resource tracking, and structured output.
- **`TestConfig`** — Detects environment capabilities (GPU, Ollama, network) and configures test markers.
- **`ReportGenerator`** — Produces HTML and JSON test reports with timing and coverage data.
- **`ResourceMonitor`** — Tracks memory and CPU usage during test execution for performance regression detection.

## Parent Module

See [tests/AGENTS.md](../AGENTS.md) for the overall test architecture.

**Version**: 1.6.0
