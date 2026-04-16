# Logging Sub-module

## Overview

Centralized logging infrastructure for the GNN pipeline. Provides step-level logging, visual progress indicators, color-coded output, and structured logging with correlation ID tracking.

## Architecture

```
logging/
├── __init__.py          # Re-exports core logging functions (23 lines)
└── logging_utils.py     # Full logging implementation (1070 lines)
```

## Key Functions

- **`setup_step_logging(step_name, verbose)`** — Configure logging for a pipeline step with appropriate handlers and formatters.
- **`log_step_start(step, description)`** — Log step initiation with visual progress indicators.
- **`log_step_success(step, summary)`** — Log step completion with timing and result summary.
- **`log_step_error(step, error)`** — Log step failure with traceback and recovery suggestions.
- **`log_step_warning(step, warning)`** — Log non-fatal warnings with context.

## Features

- **Color-coded output** — Green (success), yellow (warning), red (error), blue (info).
- **Correlation IDs** — Unique tracking IDs for debugging across pipeline steps.
- **Screen reader support** — Accessible output with emoji-free alternatives.
- **Performance tracking** — Built-in timing and memory usage instrumentation.

## Parent Module

See [utils/AGENTS.md](../AGENTS.md) for the overall utilities architecture.

**Version**: 1.6.0
