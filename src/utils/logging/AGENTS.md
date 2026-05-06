# Logging Sub-module

## Overview

Centralized, production-grade logging infrastructure for the GNN pipeline. Provides correlation-aware tracing across all 25 steps, visual progress indicators, and multi-format structured output (JSON-L + Text).

## Architecture

Follows the **Thin Orchestrator** pattern where individual steps delegate to this centralized system to ensure consistent formatting and deduplication.

```
logging/
├── __init__.py          # Public API exports
├── logging_utils.py     # Core implementation (Hardened v1.6.0)
├── AGENTS.md            # Agent capabilities and status
├── README.md            # Developer usage guide
└── SPEC.md              # Technical specification
```

## Hardening Achievements (v1.6.0)

- **Remediated Duplication**: Fixed a legacy collision between root and step-level handlers that caused duplicate terminal output.
- **Unified Tracing**: Established a single-source-of-truth for correlation IDs, ensuring logs from subprocesses are correctly tagged and formatted.
- **Compatibility Shim**: Provided a transparent shim in `src/utils/logging_utils.py` to upgrade legacy modules to the modern structured system automatically.

## Key Capabilities

- **`setup_step_logging`** — Comprehensive configuration with auto-detection of terminal capabilities.
- **`log_step_*` Suite** — Semantic logging with visual icons (🚀, ✅, ⚠️, ❌) and progress tracking.
- **`timed_operation`** — Context manager for automatic $O(N)$ performance instrumentation.
- **Structured JSON-L** — Concurrent machine-readable logging for downstream analysis.
- **`logging.TRACE` (numeric 5)** — Registered for high-volume parser diagnostics; enable root/handlers at TRACE when debugging parse traces without flooding default `--verbose` output.

## Features

- **Correlation Tracing**: `[ID:STEP]` tags on every line for cross-module debugging.
- **Rich Aesthetics**: Vibrant terminal output with glassmorphism-compatible visual logging.
- **Performance Aware**: Integrated memory and duration tracking in every log line.
- **Resilient Initialization**: Defensive handler management prevents double-logging even with redundant calls.

## Documentation
- [Technical Specification](SPEC.md)
- [Developer Guide](README.md)

---
**Version**: 1.6.0  
**Status**: Production Hardened  
**Pipeline Integration**: Steps 0-24
