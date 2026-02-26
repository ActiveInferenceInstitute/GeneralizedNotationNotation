---
name: gnn-shared-utilities
description: GNN shared utility functions and helper modules. Use when working with common pipeline utilities, logging helpers, file I/O wrappers, path management, or pipeline template infrastructure.
---

# GNN Shared Utilities

## Purpose

Provides shared utility functions, helper modules, and common infrastructure used across all 25 pipeline steps. Includes logging, error handling, configuration, dependency validation, and the standardized pipeline template.

## Key Commands

```bash
# Utils is a shared library — not run directly but imported by all pipeline steps.
# Validate pipeline dependencies
python -c "from utils import validate_pipeline_dependencies; validate_pipeline_dependencies()"

# Check optional dependency status
python -c "from utils import get_dependency_status; print(get_dependency_status())"

# Run pipeline health check
python -c "from utils import generate_pipeline_health_report; print(generate_pipeline_health_report())"
```

## Key Modules

| Module | Key Exports | Purpose |
| -------- | ------------ | --------- |
| `pipeline_template` | `log_step_start`, `log_step_success`, `log_step_error`, `log_step_warning` | Visual step logging |
| `logging` | `PipelineLogger`, `setup_step_logging`, `StructuredLogger` | Structured logging |
| `error_handling` | `ErrorRecoveryManager`, `PipelineErrorHandler`, `generate_correlation_id` | Error handling & recovery |
| `configuration` | `ConfigurationManager`, `get_config`, `set_config`, `validate_config` | Pipeline configuration |
| `dependency` | `DependencyValidator`, `DependencyAuditor`, `validate_pipeline_dependencies` | Dependency management |
| `performance` | `PerformanceTracker`, `track_operation_standalone` | Performance monitoring |

## API

```python
from utils import (
    # Logging
    PipelineLogger, setup_step_logging, setup_main_logging,
    log_step_start, log_step_success, log_step_error, log_step_warning,
    StructuredLogger, get_pipeline_logger,

    # Error handling
    ErrorRecoveryManager, PipelineErrorHandler, generate_correlation_id,
    format_error_message, get_recovery_manager,

    # Configuration
    ConfigurationManager, get_config, set_config, validate_config,
    get_pipeline_config, load_config, save_config,

    # Dependencies
    DependencyValidator, validate_pipeline_dependencies,
    check_optional_dependencies, get_dependency_status,

    # Performance
    PerformanceTracker, performance_tracker, get_performance_summary,

    # Pipeline utilities
    parse_arguments, validate_and_convert_paths,
    get_output_dir_for_script, validate_output_directory,
    BaseProcessor, ProcessingResult, create_processor
)

# Setup logging for a pipeline step
logger = setup_step_logging("step_name", verbose=True)
log_step_start(logger, "Processing started")
log_step_success(logger, "Processing completed")

# Configuration management
config = get_pipeline_config()
validate_config(config)

# Dependency validation
validate_pipeline_dependencies()

# Performance tracking
with performance_tracker.track("operation_name"):
    do_work()
```

## Visual Logging Features

- 🎨 Color-coded status indicators (green=success, yellow=warning, red=error)
- 📊 Progress bars and completion indicators
- 🔢 Correlation ID tracking for debugging
- 📋 Structured summary tables
- ⏱️ Performance timing and memory tracking


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `get_utility_info`
- `validate_dependencies`
- `get_pipeline_template_info`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
