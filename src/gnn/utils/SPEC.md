# Utils Module Specification

## Overview

The `utils` module provides standardized infrastructure for the GNN pipeline, including logging, configuration, error handling, and base classes for processing.

## Core Components

### 1. logging_utils.py + observability/structured_logging.py
Logging facade (`logging_utils.py`) and structured pipeline logging with correlation tracking (`observability/structured_logging.py`). The structured logger decorates lifecycle events with status markers in the log output.

**Functions (facade):**
- `log_step_start(logger, msg)` - Log step start (rocket marker in structured output)
- `log_step_success(logger, msg)` - Log success (check marker)
- `log_step_warning(logger, msg)` - Log warning
- `log_step_error(logger, msg)` - Log error
- `setup_step_logging(name, verbose)` - Create configured logger

**Design:** Standard library only for the facade; recovery-safe.

### 2. pipeline_orchestration/base_processor.py
Abstract base class for standardized processing patterns.

**Classes:**
- `ProcessingResult` - Dataclass for processing outcomes
- `BaseProcessor` - ABC with file discovery, error handling, reporting

**Factory:**
- `create_processor(step_name, process_func)` - Wrap simple functions

### 3. arguments/ (arg_parsing, arg_definitions, step_config, ...)
Argument parsing and validation: `ArgumentParser.parse_step_arguments`, `build_step_command_args`, `audit_step_contracts`, shared `STEP_ARGUMENTS` and `StepConfiguration`. Recovery defaults come from the shared `fallback_default_for` table (`_FALLBACK_DEFAULTS`); `StepConfiguration.validate_step_args` accepts an injectable `project_root`.

### 4. config_io/config_loader.py
Pipeline configuration management: `load_config`, `get_config_value`, `set_config_value`, `validate_config`.

### 5. runtime_safety/dependency_validator.py + observability/performance_tracking.py
Dependency validation (`validate_pipeline_dependencies`, `check_optional_dependencies`, `get_dependency_status`) and performance tracking (`PerformanceTracker.track_operation`, `track_operation_standalone`).

## Import Patterns

```python
# Logging facade (always works)
from gnn.utils.logging_utils import log_step_start, log_step_success

# Full utilities
from gnn.utils import BaseProcessor, ProcessingResult, PipelineLogger
```

## Testing

Tests in: `tests/utils/` (including `test_new_utils.py`, `test_utils_core.py`, and `test_shared_helpers.py`).

Run: `uv run --extra dev python -m pytest tests/utils/ -v`

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
