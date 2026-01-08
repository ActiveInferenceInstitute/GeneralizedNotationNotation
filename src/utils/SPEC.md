# Utils Module Specification

## Overview

The `utils` module provides standardized infrastructure for the GNN pipeline, including logging, configuration, error handling, and base classes for processing.

## Core Components

### 1. step_logging.py
Minimal, always-importable logging functions for pipeline steps.

**Functions:**
- `log_step_start(logger, msg)` - Log step start with 🚀 emoji
- `log_step_success(logger, msg)` - Log success with ✅ emoji
- `log_step_warning(logger, msg)` - Log warning with ⚠️ emoji
- `log_step_error(logger, msg)` - Log error with ❌ emoji
- `setup_step_logging(name, verbose)` - Create configured logger

**Design:** Zero external dependencies, fallback-safe.

### 2. base_processor.py
Abstract base class for standardized processing patterns.

**Classes:**
- `ProcessingResult` - Dataclass for processing outcomes
- `BaseProcessor` - ABC with file discovery, error handling, reporting

**Factory:**
- `create_processor(step_name, process_func)` - Wrap simple functions

### 3. logging_utils.py (1071 lines)
Full-featured logging with correlation tracking.

### 4. argument_utils.py (1225 lines)
Argument parsing and validation utilities.

### 5. config_loader.py
YAML configuration management.

## Import Patterns

```python
# Minimal (always works)
from utils.step_logging import log_step_start, log_step_success

# Full utilities
from utils import BaseProcessor, ProcessingResult, PipelineLogger
```

## Testing

Tests in: `tests/test_new_utils.py` (11 passing tests)
