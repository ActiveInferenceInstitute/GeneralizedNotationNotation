# Logging Utilities Module

## Purpose

This module provides centralized logging utilities for the GNN processing pipeline. It standardizes log formatting, file handling, and output across all pipeline steps.

## Key Files

- `logging_utils.py` - Core logging configuration and utilities

## Functionality

- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- File and console logging handlers
- Timestamped log entries
- Pipeline step-specific loggers
- Performance metrics logging

## Usage

```python
from utils.logging.logging_utils import setup_logging, get_logger

# Setup logging for a pipeline step
setup_logging(verbose=True, log_file="output/step.log")

# Get a logger for a module
logger = get_logger(__name__)
logger.info("Processing started")
```

## Integration Points

- All 24 pipeline step scripts use these utilities
- MCP tools log through this system
- Test framework uses consistent logging

## Maintenance

When modifying logging:
1. Maintain backward compatibility with existing log formats
2. Ensure thread-safe logging for parallel execution
3. Test log rotation and file handling
