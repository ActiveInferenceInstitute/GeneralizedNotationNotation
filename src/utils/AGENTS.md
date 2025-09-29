# Utils Module - Agent Scaffolding

## Module Overview

**Purpose**: Centralized utility functions and infrastructure components for the entire GNN pipeline

**Category**: Core Infrastructure / Utilities

---

## Core Functionality

### Primary Responsibilities
1. Standardized logging and progress tracking
2. Argument parsing and validation
3. Error handling and recovery
4. Resource management
5. Performance monitoring
6. Dependency management

### Key Capabilities
- Correlation ID generation and tracking
- Structured logging with step context
- Robust argument parsing with fallbacks
- Circuit breaker patterns
- Memory and timing tracking
- Safe file operations

---

## Module Components

### Logging Utilities
- `logging_utils.py` - Structured logging, PipelineLogger
- `progress_tracking.py` - PipelineProgressTracker

### Argument Processing
- `argument_utils.py` - ArgumentDefinition, PipelineArguments, EnhancedArgumentParser

### Error Management
- `error_handling.py` - StandardizedErrorHandler, ErrorCategory
- `error_recovery.py` - ErrorRecoveryManager, RecoveryStrategy

### Resource Management
- `resource_manager.py` - ResourceManager, resource limits
- `performance_tracker.py` - PerformanceTracker, timing utilities

### Pipeline Support
- `pipeline_template.py` - create_standardized_pipeline_script
- `dependency_manager.py` - DependencyManager, availability checks

---

## API Reference

### Logging

#### `setup_step_logging(script_name: str, log_dir: Path) -> logging.Logger`
**Description**: Setup standardized logging for pipeline step

#### `log_step_start(logger, step_name, correlation_id)`
**Description**: Log step start with correlation tracking

#### `log_step_success(logger, step_name, duration)`
**Description**: Log successful step completion

#### `log_step_error(logger, step_name, error, correlation_id)`
**Description**: Log step error with context

---

### Argument Parsing

#### `EnhancedArgumentParser`
**Description**: Extended ArgumentParser with pipeline-specific features

**Methods**:
- `parse_step_arguments(script_name: str) -> argparse.Namespace`
- `add_pipeline_arguments()`
- `validate_arguments(args: argparse.Namespace) -> bool`

---

### Error Handling

#### `StandardizedErrorHandler`
**Description**: Centralized error handling with categorization

**Methods**:
- `handle_error(error: Exception, context: Dict) -> ErrorResult`
- `categorize_error(error: Exception) -> ErrorCategory`
- `suggest_recovery(error_category: ErrorCategory) -> List[str]`

---

### Performance Tracking

#### `PerformanceTracker`
**Description**: Track timing and resource usage

**Methods**:
- `start_timing(operation: str)`
- `stop_timing(operation: str) -> float`
- `get_memory_usage() -> float`
- `generate_report() -> Dict`

---

## Usage Examples

### Logging Setup
```python
from utils.logging_utils import setup_step_logging, log_step_start

logger = setup_step_logging("3_gnn.py", Path("output/logs"))
correlation_id = log_step_start(logger, "GNN Processing", None)
```

### Argument Parsing
```python
from utils.argument_utils import EnhancedArgumentParser

parser = EnhancedArgumentParser("5_type_checker.py")
args = parser.parse_step_arguments("5_type_checker.py")
```

### Error Handling
```python
from utils.error_handling import StandardizedErrorHandler, ErrorCategory

handler = StandardizedErrorHandler()
try:
    process_file(path)
except Exception as e:
    result = handler.handle_error(e, {"file": path})
    logger.error(f"Error: {result.message}")
```

### Performance Tracking
```python
from utils.performance_tracker import PerformanceTracker

tracker = PerformanceTracker()
tracker.start_timing("parsing")
# ... do work ...
duration = tracker.stop_timing("parsing")
report = tracker.generate_report()
```

---

## Dependencies

### Required Dependencies
- `logging` - Python logging
- `argparse` - Argument parsing
- `pathlib` - Path operations
- `json` - Configuration loading
- `datetime` - Timestamp generation
- `psutil` - Resource monitoring

---

## Error Categories

### Supported Categories
- `DEPENDENCY_ERROR` - Missing dependencies
- `FILE_ERROR` - File operation failures
- `VALIDATION_ERROR` - Data validation failures
- `RESOURCE_ERROR` - Resource exhaustion
- `TIMEOUT_ERROR` - Operation timeouts
- `RUNTIME_ERROR` - General runtime errors

---

## Performance Characteristics

### Typical Overhead
- Logging setup: <1ms
- Argument parsing: <5ms
- Performance tracking: <0.1ms per operation
- Error handling: <1ms

---

## Testing

### Test Files
- `src/tests/test_utils_*.py` - Component-specific tests
- `src/tests/test_pipeline_improvements_validation.py` - Integration tests

### Test Coverage
- **Current**: 88%
- **Target**: 90%+

---

**Last Updated**: September 29, 2025  
**Status**: ✅ Production Ready - Core Infrastructure

