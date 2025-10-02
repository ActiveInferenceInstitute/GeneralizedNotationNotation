# Utils Module - Agent Scaffolding

## Module Overview

**Purpose**: Shared utilities and helper functions for the GNN processing pipeline

**Pipeline Step**: Infrastructure module (not a numbered step)

**Category**: Utility Functions / Infrastructure Support

---

## Core Functionality

### Primary Responsibilities
1. Pipeline orchestration and coordination utilities
2. Logging and diagnostic utilities
3. Configuration and argument parsing utilities
4. Resource management and monitoring utilities
5. Error handling and recovery utilities
6. Performance tracking and optimization utilities

### Key Capabilities
- Centralized logging and diagnostic system
- Argument parsing and configuration management
- Resource monitoring and performance tracking
- Error handling and recovery mechanisms
- Pipeline orchestration and coordination
- Utility functions for common operations

---

## API Reference

### Public Functions

#### `setup_step_logging(step_name, verbose=False) -> logging.Logger`
**Description**: Set up logging for a pipeline step

**Parameters**:
- `step_name`: Name of the pipeline step
- `verbose`: Enable verbose logging

**Returns**: Configured logger instance

#### `get_output_dir_for_script(script_name, base_output_dir) -> Path`
**Description**: Get output directory for a specific script

**Parameters**:
- `script_name`: Name of the script
- `base_output_dir`: Base output directory

**Returns**: Path to script's output directory

#### `create_standardized_pipeline_script(step_name, module_function, description, **kwargs) -> Callable`
**Description**: Create standardized pipeline script wrapper

**Parameters**:
- `step_name`: Name of the pipeline step
- `module_function`: Main processing function
- `description`: Step description
- `**kwargs`: Additional arguments

**Returns**: Wrapped pipeline script function

#### `get_current_memory_usage() -> float`
**Description**: Get current memory usage

**Returns**: Memory usage in MB

#### `attempt_step_recovery(script_name, step_result, args, logger) -> Optional[Dict]`
**Description**: Attempt to recover from step failure

**Parameters**:
- `script_name`: Name of failed script
- `step_result`: Step execution result
- `args`: Pipeline arguments
- `logger`: Logger instance

**Returns**: Recovery result or None

---

## Dependencies

### Required Dependencies
- `pathlib` - Path manipulation
- `logging` - Logging functionality
- `argparse` - Argument parsing
- `typing` - Type hints

### Optional Dependencies
- `psutil` - System resource monitoring
- `numpy` - Numerical computations

### Internal Dependencies
- None (base infrastructure module)

---

## Configuration

### Logging Configuration
```python
LOGGING_CONFIG = {
    'console_level': 'INFO',
    'file_level': 'DEBUG',
    'correlation_tracking': True,
    'structured_logging': True
}
```

### Performance Configuration
```python
PERFORMANCE_CONFIG = {
    'memory_tracking': True,
    'timing_tracking': True,
    'resource_monitoring': True
}
```

---

## Usage Examples

### Step Logging Setup
```python
from utils.logging_utils import setup_step_logging

logger = setup_step_logging("3_gnn.py", verbose=True)
logger.info("Starting GNN processing")
```

### Output Directory Management
```python
from utils.pipeline import get_output_dir_for_script

output_dir = get_output_dir_for_script("3_gnn.py", Path("output"))
print(f"GNN output directory: {output_dir}")
```

### Pipeline Script Creation
```python
from utils.pipeline_template import create_standardized_pipeline_script

run_script = create_standardized_pipeline_script(
    "3_gnn.py",
    process_gnn_files,
    "GNN file processing"
)

# Execute the script
exit_code = run_script()
```

### Memory Monitoring
```python
from utils.resource_manager import get_current_memory_usage

memory_before = get_current_memory_usage()
# ... do some work ...
memory_after = get_current_memory_usage()
print(f"Memory delta: {memory_after - memory_before} MB")
```

---

## Output Specification

### Output Products
- Log files in configured log directory
- Performance metrics and timing data
- Error reports and recovery logs
- Configuration validation reports

### Output Directory Structure
```
output/
├── logs/
│   ├── pipeline.log
│   ├── step_logs/
│   └── error_logs/
└── performance/
    ├── timing_data.json
    └── memory_usage.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: Variable (utility functions)
- **Memory**: ~10-50MB overhead
- **Status**: ✅ Production Ready

### Expected Performance
- **Logging**: < 1ms per log entry
- **Path Operations**: < 1ms per operation
- **Memory Monitoring**: < 5ms per check
- **Configuration**: < 10ms per operation

---

## Error Handling

### Utility Errors
1. **Configuration Errors**: Invalid configuration parameters
2. **Path Errors**: Invalid or inaccessible paths
3. **Logging Errors**: Logging system failures
4. **Resource Errors**: Resource monitoring failures

### Recovery Strategies
- **Configuration Repair**: Use default values
- **Path Resolution**: Resolve relative paths
- **Logging Fallback**: Use basic logging
- **Resource Monitoring**: Continue without monitoring

---

## Integration Points

### Orchestrated By
- All pipeline scripts and modules

### Imports From
- None (base infrastructure module)

### Imported By
- All pipeline scripts (0_template.py through 23_report.py)
- All pipeline modules

### Data Flow
```
Configuration → Logging Setup → Resource Monitoring → Error Handling → Performance Tracking
```

---

## Testing

### Test Files
- `src/tests/test_utils_integration.py` - Integration tests
- `src/tests/test_utils_functionality.py` - Functionality tests

### Test Coverage
- **Current**: 93%
- **Target**: 95%+

### Key Test Scenarios
1. Logging and diagnostic utilities
2. Configuration and argument parsing
3. Resource management and monitoring
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `utils.get_system_info` - Get system information
- `utils.get_environment_info` - Get environment information
- `utils.get_logging_info` - Get logging configuration
- `utils.validate_dependencies` - Validate dependencies
- `utils.get_performance_metrics` - Get performance metrics

### Tool Endpoints
```python
@mcp_tool("utils.get_system_info")
def get_system_info_tool():
    """Get system information"""
    # Implementation
```

---

**Last Updated**: October 1, 2025
**Status**: ✅ Production Ready