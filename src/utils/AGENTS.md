# Utils Module - Agent Scaffolding

## Module Overview

**Purpose**: Shared utilities and helper functions for the GNN processing pipeline

**Pipeline Step**: Infrastructure module (not a numbered step)

**Category**: Utility Functions / Infrastructure Support

**Status**: ✅ Production Ready

**Version**: 2.0.0

**Last Updated**: 2026-01-07

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

### Logging Functions

#### `setup_step_logging(step_name: str, verbose: bool = False) -> logging.Logger`
**Description**: Set up standardized logging for a pipeline step with correlation ID tracking

**Parameters**:
- `step_name` (str): Name of the pipeline step (e.g., "3_gnn")
- `verbose` (bool): Enable verbose logging (default: False)

**Returns**: `logging.Logger` - Configured logger instance with correlation ID

**Example**:
```python
from utils import setup_step_logging
logger = setup_step_logging("3_gnn", verbose=True)
```

#### `setup_main_logging(log_dir: Optional[Path] = None, verbose: bool = False) -> logging.Logger`
**Description**: Set up logging for main pipeline orchestrator

**Parameters**:
- `log_dir` (Optional[Path]): Directory for log files (default: None)
- `verbose` (bool): Enable verbose logging (default: False)

**Returns**: `logging.Logger` - Configured main logger instance

#### `log_step_start(logger_or_step_name: Union[logging.Logger, str], message: str = None, step_number: int = None, **metadata) -> None`
**Description**: Log the start of a pipeline step with performance tracking

**Parameters**:
- `logger_or_step_name` (Union[logging.Logger, str]): Logger instance or step name
- `message` (str, optional): Custom start message
- `step_number` (int, optional): Step number for display
- `**metadata`: Additional metadata to log

**Returns**: `None`

#### `log_step_success(logger_or_step_name: Union[logging.Logger, str], message: str = None, step_number: int = None, **metadata) -> None`
**Description**: Log successful completion of a pipeline step with metrics

**Parameters**:
- `logger_or_step_name` (Union[logging.Logger, str]): Logger instance or step name
- `message` (str, optional): Custom success message
- `step_number` (int, optional): Step number for display
- `**metadata`: Additional metadata (results, file counts, etc.)

**Returns**: `None`

#### `log_step_error(logger_or_step_name: Union[logging.Logger, str], message: str = None, step_number: int = None, **metadata) -> None`
**Description**: Log an error during pipeline step execution with context

**Parameters**:
- `logger_or_step_name` (Union[logging.Logger, str]): Logger instance or step name
- `message` (str, optional): Custom error message
- `step_number` (int, optional): Step number for display
- `**metadata`: Error context (exception, traceback, etc.)

**Returns**: `None`

#### `log_step_warning(logger_or_step_name: Union[logging.Logger, str], message: str = None, step_number: int = None, **metadata) -> None`
**Description**: Log a warning during pipeline step execution

**Parameters**:
- `logger_or_step_name` (Union[logging.Logger, str]): Logger instance or step name
- `message` (str, optional): Warning message
- `step_number` (int, optional): Step number for display
- `**metadata`: Warning context

**Returns**: `None`

#### `get_performance_summary() -> Dict[str, Any]`
**Description**: Get summary of performance metrics across all tracked operations

**Returns**: `Dict[str, Any]` - Performance summary with timing, memory, and resource usage

#### `setup_correlation_context(step_name: str, correlation_id: Optional[str] = None) -> str`
**Description**: Set up correlation context for request tracking

**Parameters**:
- `step_name` (str): Name of the pipeline step
- `correlation_id` (Optional[str]): Existing correlation ID or None to generate new

**Returns**: `str` - Correlation ID for this context

### Argument Parsing Functions

#### `ArgumentParser.parse_step_arguments(step_name: str) -> argparse.Namespace`
**Description**: Parse arguments for a specific pipeline step with fallback support

**Parameters**:
- `step_name` (str): Name of the pipeline step

**Returns**: `argparse.Namespace` - Parsed arguments with standard pipeline options

**Standard Arguments**:
- `--target-dir`: Target directory for input files
- `--output-dir`: Output directory for results
- `--verbose`: Enable verbose logging
- `--recursive`: Recursively process directories

#### `build_step_command_args(step_name: str, args: argparse.Namespace) -> List[str]`
**Description**: Build command-line arguments for a pipeline step

**Parameters**:
- `step_name` (str): Name of the pipeline step
- `args` (argparse.Namespace): Parsed arguments

**Returns**: `List[str]` - Command-line argument list

#### `validate_and_convert_paths(args: argparse.Namespace) -> argparse.Namespace`
**Description**: Validate and convert string paths to Path objects

**Parameters**:
- `args` (argparse.Namespace): Arguments with path strings

**Returns**: `argparse.Namespace` - Arguments with Path objects

### Pipeline Utilities

#### `get_output_dir_for_script(script_name: str, base_output_dir: Optional[Path] = None) -> Path`
**Description**: Get standardized output directory for a pipeline script

**Parameters**:
- `script_name` (str): Name of the script (e.g., "3_gnn.py")
- `base_output_dir` (Optional[Path]): Base output directory (default: Path("output"))

**Returns**: `Path` - Output directory path (e.g., "output/3_gnn_output/")

#### `validate_output_directory(output_dir: Path, create: bool = True) -> bool`
**Description**: Validate and optionally create output directory

**Parameters**:
- `output_dir` (Path): Output directory path
- `create` (bool): Create directory if it doesn't exist (default: True)

**Returns**: `bool` - True if directory is valid/created, False otherwise

### Resource Management Functions

#### `get_current_memory_usage() -> float`
**Description**: Get current process memory usage

**Returns**: `float` - Memory usage in megabytes (MB)

### Error Recovery Functions

#### `ErrorRecoveryManager.recover(step_name: str, error: Exception, context: Dict[str, Any]) -> Optional[Dict[str, Any]]`
**Description**: Attempt to recover from a step failure

**Parameters**:
- `step_name` (str): Name of the failed step
- `error` (Exception): The exception that occurred
- `context` (Dict[str, Any]): Error context and state

**Returns**: `Optional[Dict[str, Any]]` - Recovery result or None if recovery not possible

#### `format_and_log_error(logger: logging.Logger, error: Exception, context: Dict[str, Any] = None) -> None`
**Description**: Format and log an error with full context

**Parameters**:
- `logger` (logging.Logger): Logger instance
- `error` (Exception): The exception to log
- `context` (Dict[str, Any], optional): Additional error context

**Returns**: `None`

### Configuration Functions

#### `load_config(config_path: Path) -> Dict[str, Any]`
**Description**: Load configuration from YAML or JSON file

**Parameters**:
- `config_path` (Path): Path to configuration file

**Returns**: `Dict[str, Any]` - Configuration dictionary

#### `get_config_value(key: str, default: Any = None) -> Any`
**Description**: Get a configuration value by key

**Parameters**:
- `key` (str): Configuration key (supports dot notation, e.g., "pipeline.steps")
- `default` (Any): Default value if key not found

**Returns**: `Any` - Configuration value or default

#### `set_config_value(key: str, value: Any) -> None`
**Description**: Set a configuration value

**Parameters**:
- `key` (str): Configuration key (supports dot notation)
- `value` (Any): Value to set

**Returns**: `None`

### Dependency Management Functions

#### `validate_pipeline_dependencies() -> Dict[str, bool]`
**Description**: Validate all pipeline dependencies are installed

**Returns**: `Dict[str, bool]` - Dependency status (package_name: is_installed)

#### `check_optional_dependencies(dependency_group: str) -> bool`
**Description**: Check if optional dependency group is available

**Parameters**:
- `dependency_group` (str): Dependency group name (e.g., "pymdp", "jax")

**Returns**: `bool` - True if dependencies are available

#### `install_missing_dependencies(dependencies: List[str]) -> bool`
**Description**: Install missing dependencies

**Parameters**:
- `dependencies` (List[str]): List of package names to install

**Returns**: `bool` - True if installation succeeded

### Performance Tracking Functions

#### `PerformanceTracker.track_operation(name: str, func: Callable, *args, **kwargs) -> Any`
**Description**: Track performance of an operation

**Parameters**:
- `name` (str): Operation name
- `func` (Callable): Function to track
- `*args`: Function arguments
- `**kwargs`: Function keyword arguments

**Returns**: `Any` - Function return value

#### `track_operation_standalone(name: str, func: Callable, *args, **kwargs) -> Any`
**Description**: Standalone function to track operation performance

**Parameters**:
- `name` (str): Operation name
- `func` (Callable): Function to track
- `*args`: Function arguments
- `**kwargs`: Function keyword arguments

**Returns**: `Any` - Function return value

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

## Troubleshooting

### Common Issues

#### Issue 1: Logging not working
**Symptom**: No log output or logs in wrong location  
**Cause**: Logging configuration incorrect or permissions issues  
**Solution**: 
- Verify log directory exists and is writable
- Check logging level configuration
- Use `--verbose` flag for detailed logging
- Review logging configuration in pipeline config

#### Issue 2: Argument parsing errors
**Symptom**: Script fails with argument parsing errors  
**Cause**: Argument definition mismatch or missing required arguments  
**Solution**:
- Verify argument definitions match script usage
- Check required arguments are provided
- Review argument parser configuration
- Use `--help` flag to see expected arguments

---

## Version History

### Current Version: 2.0.0

**Features**:
- Centralized logging system
- Argument parsing utilities
- Resource monitoring
- Performance tracking
- Error handling utilities

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced performance monitoring
- **Future**: Real-time resource tracking

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [Pipeline Module](../pipeline/AGENTS.md)

### External Resources
- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)

---

**Last Updated**: 2026-01-07
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: 2.0.0
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern