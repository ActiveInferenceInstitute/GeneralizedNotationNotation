# Pipeline Module - Agent Scaffolding

## Module Overview

**Purpose**: Pipeline orchestration, configuration management, and execution coordination for the GNN processing system

**Pipeline Step**: Infrastructure module (not a numbered step)

**Category**: Pipeline Infrastructure / Orchestration

**Status**: ✅ Production Ready

**Version**: 2.0.0

**Last Updated**: 2026-01-07

---

## Core Functionality

### Primary Responsibilities
1. Pipeline execution orchestration and step coordination
2. Configuration management and validation
3. Step discovery and dependency management
4. Pipeline health monitoring and diagnostics
5. Execution planning and resource estimation
6. Pipeline validation and verification

### Key Capabilities
- Multi-step pipeline orchestration
- Dynamic step discovery and configuration
- Pipeline health monitoring and alerting
- Resource estimation and allocation
- Execution plan generation
- Performance tracking and optimization
- Error recovery and retry mechanisms

---

## API Reference

### Configuration Functions

#### `get_pipeline_config() -> Dict[str, Any]`
**Description**: Get the complete pipeline configuration including step metadata and settings

**Returns**: `Dict[str, Any]` - Pipeline configuration dictionary with:
- `steps`: List of step names
- `output_base_dir`: Base output directory
- `log_level`: Logging level
- `step_configs`: Step-specific configurations

**Example**:
```python
from pipeline import get_pipeline_config
config = get_pipeline_config()
print(f"Steps: {config['steps']}")
```

#### `set_pipeline_config(config: Dict[str, Any]) -> None`
**Description**: Set pipeline configuration

**Parameters**:
- `config` (Dict[str, Any]): Configuration dictionary to set

**Returns**: `None`

#### `get_output_dir_for_script(script_name: str, base_output_dir: Optional[Path] = None) -> Path`
**Description**: Get standardized output directory for a pipeline script

**Parameters**:
- `script_name` (str): Name of the pipeline script (e.g., "3_gnn.py")
- `base_output_dir` (Optional[Path]): Base output directory (default: Path("output"))

**Returns**: `Path` - Output directory path (e.g., "output/3_gnn_output/")

**Example**:
```python
from pipeline import get_output_dir_for_script
from pathlib import Path
output_dir = get_output_dir_for_script("3_gnn.py", Path("output"))
```

### Validation Functions

#### `validate_step_prerequisites(script_name: str, args: argparse.Namespace, logger: logging.Logger) -> Dict[str, Any]`
**Description**: Validate prerequisites for a pipeline step (input files, dependencies, etc.)

**Parameters**:
- `script_name` (str): Name of the script to validate (e.g., "3_gnn.py")
- `args` (argparse.Namespace): Command line arguments
- `logger` (logging.Logger): Logger instance

**Returns**: `Dict[str, Any]` - Validation results with:
- `passed` (bool): Whether validation passed
- `warnings` (List[str]): List of warnings
- `errors` (List[str]): List of errors
- `missing_dependencies` (List[str]): Missing dependencies

**Example**:
```python
from pipeline.pipeline_validator import validate_step_prerequisites
result = validate_step_prerequisites("3_gnn.py", args, logger)
if not result["passed"]:
    print(f"Validation failed: {result['errors']}")
```

#### `validate_pipeline_step_sequence(steps_to_execute: List[tuple], logger: logging.Logger) -> Dict[str, Any]`
**Description**: Validate the execution sequence of pipeline steps (dependency order, circular dependencies)

**Parameters**:
- `steps_to_execute` (List[tuple]): List of step tuples (step_name, step_function)
- `logger` (logging.Logger): Logger instance

**Returns**: `Dict[str, Any]` - Sequence validation results with:
- `valid` (bool): Whether sequence is valid
- `dependency_errors` (List[str]): Dependency violations
- `circular_dependencies` (List[str]): Detected circular dependencies
- `recommended_order` (List[str]): Recommended execution order

### Execution Planning Functions

#### `generate_execution_plan(steps_to_execute: List[tuple], args: argparse.Namespace, logger: logging.Logger) -> Dict[str, Any]`
**Description**: Generate execution plan for pipeline steps with resource estimates

**Parameters**:
- `steps_to_execute` (List[tuple]): List of steps to execute
- `args` (argparse.Namespace): Pipeline arguments
- `logger` (logging.Logger): Logger instance

**Returns**: `Dict[str, Any]` - Execution plan with:
- `estimated_duration` (float): Estimated total duration in seconds
- `resource_requirements` (Dict[str, Any]): Memory, CPU, disk requirements
- `step_estimates` (List[Dict]): Per-step estimates
- `parallelization_opportunities` (List[str]): Steps that can run in parallel

**Example**:
```python
from pipeline.pipeline_planner import generate_execution_plan
plan = generate_execution_plan(steps_to_execute, args, logger)
print(f"Estimated time: {plan['estimated_duration']}s")
```

### Execution Functions

#### `run_pipeline(target_dir: Path, output_dir: Path, steps: Optional[List[str]] = None, **kwargs) -> bool`
**Description**: Execute the complete GNN processing pipeline

**Parameters**:
- `target_dir` (Path): Directory containing input files
- `output_dir` (Path): Output directory for results
- `steps` (Optional[List[str]]): Steps to execute (None = all steps)
- `**kwargs`: Additional pipeline options (verbose, skip_steps, etc.)

**Returns**: `bool` - True if pipeline executed successfully, False otherwise

#### `execute_pipeline_step(step_name: str, target_dir: Path, output_dir: Path, **kwargs) -> StepExecutionResult`
**Description**: Execute a single pipeline step

**Parameters**:
- `step_name` (str): Name of the step to execute
- `target_dir` (Path): Input directory
- `output_dir` (Path): Output directory
- `**kwargs`: Step-specific options

**Returns**: `StepExecutionResult` - Result object with:
- `success` (bool): Whether step succeeded
- `duration` (float): Execution time in seconds
- `output_files` (List[Path]): Generated output files
- `errors` (List[str]): Errors encountered

#### `get_pipeline_status() -> Dict[str, Any]`
**Description**: Get current pipeline execution status

**Returns**: `Dict[str, Any]` - Status information with:
- `current_step` (str): Currently executing step
- `completed_steps` (List[str]): Completed steps
- `failed_steps` (List[str]): Failed steps
- `progress` (float): Completion percentage (0.0-1.0)

### Health Check Functions

#### `run_enhanced_health_check() -> Dict[str, Any]`
**Description**: Run comprehensive pipeline health check

**Returns**: `Dict[str, Any]` - Health check results with:
- `overall_status` (str): "healthy", "degraded", or "unhealthy"
- `component_status` (Dict[str, str]): Status of each component
- `issues` (List[str]): Detected issues
- `recommendations` (List[str]): Recommended actions

---

## Dependencies

### Required Dependencies
- `pathlib` - Path manipulation
- `typing` - Type hints
- `logging` - Logging functionality

### Internal Dependencies
- `utils.argument_utils` - Argument parsing utilities
- `utils.logging_utils` - Enhanced logging utilities
- `utils.pipeline_template` - Pipeline template utilities

---

## Configuration

### Environment Variables
- `PIPELINE_PERFORMANCE_MODE` - Performance optimization level ("low", "medium", "high")
- `PIPELINE_TIMEOUT` - Maximum execution time per step (seconds)
- `PIPELINE_MAX_RETRIES` - Maximum retry attempts for failed steps

### Configuration Files
- `pipeline_config.yaml` - Pipeline-specific configuration
- `step_configs.json` - Step-specific configurations

### Default Settings
```python
DEFAULT_CONFIG = {
    'performance_mode': 'low',
    'timeout_per_step': 300,
    'max_retries': 3,
    'parallel_execution': False,
    'resource_monitoring': True,
    'health_check_interval': 30
}
```

---

## Usage Examples

### Basic Pipeline Configuration
```python
from pipeline.config import get_pipeline_config, get_output_dir_for_script

# Get current configuration
config = get_pipeline_config()
print(f"Output directory: {config['output_dir']}")

# Get output directory for specific step
output_dir = get_output_dir_for_script("3_gnn.py", Path("output"))
print(f"GNN output directory: {output_dir}")
```

### Pipeline Validation
```python
from pipeline.pipeline_validator import validate_step_prerequisites

# Validate step prerequisites
validation = validate_step_prerequisites("3_gnn.py", args, logger)
if not validation["passed"]:
    print("Prerequisites not met:")
    for warning in validation["warnings"]:
        print(f"  - {warning}")
```

### Execution Planning
```python
from pipeline.pipeline_planner import generate_execution_plan

# Generate execution plan
plan = generate_execution_plan(steps_to_execute, args, logger)
print(f"Estimated execution time: {plan['estimated_duration']}s")
print(f"Resource requirements: {plan['resource_requirements']}")
```

---

## Output Specification

### Output Products
- `pipeline_config.yaml` - Pipeline configuration file
- `pipeline_execution_summary.json` - Execution summary
- `pipeline_health_report.json` - Health monitoring report
- `step_execution_reports/` - Individual step reports

### Output Directory Structure
```
output/
├── pipeline_config.yaml
├── pipeline_execution_summary.json
├── pipeline_health_report.json
└── step_execution_reports/
    ├── 0_template_execution.json
    ├── 1_setup_execution.json
    └── ...
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: Variable (depends on pipeline length)
- **Memory**: ~10-50MB for orchestration
- **Status**: ✅ Production Ready

### Expected Performance
- **Orchestration Overhead**: < 5% of total pipeline time
- **Configuration Loading**: < 100ms
- **Step Discovery**: < 500ms
- **Health Monitoring**: < 10ms per check

---

## Error Handling

### Pipeline Errors
1. **Configuration Errors**: Invalid pipeline configuration
2. **Dependency Errors**: Missing step dependencies
3. **Resource Errors**: Insufficient resources for execution
4. **Timeout Errors**: Step execution timeout
5. **Validation Errors**: Invalid step sequence or parameters

### Recovery Strategies
- **Auto-retry**: Automatic retry for transient failures
- **Graceful degradation**: Continue with available steps
- **Resource reallocation**: Adjust resource allocation
- **Configuration repair**: Attempt to fix configuration issues

---

## Integration Points

### Orchestrated By
- **Script**: `main.py` (Main pipeline orchestrator)
- **Function**: Pipeline execution coordination

### Imports From
- `utils.argument_utils` - Argument parsing
- `utils.logging_utils` - Enhanced logging
- `utils.pipeline_template` - Template utilities

### Imported By
- All pipeline scripts (0_template.py through 23_report.py)
- `tests.test_pipeline_*` - Pipeline tests
- `mcp.pipeline_tools` - MCP pipeline tools

### Data Flow
```
Configuration → Step Discovery → Dependency Validation → Execution Planning → Step Execution → Health Monitoring
```

---

## Testing

### Test Files
- `src/tests/test_pipeline_integration.py` - Integration tests
- `src/tests/test_pipeline_functionality.py` - Functionality tests
- `src/tests/test_pipeline_performance.py` - Performance tests

### Test Coverage
- **Current**: 90%
- **Target**: 95%+

### Key Test Scenarios
1. Pipeline configuration validation
2. Step dependency resolution
3. Execution plan generation
4. Health monitoring functionality
5. Error recovery mechanisms

---

## MCP Integration

### Tools Registered
- `pipeline.get_config` - Get pipeline configuration
- `pipeline.validate_steps` - Validate pipeline step sequence
- `pipeline.get_health` - Get pipeline health status
- `pipeline.plan_execution` - Generate execution plan

### Tool Endpoints
```python
@mcp_tool("pipeline.get_config")
def get_pipeline_config_tool():
    """Get current pipeline configuration"""
    # Implementation
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Step discovery fails
**Symptom**: Pipeline can't find or load pipeline steps  
**Cause**: Step scripts missing or naming convention mismatch  
**Solution**: 
- Verify all step scripts exist in `src/` directory
- Check script naming follows pattern `N_module.py`
- Ensure scripts are executable and have correct imports
- Review step discovery logs

#### Issue 2: Configuration validation errors
**Symptom**: Pipeline fails with configuration errors  
**Cause**: Invalid configuration values or missing required settings  
**Solution**:
- Verify configuration file format is valid
- Check all required configuration keys are present
- Review configuration validation logs
- Use default configuration if issues persist

---

## Version History

### Current Version: 2.0.0

**Features**:
- Pipeline orchestration
- Configuration management
- Step discovery and dependency management
- Health monitoring
- Resource estimation

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced dependency analysis
- **Future**: Real-time pipeline monitoring

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [Utils Module](../utils/AGENTS.md)

### External Resources
- [Pipeline Architecture Documentation](../../doc/pipeline/)

---

**Last Updated**: 2026-01-07
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: 2.0.0
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern