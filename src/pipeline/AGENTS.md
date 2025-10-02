# Pipeline Module - Agent Scaffolding

## Module Overview

**Purpose**: Pipeline orchestration, configuration management, and execution coordination for the GNN processing system

**Pipeline Step**: Infrastructure module (not a numbered step)

**Category**: Pipeline Infrastructure / Orchestration

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

### Public Functions

#### `get_pipeline_config() -> Dict[str, Any]`
**Description**: Get the current pipeline configuration

**Returns**: Dictionary containing pipeline configuration parameters

#### `get_output_dir_for_script(script_name: str, base_output_dir: Path) -> Path`
**Description**: Get the output directory for a specific pipeline script

**Parameters**:
- `script_name`: Name of the pipeline script (e.g., "3_gnn.py")
- `base_output_dir`: Base output directory

**Returns**: Path to the script's output directory

#### `validate_step_prerequisites(script_name: str, args, logger) -> Dict[str, Any]`
**Description**: Validate prerequisites for a pipeline step

**Parameters**:
- `script_name`: Name of the script to validate
- `args`: Command line arguments
- `logger`: Logger instance

**Returns**: Dictionary with validation results and warnings

#### `validate_pipeline_step_sequence(steps_to_execute: List[tuple], logger) -> Dict[str, Any]`
**Description**: Validate the sequence of pipeline steps

**Parameters**:
- `steps_to_execute`: List of step tuples to execute
- `logger`: Logger instance

**Returns**: Dictionary with sequence validation results

#### `generate_execution_plan(steps_to_execute: List[tuple], args, logger) -> Dict[str, Any]`
**Description**: Generate execution plan for pipeline steps

**Parameters**:
- `steps_to_execute`: List of steps to execute
- `args`: Pipeline arguments
- `logger`: Logger instance

**Returns**: Dictionary with execution plan and resource estimates

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

**Last Updated**: October 1, 2025
**Status**: ✅ Production Ready