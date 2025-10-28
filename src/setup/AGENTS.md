# Setup Module - Agent Scaffolding

## Module Overview

**Purpose**: Environment setup, dependency management, and system configuration for the GNN processing pipeline

**Pipeline Step**: Step 1: Environment setup (1_setup.py)

**Category**: Environment Management / Dependency Installation

---

## Core Functionality

### Primary Responsibilities
1. Virtual environment creation and management
2. Dependency installation and validation
3. System requirement verification
4. UV (Python package manager) integration
5. Environment configuration and optimization

### Key Capabilities
- Automated virtual environment setup
- Comprehensive dependency management
- System requirement validation
- UV environment optimization
- Dependency conflict resolution
- Environment health monitoring

---

## API Reference

### Public Functions

#### `setup_uv_environment(verbose=False, recreate=False, dev=True, extras=[], skip_jax_test=True, output_dir=None) -> bool`
**Description**: Set up UV virtual environment with dependencies

**Parameters**:
- `verbose`: Enable verbose output
- `recreate`: Recreate existing environment
- `dev`: Install development dependencies
- `extras`: Additional package groups to install
- `skip_jax_test`: Skip JAX functionality test
- `output_dir`: Output directory for setup logs

**Returns**: `True` if setup succeeded

#### `check_system_requirements(verbose=False) -> bool`
**Description**: Check system requirements for GNN pipeline

**Parameters**:
- `verbose`: Enable verbose output

**Returns**: `True` if requirements are met

#### `install_uv_dependencies(verbose=False, dev=False, extras=None) -> bool`
**Description**: Install UV dependencies

**Parameters**:
- `verbose`: Enable verbose output
- `dev`: Install development dependencies
- `extras`: Additional package groups

**Returns**: `True` if installation succeeded

---

## Dependencies

### Required Dependencies
- `uv` - Python package manager
- `python` - Python interpreter
- `pip` - Package installer

### Optional Dependencies
- `virtualenv` - Virtual environment management
- `conda` - Alternative environment manager

### Internal Dependencies
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### Environment Settings
```python
UV_CONFIG = {
    'python_version': '3.11',
    'environment_name': 'gnn-pipeline',
    'dependency_groups': ['core', 'llm', 'visualization', 'audio'],
    'dev_dependencies': True,
    'test_dependencies': True
}
```

### System Requirements
```python
SYSTEM_REQUIREMENTS = {
    'python_version_min': '3.9',
    'memory_min_gb': 4,
    'disk_space_min_gb': 2,
    'cpu_cores_min': 2
}
```

---

## Usage Examples

### Basic Environment Setup
```python
from setup.setup import setup_uv_environment

success = setup_uv_environment(
    verbose=True,
    dev=True,
    extras=["llm", "visualization", "audio"]
)
```

### System Requirements Check
```python
from setup.setup import check_system_requirements

requirements_met = check_system_requirements(verbose=True)
if requirements_met:
    print("System requirements satisfied")
else:
    print("System requirements not met")
```

### Dependency Installation
```python
from setup.setup import install_uv_dependencies

success = install_uv_dependencies(
    dev=True,
    extras=["ml_ai", "gui"]
)
```

---

## Output Specification

### Output Products
- `setup_summary.json` - Setup completion summary
- `environment_info.json` - Environment information
- `dependency_status.json` - Dependency installation status
- `setup_log.txt` - Detailed setup log

### Output Directory Structure
```
output/1_setup_output/
├── setup_summary.json
├── environment_info.json
├── dependency_status.json
├── setup_log.txt
└── environment_details/
    ├── python_version.txt
    └── package_list.txt
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~2-5 minutes for full setup
- **Memory**: ~50-100MB during installation
- **Status**: ✅ Production Ready

### Expected Performance
- **Environment Creation**: 30-60 seconds
- **Dependency Installation**: 1-3 minutes
- **System Validation**: < 30 seconds
- **Health Check**: < 10 seconds

---

## Error Handling

### Setup Errors
1. **Environment Creation**: Virtual environment creation failures
2. **Dependency Installation**: Package installation errors
3. **System Requirements**: Insufficient system resources
4. **Network Issues**: Package download failures
5. **Permission Errors**: Insufficient file permissions

### Recovery Strategies
- **Retry Logic**: Automatic retry for transient failures
- **Alternative Sources**: Use alternative package sources
- **Graceful Degradation**: Continue with available packages
- **Manual Instructions**: Provide manual installation guidance

---

## Integration Points

### Orchestrated By
- **Script**: `1_setup.py` (Step 1)
- **Function**: `setup_uv_environment()`

### Imports From
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `main.py` - Pipeline orchestration
- `tests.test_setup_*` - Setup tests

### Data Flow
```
System Check → Environment Creation → Dependency Installation → Validation → Health Report
```

---

## Testing

### Test Files
- `src/tests/test_setup_integration.py` - Integration tests
- `src/tests/test_setup_validation.py` - Validation tests

### Test Coverage
- **Current**: 90%
- **Target**: 95%+

### Key Test Scenarios
1. Environment creation and setup
2. Dependency installation and validation
3. System requirement verification
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `setup.check_environment` - Check system environment
- `setup.create_environment` - Create UV environment
- `setup.install_dependencies` - Install dependencies
- `setup.validate_setup` - Validate setup completion

### Tool Endpoints
```python
@mcp_tool("setup.check_environment")
def check_environment_tool():
    """Check system environment for GNN pipeline"""
    # Implementation
```

---

**Last Updated: October 28, 2025
**Status**: ✅ Production Ready