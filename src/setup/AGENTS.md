# Setup Module - Agent Scaffolding

## Module Overview

**Purpose**: Environment setup, dependency management, and system configuration for the GNN processing pipeline

**Pipeline Step**: Step 1: Environment setup (1_setup.py)

**Category**: Environment Management / Dependency Installation

---

## Core Functionality

### Primary Responsibilities
1. Virtual environment creation and management
2. Dependency installation and validation via native UV commands
3. System requirement verification
4. UV (Python package manager) integration
5. Environment configuration and optimization

### Key Capabilities
- Automated virtual environment setup using UV
- Comprehensive dependency management via `uv sync`
- System requirement validation
- UV environment optimization
- Dependency conflict resolution via `uv.lock`
- Environment health monitoring
- Native UV dependency operations (`add`, `remove`, `sync`, `lock`)

---

## API Reference

### Public Functions

#### `setup_uv_environment(verbose=False, recreate=False, dev=True, extras=[], skip_jax_test=True, output_dir=None) -> bool`
**Description**: Set up UV virtual environment with dependencies using native UV sync

**Parameters**:
- `verbose`: Enable verbose output
- `recreate`: Recreate existing environment
- `dev`: Install development dependencies
- `extras`: Additional package groups to install
- `skip_jax_test`: Skip JAX functionality test
- `output_dir`: Output directory for setup logs

**Returns**: `True` if setup succeeded

#### `install_uv_dependencies(verbose=False, dev=False, extras=None) -> bool`
**Description**: Install UV dependencies using `uv sync` from pyproject.toml

**Parameters**:
- `verbose`: Enable verbose output
- `dev`: Install development dependencies
- `extras`: Additional package groups

**Returns**: `True` if installation succeeded

#### `add_uv_dependency(package: str, dev: bool = False, verbose: bool = False) -> bool`
**Description**: Add a dependency using `uv add` command

**Parameters**:
- `package`: Package name with optional version specifier
- `dev`: Add as development dependency
- `verbose`: Enable verbose logging

**Returns**: `True` if successful

#### `remove_uv_dependency(package: str, verbose: bool = False) -> bool`
**Description**: Remove a dependency using `uv remove` command

**Parameters**:
- `package`: Package name to remove
- `verbose`: Enable verbose logging

**Returns**: `True` if successful

#### `update_uv_dependencies(verbose: bool = False, upgrade: bool = False) -> bool`
**Description**: Update dependencies using `uv sync` command

**Parameters**:
- `verbose`: Enable verbose logging
- `upgrade`: Upgrade dependencies to latest compatible versions

**Returns**: `True` if successful

#### `lock_uv_dependencies(verbose: bool = False) -> bool`
**Description**: Update lock file using `uv lock` command

**Parameters**:
- `verbose`: Enable verbose logging

**Returns**: `True` if successful

#### `check_system_requirements(verbose=False) -> bool`
**Description**: Check system requirements for GNN pipeline

**Parameters**:
- `verbose`: Enable verbose output

**Returns**: `True` if requirements are met

---

## Dependencies

### Required Dependencies
- `uv` - Python package manager (required, native commands used)
- `python` - Python interpreter (>=3.9)
- `pyproject.toml` - Project dependencies configuration

### Optional Dependencies
- None (UV handles all dependency management)

### Internal Dependencies
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### Environment Settings
```python
UV_CONFIG = {
    'python_version': '3.11',
    'environment_name': 'gnn-pipeline',
    'dependency_source': 'pyproject.toml',  # Primary source
    'lock_file': 'uv.lock',  # Dependency lock file
    'use_native_uv': True,  # Use native UV commands
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
    'cpu_cores_min': 2,
    'uv_required': True
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

### Add New Dependency
```python
from setup.setup import add_uv_dependency

# Add production dependency
success = add_uv_dependency("requests>=2.28.0", dev=False, verbose=True)

# Add development dependency
success = add_uv_dependency("pytest>=7.0.0", dev=True, verbose=True)
```

### Remove Dependency
```python
from setup.setup import remove_uv_dependency

success = remove_uv_dependency("old-package", verbose=True)
```

### Update Dependencies
```python
from setup.setup import update_uv_dependencies

# Sync with lock file
success = update_uv_dependencies(verbose=True, upgrade=False)

# Upgrade to latest compatible versions
success = update_uv_dependencies(verbose=True, upgrade=True)
```

### Lock Dependencies
```python
from setup.setup import lock_uv_dependencies

# Update uv.lock file
success = lock_uv_dependencies(verbose=True)
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

---

## Output Specification

### Output Products
- `setup_summary.json` - Setup completion summary
- `environment_info.json` - Environment information
- `dependency_status.json` - Dependency installation status
- `setup_log.txt` - Detailed setup log
- `uv.lock` - Dependency lock file (updated)

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
- **Dependency Installation**: 1-3 minutes (via `uv sync`)
- **System Validation**: < 30 seconds
- **Health Check**: < 10 seconds
- **Dependency Lock**: < 10 seconds

---

## Error Handling

### Setup Errors
1. **Environment Creation**: Virtual environment creation failures
2. **Dependency Installation**: Package installation errors via `uv sync`
3. **System Requirements**: Insufficient system resources or missing UV
4. **Network Issues**: Package download failures
5. **Permission Errors**: Insufficient file permissions
6. **Lock File Conflicts**: Lock file corruption or conflicts

### Recovery Strategies
- **Retry Logic**: Automatic retry for transient failures
- **Lock File Regeneration**: Regenerate uv.lock if corrupted
- **Graceful Degradation**: Continue with available packages
- **Manual Instructions**: Provide manual installation guidance
- **Environment Recreate**: Option to recreate environment from scratch

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
System Check → UV Environment Creation → UV Sync (pyproject.toml → uv.lock) → Validation → Health Report
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
2. Dependency installation via UV sync
3. System requirement verification
4. Native UV command operations
5. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `setup.check_environment` - Check system environment
- `setup.create_environment` - Create UV environment
- `setup.install_dependencies` - Install dependencies via UV sync
- `setup.validate_setup` - Validate setup completion
- `setup.add_dependency` - Add dependency via UV add
- `setup.remove_dependency` - Remove dependency via UV remove
- `setup.update_dependencies` - Update dependencies via UV sync
- `setup.lock_dependencies` - Update lock file via UV lock

### Tool Endpoints
```python
@mcp_tool("setup.check_environment")
def check_environment_tool():
    """Check system environment for GNN pipeline"""
    # Implementation

@mcp_tool("setup.add_dependency")
def add_dependency_tool(package: str, dev: bool = False):
    """Add a dependency using UV add"""
    # Implementation
```

---

**Last Updated**: October 28, 2025  
**Status**: ✅ Production Ready  
**UV Integration**: ✅ 100% Native UV Commands (sync, add, remove, lock)