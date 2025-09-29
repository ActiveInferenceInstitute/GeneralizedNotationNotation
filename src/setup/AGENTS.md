# Setup Module - Agent Scaffolding

## Module Overview

**Purpose**: Environment setup, UV package manager integration, virtual environment management, and dependency installation

**Pipeline Step**: Step 1: Environment setup (1_setup.py)

**Category**: Core Infrastructure

---

## Core Functionality

### Primary Responsibilities
1. UV environment setup and validation
2. Virtual environment creation and management
3. Dependency installation (core + optional extras)
4. System information logging
5. Project structure creation

### Key Capabilities
- Automatic UV installation if not available
- Virtual environment management with UV
- Optional dependency group installation (llm, visualization, audio, gui)
- System compatibility checking
- Environment validation

---

## API Reference

### Public Functions

#### `setup_uv_environment(verbose=False, recreate=False, dev=True, extras=[], skip_jax_test=True) -> bool`
**Description**: Main UV environment setup function

**Parameters**:
- `verbose` (bool): Enable verbose logging
- `recreate` (bool): Recreate virtual environment
- `dev` (bool): Install dev dependencies
- `extras` (List[str]): Optional dependency groups to install
- `skip_jax_test` (bool): Skip JAX availability test (faster setup)

**Returns**: `True` if setup succeeded

#### `check_uv_availability(logger) -> bool`
**Description**: Check if UV is available and working

**Returns**: `True` if UV is available

#### `validate_uv_setup() -> Dict[str, Any]`
**Description**: Validate UV setup and return validation results

**Returns**: Dictionary with validation status and details

---

## Dependencies

### Required Dependencies
- `subprocess` - Command execution
- `pathlib` - File path manipulation
- `shutil` - File operations

### Optional Dependencies
- None (installs all dependencies for pipeline)

---

## Usage Examples

### Basic Usage
```python
from setup import setup_uv_environment

success = setup_uv_environment(
    verbose=True,
    dev=True,
    extras=["llm", "visualization", "audio", "gui"]
)
```

### Pipeline Integration
```python
# From 1_setup.py
success = setup_uv_environment(
    verbose=verbose,
    recreate=False,
    dev=True,
    extras=["llm", "visualization", "audio", "gui"],
    skip_jax_test=True
)
```

---

## Output Specification

### Output Products
- `environment_setup_summary.json` - Setup results
- `project_structure.yaml` - Created directory structure

### Output Directory Structure
```
output/1_setup_output/
├── environment_setup_summary.json
└── project_structure.yaml
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 1.73s
- **Memory**: 28.9 MB
- **Status**: SUCCESS
- **Exit Code**: 0

---

## Testing

### Test Files
- `src/tests/test_setup_integration.py`
- `src/tests/test_environment_system.py`

### Test Coverage
- **Current**: 90%
- **Target**: 90%+

---

**Last Updated**: September 29, 2025  
**Status**: ✅ Production Ready


