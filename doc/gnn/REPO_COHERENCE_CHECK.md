# Repo-Wide Coherence and Improvement Check - Mega-Prompt

## Pipeline Architecture References

This document validates the complete GNN pipeline architecture. For current implementation:
- **[src/AGENTS.md](../../src/AGENTS.md)**: Master agent scaffolding and complete 24-step pipeline registry
- **[src/README.md](../../src/README.md)**: Pipeline architecture and thin orchestrator pattern
- **[src/main.py](../../src/main.py)**: Pipeline orchestrator implementation (24 steps: 0-23)
- [Architecture Reference](architecture_reference.md): Implementation patterns and cross-module data flow
- [Technical Reference](technical_reference.md): Complete entry points and round-trip data flow

- ✅ 24 steps (0-23): All operational with 100% success rate
- ✅ Execution time: ~2 minutes for full pipeline
- ✅ Memory usage: < 25MB peak
- ✅ All modules use thin orchestrator pattern
- ✅ 28/28 modules have AGENTS.md documentation

---

## Executive Summary

This document serves as a comprehensive checklist and analysis framework for validating the entire GNN (Generalized Notation Notation) codebase against established standards, patterns, and best practices. It covers all 24 pipeline steps (0-23) and 28 specialized modules, ensuring architectural compliance, code quality, documentation completeness, and integration consistency.

### High-Level Coherence Metrics

- **Total Pipeline Steps**: 24 (0_template.py through 23_report.py)
- **Total Modules**: 28 specialized agent modules
- **AGENTS.md Files**: 41 documentation files (including sub-modules)
- **Architecture Pattern**: Thin orchestrator with modular implementation
- **Testing Policy**: No mocks - real data and real code paths only (some legacy tests still use mocks - see Section 16.1)
- **Documentation Standard**: AGENTS.md + README.md for each module
- **Output Directory Pattern**: `N_[module]_output/` for all steps (enforced via `get_output_dir_for_script()`)

### Validation Scope

This mega-prompt validates:
1. Architecture compliance across all 24 pipeline steps
2. Code quality standards (type hints, docstrings, error handling)
3. Module structure consistency (directory patterns, __init__.py, MCP integration)
4. Documentation completeness (AGENTS.md, README.md, API docs)
5. Testing standards (no-mock policy, real data, integration tests)
6. Pipeline integration (dependencies, data flow, output consistency)
7. Performance standards (execution time, memory usage, resource efficiency)
8. Security and validation patterns
9. Naming conventions consistency
10. Dependency management and module coupling

---

## 1. Architecture Compliance

### 1.1 Thin Orchestrator Pattern Validation

**Requirement**: All 24 numbered pipeline scripts must follow the thin orchestrator pattern, delegating core functionality to module implementations.

#### Validation Checklist for Each Pipeline Script (0-23)

For each script `N_[module_name].py`, verify:

- [ ] **File Length**: Script is < 150 lines (thin orchestrator requirement)
- [ ] **Import Pattern**: Imports from `utils` and `pipeline` modules, not inline implementations
- [ ] **Logging Setup**: Uses `setup_step_logging()` or `setup_main_logging()` (for main.py)
- [ ] **Argument Parsing**: Uses `EnhancedArgumentParser.parse_step_arguments()` or `create_standardized_pipeline_script()`
- [ ] **Core Logic Delegation**: All domain logic delegated to module functions (e.g., `process_render()`, `process_validation()`)
- [ ] **No Inline Implementations**: No function definitions > 20 lines in numbered scripts
- [ ] **Exit Codes**: Returns proper exit codes (0=success, 1=error, 2=warnings)
- [ ] **Output Directory**: Uses `get_output_dir_for_script()` for centralized output management
- [ ] **Error Handling**: Uses `log_step_error()`, `log_step_warning()` for structured error reporting
- [ ] **Visual Logging**: Uses visual logging utilities where appropriate

#### Specific Pipeline Scripts to Validate

**Step 0**: `src/0_template.py` → `src/template/`
- Verify: Uses `create_standardized_pipeline_script()` pattern
- Verify: Delegates to `process_template_standardized()`

**Step 1**: `src/1_setup.py` → `src/setup/`
- Verify: Environment setup delegation
- Verify: Dependency management patterns

**Step 2**: `src/2_tests.py` → `src/tests/`
- Verify: Test orchestration delegation
- Verify: Real test execution (no mocks)

**Step 3**: `src/3_gnn.py` → `src/gnn/`
- Verify: GNN file discovery and parsing delegation
- Verify: Multi-format support delegation
- Verify: Delegates to `process_gnn_multi_format()` from `gnn.multi_format_processor`

**Step 4**: `src/4_model_registry.py` → `src/model_registry/`
- Verify: Registry management delegation
- Verify: Versioning and metadata handling

**Step 5**: `src/5_type_checker.py` → `src/type_checker/`
- Verify: Type checking delegation
- Verify: Resource estimation delegation

**Step 6**: `src/6_validation.py` → `src/validation/`
- Verify: Validation logic delegation
- Verify: Consistency checking delegation

**Step 7**: `src/7_export.py` → `src/export/`
- Verify: Multi-format export delegation
- Verify: Format-specific exporters

**Step 8**: `src/8_visualization.py` → `src/visualization/`
- Verify: Visualization generation delegation
- Verify: Graph and matrix visualization
- Verify: Delegates to `process_visualization_main()` from `visualization` module

**Step 9**: `src/9_advanced_viz.py` → `src/advanced_visualization/`
- Verify: Advanced visualization delegation
- Verify: Interactive plot generation

**Step 10**: `src/10_ontology.py` → `src/ontology/`
- Verify: Ontology processing delegation
- Verify: Active Inference term mapping

**Step 11**: `src/11_render.py` → `src/render/`
- Verify: Code generation delegation
- Verify: Multi-framework rendering (PyMDP, RxInfer, ActiveInference.jl)

**Step 12**: `src/12_execute.py` → `src/execute/`
- Verify: Execution orchestration delegation
- Verify: Multi-environment execution

**Step 13**: `src/13_llm.py` → `src/llm/`
- Verify: LLM processing delegation
- Verify: AI-enhanced analysis

**Step 14**: `src/14_ml_integration.py` → `src/ml_integration/`
- Verify: ML integration delegation
- Verify: Model training and evaluation

**Step 15**: `src/15_audio.py` → `src/audio/`
- Verify: Audio generation delegation
- Verify: Multi-backend audio (SAPF, Pedalboard)

**Step 16**: `src/16_analysis.py` → `src/analysis/`
- Verify: Statistical analysis delegation
- Verify: Performance metric computation
- Verify: Delegates to `process_analysis()` from `analysis` module

**Step 17**: `src/17_integration.py` → `src/integration/`
- Verify: System integration delegation
- Verify: Cross-module coordination

**Step 18**: `src/18_security.py` → `src/security/`
- Verify: Security validation delegation
- Verify: Access control implementation

**Step 19**: `src/19_research.py` → `src/research/`
- Verify: Research tools delegation
- Verify: Experimental features

**Step 20**: `src/20_website.py` → `src/website/`
- Verify: Website generation delegation
- Verify: Static HTML generation

**Step 21**: `src/21_mcp.py` → `src/mcp/`
- Verify: MCP processing delegation
- Verify: Tool registration

**Step 22**: `src/22_gui.py` → `src/gui/`
- Verify: GUI generation delegation
- Verify: Interactive model construction

**Step 23**: `src/23_report.py` → `src/report/`
- Verify: Report generation delegation
- Verify: Comprehensive analysis reports

#### Anti-Patterns to Identify

**Incorrect Pattern Examples**:
```python
# ❌ WRONG: Long function definition in numbered script
def generate_matrix_heatmap(data):
    # 50+ lines of visualization code
    ...
```

**Correct Pattern Examples**:
```python
# ✅ CORRECT: Thin orchestrator delegating to module
from utils.pipeline_template import create_standardized_pipeline_script
from visualization import process_visualization_main

run_script = create_standardized_pipeline_script(
    "8_visualization.py",
    process_visualization_main,
    "Matrix and network visualization processing"
)

def main() -> int:
    """Main entry point."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
```

### 1.2 Centralized Utilities Usage

**Requirement**: All pipeline scripts must use centralized utilities from `src/utils/`.

#### Validation Checklist

- [ ] **Logging Utilities**: Uses `setup_step_logging()`, `log_step_start()`, `log_step_success()`, `log_step_error()`, `log_step_warning()`
- [ ] **Argument Parsing**: Uses `EnhancedArgumentParser` or `create_standardized_pipeline_script()`
- [ ] **Path Management**: Uses `pathlib.Path` objects, not string paths
- [ ] **Output Directory**: Uses `get_output_dir_for_script()` from `pipeline.config`
- [ ] **Configuration**: Uses `get_pipeline_config()` for centralized configuration
- [ ] **Visual Logging**: Uses `utils.visual_logging` for progress indicators and status messages
- [ ] **Error Recovery**: Uses `utils.error_recovery` for graceful degradation
- [ ] **Resource Management**: Uses `utils.resource_manager` for memory tracking

#### Import Pattern Validation

**Correct Import Pattern**:
```python
from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error
)
from utils.argument_utils import ArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config
```

**Incorrect Import Pattern**:
```python
# ❌ WRONG: Direct logging setup without centralized utilities
import logging
logger = logging.getLogger(__name__)
```

### 1.3 Exit Code Patterns

**Requirement**: All pipeline scripts must return standardized exit codes.

#### Exit Code Standards

- **0**: Success - step completed without errors
- **1**: Critical Error - step failed and cannot continue
- **2**: Success with Warnings - step completed but with non-critical warnings

#### Validation Checklist

- [ ] **Main Function**: Returns integer exit code
- [ ] **Error Handling**: Returns 1 on critical errors
- [ ] **Warning Handling**: Returns 2 on success with warnings
- [ ] **Success Handling**: Returns 0 on complete success
- [ ] **Exit Statement**: Uses `sys.exit(main())` pattern

---

## 2. Code Quality Standards

### 2.1 Type Hints Coverage

**Requirement**: All public functions must have complete type hints with generic types for containers.

#### Validation Checklist

- [ ] **Function Signatures**: All public functions have type hints
- [ ] **Return Types**: All functions specify return types (including `None`)
- [ ] **Parameter Types**: All parameters have type annotations
- [ ] **Generic Types**: Container types use generics (e.g., `List[str]`, `Dict[str, Any]`)
- [ ] **Union Types**: Optional parameters use `Optional[T]` or `T | None`
- [ ] **Type Imports**: Proper imports from `typing` module

#### Example Correct Pattern

```python
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

def process_validation(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs: Any
) -> bool:
    """Process validation for GNN models."""
    ...
```

### 2.2 Docstring Standards

**Requirement**: Every public function/class must have comprehensive docstrings with examples.

#### Validation Checklist

- [ ] **Module Docstrings**: All modules have module-level docstrings
- [ ] **Class Docstrings**: All classes have class-level docstrings
- [ ] **Function Docstrings**: All public functions have docstrings
- [ ] **Parameter Documentation**: Parameters documented in docstrings
- [ ] **Return Documentation**: Return values documented
- [ ] **Example Usage**: Examples provided for complex functions
- [ ] **Raises Documentation**: Exceptions documented where applicable

#### Docstring Format

```python
def process_validation(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs: Any
) -> bool:
    """
    Process validation for GNN models.
    
    Args:
        target_dir: Directory containing GNN files to validate
        output_dir: Output directory for validation results
        verbose: Enable verbose logging
        **kwargs: Additional validation options
        
    Returns:
        True if validation succeeded, False otherwise
        
    Raises:
        ValueError: If target_dir does not exist
        PermissionError: If output_dir is not writable
        
    Example:
        >>> from pathlib import Path
        >>> result = process_validation(
        ...     Path("input/gnn_files"),
        ...     Path("output"),
        ...     verbose=True
        ... )
        >>> assert result is True
    """
    ...
```

### 2.3 Error Handling Patterns

**Requirement**: Comprehensive error handling with graceful degradation and recovery mechanisms.

#### Validation Checklist

- [ ] **Try-Except Blocks**: Critical operations wrapped in try-except
- [ ] **Specific Exceptions**: Catch specific exceptions, not bare `except:`
- [ ] **Error Logging**: Errors logged using `log_step_error()`
- [ ] **Recovery Mechanisms**: Graceful degradation when possible
- [ ] **Error Messages**: Clear, actionable error messages
- [ ] **Resource Cleanup**: Resources cleaned up in finally blocks
- [ ] **Error Propagation**: Appropriate error propagation to callers

#### Error Handling Pattern

```python
def process_with_error_handling(target_dir: Path, output_dir: Path) -> bool:
    """Process with comprehensive error handling."""
    try:
        # Critical operation
        result = perform_operation(target_dir, output_dir)
        return result
    except FileNotFoundError as e:
        log_step_error(logger, f"File not found: {e}")
        return False
    except PermissionError as e:
        log_step_error(logger, f"Permission denied: {e}")
        return False
    except Exception as e:
        log_step_error(logger, f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        # Resource cleanup
        cleanup_resources()
```

### 2.4 Resource Management

**Requirement**: Proper cleanup of resources (files, connections, memory) in all code paths.

#### Validation Checklist

- [ ] **File Handles**: Files closed properly (use context managers)
- [ ] **Memory Management**: Large objects cleaned up when no longer needed
- [ ] **Connection Management**: Network connections closed properly
- [ ] **Context Managers**: Use `with` statements for resource management
- [ ] **Memory Tracking**: Use `utils.resource_manager` for memory monitoring

#### Resource Management Pattern

```python
from pathlib import Path
from utils.resource_manager import get_current_memory_usage

def process_with_resource_management(target_dir: Path) -> bool:
    """Process with proper resource management."""
    initial_memory = get_current_memory_usage()
    
    try:
        # Use context managers for file operations
        with open(target_dir / "data.json", "r") as f:
            data = json.load(f)
        
        # Process data
        result = process_data(data)
        
        return result
    finally:
        # Cleanup
        final_memory = get_current_memory_usage()
        memory_delta = final_memory - initial_memory
        if memory_delta > 100 * 1024 * 1024:  # 100MB threshold
            logger.warning(f"High memory usage: {memory_delta} bytes")
```

### 2.5 Performance Monitoring

**Requirement**: Built-in timing and memory usage tracking for all major operations.

#### Validation Checklist

- [ ] **Timing**: Operations timed using `time.time()` or `time.perf_counter()`
- [ ] **Memory Tracking**: Memory usage tracked using `utils.resource_manager`
- [ ] **Performance Logging**: Performance metrics logged appropriately
- [ ] **Performance Reports**: Performance data included in output reports

---

## 3. Module Structure Consistency

### 3.1 Directory Structure Validation

**Requirement**: All modules follow the established directory structure pattern.

#### Standard Module Structure

```
src/[module_name]/
├── __init__.py          # Public API exports
├── AGENTS.md            # Module documentation
├── README.md            # Module README (optional but recommended)
├── processor.py         # Core processing logic
├── [module_specific].py # Module-specific functionality
├── mcp.py               # MCP tool registration (where applicable)
└── [submodules]/        # Sub-modules if needed
```

#### Validation Checklist for Each Module

- [ ] **Directory Exists**: Module directory exists in `src/`
- [ ] **__init__.py Present**: Module has `__init__.py` file
- [ ] **AGENTS.md Present**: Module has `AGENTS.md` documentation
- [ ] **README.md Present**: Module has `README.md` (recommended)
- [ ] **Core Logic Files**: Core logic in separate files (not in numbered scripts)
- [ ] **MCP Integration**: `mcp.py` present where applicable
- [ ] **Sub-modules**: Sub-modules properly organized

### 3.2 __init__.py Public API Exports

**Requirement**: Module `__init__.py` files must export public API functions.

#### Validation Checklist

- [ ] **Public Functions Exported**: Main processing functions exported
- [ ] **Import Pattern**: Uses `from .processor import process_module`
- [ ] **API Clarity**: Clear public API with well-named functions
- [ ] **Module Info**: Module information functions exported (e.g., `get_module_info()`)

#### Example __init__.py Pattern

```python
"""
[Module Name] Module

[Brief description of module purpose]
"""

from pathlib import Path
from typing import Dict, Any, Optional

from .processor import (
    process_module,
    process_module_standardized,
    get_module_info
)

__all__ = [
    "process_module",
    "process_module_standardized",
    "get_module_info"
]

def get_module_info() -> Dict[str, Any]:
    """Get module information."""
    return {
        "name": "[module_name]",
        "version": "1.0.0",
        "description": "[Module description]",
        "capabilities": [
            "Capability 1",
            "Capability 2"
        ]
    }
```

### 3.3 MCP Integration

**Requirement**: Modules that provide MCP tools must have `mcp.py` with functional tool registration.

#### Validation Checklist

- [ ] **mcp.py Exists**: Module has `mcp.py` file (where applicable)
- [ ] **Tool Registration**: Tools properly registered
- [ ] **Tool Implementation**: Tools have real implementations (no stubs)
- [ ] **Tool Documentation**: Tools documented in AGENTS.md

#### MCP Integration Pattern

```python
"""
MCP Tool Registration for [Module Name]
"""

from typing import Dict, Any, List
from mcp import Server, Tool

def register_tools(server: Server) -> None:
    """Register MCP tools for this module."""
    
    @server.tool()
    def process_module_tool(target_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Process [module] with MCP tool.
        
        Args:
            target_dir: Directory containing input files
            output_dir: Output directory for results
            
        Returns:
            Processing results
        """
        from pathlib import Path
        from .processor import process_module
        
        result = process_module(
            Path(target_dir),
            Path(output_dir)
        )
        return result
```

### 3.4 Module Separation

**Requirement**: Clear separation between numbered scripts and module implementations.

#### Validation Checklist

- [ ] **No Core Logic in Scripts**: Numbered scripts contain no core logic
- [ ] **Delegation Pattern**: Scripts delegate to module functions
- [ ] **Module Independence**: Modules can be imported independently
- [ ] **Testability**: Modules can be tested without running pipeline scripts

---

## 4. Documentation Completeness

### 4.1 AGENTS.md Documentation

**Requirement**: Every module must have comprehensive AGENTS.md documentation.

#### AGENTS.md Structure Validation

Each AGENTS.md should include:

- [ ] **Module Overview**: Purpose and role in pipeline
- [ ] **Pipeline Step**: Which numbered script uses this module
- [ ] **Category**: Module category (e.g., Validation, Visualization)
- [ ] **Core Functionality**: Primary responsibilities
- [ ] **Key Capabilities**: List of key capabilities
- [ ] **API Reference**: Public functions documented
- [ ] **Dependencies**: Required and optional dependencies
- [ ] **Configuration**: Configuration options
- [ ] **Usage Examples**: Real examples with actual GNN files
- [ ] **Integration Points**: How module integrates with others
- [ ] **MCP Tools**: MCP tools provided (if applicable)

#### AGENTS.md Template

```markdown
# [Module Name] Module - Agent Scaffolding

## Module Overview

**Purpose**: [Module purpose]

**Pipeline Step**: Step N: [Step Name] (N_[module].py)

**Category**: [Category]

---

## Core Functionality

### Primary Responsibilities
1. [Responsibility 1]
2. [Responsibility 2]

### Key Capabilities
- [Capability 1]
- [Capability 2]

---

## API Reference

### Public Functions

#### `process_module(target_dir, output_dir, **kwargs) -> bool`
**Description**: [Function description]

**Parameters**:
- `target_dir`: [Parameter description]
- `output_dir`: [Parameter description]
- `**kwargs`: [Additional options]

**Returns**: [Return description]

---

## Dependencies

### Required Dependencies
- [Dependency 1]
- [Dependency 2]

### Internal Dependencies
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

[Configuration options]

---

## Usage Examples

[Real examples with actual GNN files]
```

### 4.2 README.md Presence

**Requirement**: Modules should have README.md files for additional documentation.

#### Validation Checklist

- [ ] **README.md Exists**: Module has README.md (recommended)
- [ ] **Module Description**: README describes module purpose
- [ ] **Usage Instructions**: Usage instructions provided
- [ ] **Examples**: Examples included
- [ ] **Troubleshooting**: Troubleshooting guide (if applicable)

### 4.3 API Documentation Consistency

**Requirement**: API documentation must be consistent across modules.

#### Validation Checklist

- [ ] **Function Signatures**: Docstrings match function signatures
- [ ] **Parameter Names**: Parameter names consistent across modules
- [ ] **Return Types**: Return types documented consistently
- [ ] **Example Format**: Examples follow consistent format
- [ ] **Error Documentation**: Errors documented consistently

---

## 5. Testing Standards

### 5.1 No-Mock Policy Compliance

**Requirement**: All tests must execute real code paths and real methods. No mocking frameworks.

#### Validation Checklist

- [ ] **No unittest.mock**: No `unittest.mock` imports or usage
- [ ] **No Monkeypatching**: No monkeypatching of functions or classes
- [ ] **Real Code Paths**: Tests execute actual code paths
- [ ] **Real Data**: Tests use real, representative data
- [ ] **Real Dependencies**: Tests use real dependencies (skip if unavailable)
- [ ] **File-Based Assertions**: Tests assert on real file outputs

#### Known Issues

**Current Status**: All strict no-mock policies are enforced. Previous legacy tests using `unittest.mock` have been refactored or removed.

**Policy**: Tests must execute real code paths and real methods. No mocking frameworks are allowed.


#### Anti-Pattern Detection

**Incorrect Pattern**:
```python
# ❌ WRONG: Using mocks
from unittest.mock import Mock, patch

@patch('module.process_function')
def test_processing(mock_process):
    mock_process.return_value = True
    ...
```

**Correct Pattern**:
```python
# ✅ CORRECT: Real code execution
def test_processing():
    from pathlib import Path
    from module import process_function
    
    result = process_function(
        Path("test_data/input"),
        Path("test_data/output")
    )
    assert result is True
    assert (Path("test_data/output") / "result.json").exists()
```

### 5.2 Real Data Processing

**Requirement**: Tests must use real, representative data - no synthetic or placeholder datasets.

#### Validation Checklist

- [ ] **Real GNN Files**: Tests use actual GNN files from test data
- [ ] **Real Formats**: Tests cover all supported formats (Markdown, JSON, YAML, etc.)
- [ ] **Real Outputs**: Tests validate real output files
- [ ] **Real API Calls**: Tests make real API calls (skip if unavailable)
- [ ] **Real File I/O**: Tests perform real file operations

### 5.3 Integration Test Coverage

**Requirement**: End-to-end tests must validate complete pipeline execution with real inputs/outputs.

#### Validation Checklist

- [ ] **Pipeline Tests**: Tests for full pipeline execution
- [ ] **Step Integration**: Tests for step-to-step integration
- [ ] **Cross-Module Tests**: Tests for cross-module functionality
- [ ] **Output Validation**: Tests validate actual output artifacts
- [ ] **Error Scenarios**: Tests for real error conditions

#### Integration Test Pattern

```python
def test_pipeline_integration():
    """Test full pipeline execution with real data."""
    from pathlib import Path
    import subprocess
    
    # Run actual pipeline script
    result = subprocess.run(
        ["python", "src/main.py", "--target-dir", "test_data/gnn_files"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    
    # Validate actual outputs using standardized output directory names
    output_dir = Path("output")
    assert (output_dir / "3_gnn_output" / "parsed_models.json").exists()
    assert (output_dir / "7_export_output" / "exports").exists()
    
    # Verify output directory naming follows convention
    assert output_dir.name == "output"
    assert (output_dir / "3_gnn_output").exists()
    assert (output_dir / "7_export_output").exists()
```

### 5.4 Performance and Regression Tests

**Requirement**: Include timing assertions for critical paths, memory usage validation.

#### Validation Checklist

- [ ] **Timing Assertions**: Critical paths have timing assertions
- [ ] **Memory Validation**: Memory usage validated
- [ ] **Performance Baselines**: Performance baselines established
- [ ] **Regression Detection**: Tests detect performance regressions

---

## 6. Pipeline Integration

### 6.1 Step Dependencies

**Requirement**: Pipeline steps must have correct dependency relationships.

#### Validation Checklist

- [ ] **Dependency Declaration**: Dependencies declared in configuration
- [ ] **Dependency Validation**: Dependencies validated before step execution
- [ ] **Circular Dependency Check**: No circular dependencies
- [ ] **Missing Dependency Detection**: Missing dependencies detected

#### Expected Dependency Graph

```
0_template → (no dependencies)
1_setup → (no dependencies)
2_tests → 1_setup
3_gnn → 1_setup
4_model_registry → 3_gnn
5_type_checker → 3_gnn
6_validation → 3_gnn, 5_type_checker
7_export → 3_gnn
8_visualization → 3_gnn, 7_export
9_advanced_viz → 3_gnn, 8_visualization
10_ontology → 3_gnn
11_render → 3_gnn, 10_ontology
12_execute → 11_render
13_llm → 3_gnn
14_ml_integration → 3_gnn, 12_execute
15_audio → 3_gnn
16_analysis → 3_gnn, 12_execute
17_integration → (multiple dependencies)
18_security → 3_gnn
19_research → 3_gnn
20_website → (multiple dependencies)
21_mcp → 3_gnn
22_gui → 3_gnn
23_report → (multiple dependencies)
```

### 6.2 Data Flow Validation

**Requirement**: Data flows correctly between pipeline steps.

#### Validation Checklist

- [ ] **Input Validation**: Steps validate inputs from previous steps
- [ ] **Output Format**: Steps produce expected output formats
- [ ] **Data Transformation**: Data transformations are correct
- [ ] **Semantic Preservation**: Semantic information preserved across steps

### 6.3 Output Consistency

**Requirement**: Output formats and structures must be consistent.

#### Validation Checklist

- [ ] **Output Directory Structure**: Output directories follow conventions
- [ ] **Output File Naming**: Output files follow naming conventions
- [ ] **Output Format**: Output formats are consistent
- [ ] **Metadata**: Output metadata is consistent

#### Output Directory Conventions

**Enforcement**: Output directory names are enforced via `get_output_dir_for_script()` in `src/pipeline/config.py`. This function maps script names to standardized output directories.

**Standard Pattern**: `N_[module]_output/` where N is the step number and [module] is the module name.

```
output/
├── 0_template_output/
├── 1_setup_output/
├── 2_tests_output/
├── 3_gnn_output/
├── 4_model_registry_output/
├── 5_type_checker_output/
├── 6_validation_output/
├── 7_export_output/
├── 8_visualization_output/
├── 9_advanced_viz_output/
├── 10_ontology_output/
├── 11_render_output/
├── 12_execute_output/
├── 13_llm_output/
├── 14_ml_integration_output/
├── 15_audio_output/
├── 16_analysis_output/
├── 17_integration_output/
├── 18_security_output/
├── 19_research_output/
├── 20_website_output/
├── 21_mcp_output/
├── 22_gui_output/
└── 23_report_output/
```

**Validation**: All pipeline scripts should use `get_output_dir_for_script()` from `pipeline.config` to ensure consistent output directory naming.

---

## 7. Performance Standards

### 7.1 Execution Time

**Requirement**: Full pipeline execution must complete within 30 minutes for standard workloads.

#### Validation Checklist

- [ ] **Pipeline Timing**: Full pipeline execution time < 30 minutes
- [ ] **Step Timing**: Individual step timing tracked
- [ ] **Performance Logging**: Performance metrics logged
- [ ] **Performance Reports**: Performance data in output reports

### 7.2 Memory Usage

**Requirement**: Peak memory usage must not exceed 2GB for standard workloads.

#### Validation Checklist

- [ ] **Memory Tracking**: Memory usage tracked per step
- [ ] **Memory Limits**: Memory usage within 2GB limit
- [ ] **Memory Cleanup**: Memory cleaned up between steps
- [ ] **Memory Reports**: Memory usage in performance reports

### 7.3 Success and Error Rates

**Requirement**: 
- Success rate > 99% for all pipeline steps
- Critical failure rate < 1%

#### Validation Checklist

- [ ] **Success Rate Tracking**: Success rates tracked per step
- [ ] **Error Rate Tracking**: Error rates tracked per step
- [ ] **Error Recovery**: Error recovery mechanisms in place
- [ ] **Error Reporting**: Error rates in performance reports

---

## 8. Security and Validation

### 8.1 Input Validation

**Requirement**: All inputs must be validated and sanitized.

#### Validation Checklist

- [ ] **Path Validation**: File paths validated for security
- [ ] **Input Sanitization**: User inputs sanitized
- [ ] **Type Validation**: Input types validated
- [ ] **Range Validation**: Input ranges validated

### 8.2 Error Recovery

**Requirement**: Comprehensive error recovery mechanisms.

#### Validation Checklist

- [ ] **Recovery Strategies**: Error recovery strategies defined
- [ ] **Graceful Degradation**: Graceful degradation implemented
- [ ] **Retry Logic**: Retry logic for transient errors
- [ ] **Error Reporting**: Clear error reporting with recovery suggestions

### 8.3 Security Patterns

**Requirement**: Security patterns implemented where applicable.

#### Validation Checklist

- [ ] **Access Control**: Access control where applicable
- [ ] **Secure Configuration**: Configuration managed securely
- [ ] **Threat Detection**: Threat detection patterns (if applicable)
- [ ] **Security Validation**: Security validation in security module

---

## 9. Naming Conventions

### 9.1 Module Naming

**Requirement**: Consistent naming across modules, functions, and classes.

#### Validation Checklist

- [ ] **Module Names**: Module names follow `snake_case`
- [ ] **Function Names**: Function names follow `snake_case`
- [ ] **Class Names**: Class names follow `PascalCase`
- [ ] **Constant Names**: Constants follow `UPPER_SNAKE_CASE`

### 9.2 File Naming

**Requirement**: File names follow conventions.

#### Validation Checklist

- [ ] **Pipeline Scripts**: Numbered scripts follow `N_[module].py` pattern
- [ ] **Module Files**: Module files follow `snake_case.py` pattern
- [ ] **Test Files**: Test files follow `test_[module].py` pattern

### 9.3 Output Naming

**Requirement**: Output files and directories follow naming conventions.

#### Validation Checklist

- [ ] **Output Directories**: Follow `N_[module]_output/` pattern
- [ ] **Output Files**: Output files have descriptive names
- [ ] **File Extensions**: File extensions match content types

---

## 10. Dependency Management

### 10.1 Import Patterns

**Requirement**: Consistent import patterns across the codebase.

#### Validation Checklist

- [ ] **Standard Library First**: Standard library imports first
- [ ] **Third-Party Second**: Third-party imports second
- [ ] **Local Imports Last**: Local imports last
- [ ] **Import Grouping**: Imports grouped appropriately
- [ ] **Absolute Imports**: Use absolute imports from `src/`

#### Import Pattern

```python
# Standard library
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

# Third-party
import numpy as np
import pandas as pd

# Local imports
from utils.pipeline_template import setup_step_logging
from pipeline.config import get_output_dir_for_script
from module.processor import process_module
```

### 10.2 Module Coupling

**Requirement**: Modules should minimize dependencies on other modules.

#### Validation Checklist

- [ ] **Low Coupling**: Modules have minimal dependencies
- [ ] **Dependency Injection**: Dependency injection used where appropriate
- [ ] **Interface Segregation**: Large interfaces broken into smaller ones
- [ ] **Circular Dependency Check**: No circular dependencies

### 10.3 Dependency Validation

**Requirement**: Dependencies must be properly declared and validated.

#### Validation Checklist

- [ ] **Dependency Declaration**: Dependencies declared in `pyproject.toml`
- [ ] **Dependency Validation**: Dependencies validated at runtime
- [ ] **Graceful Degradation**: Graceful degradation when dependencies unavailable
- [ ] **Optional Dependencies**: Optional dependencies handled appropriately

---

## 11. Improvement Recommendations Framework

### 11.1 Current Status Assessment

For each validation area, assess:

- **Current Status**: What exists and what's missing
- **Compliance Score**: Percentage of compliance (0-100%)
- **Critical Issues**: Must-fix items (blocking issues)
- **Improvement Opportunities**: Nice-to-have enhancements
- **Action Items**: Specific tasks with file paths and line numbers

### 11.2 Compliance Scoring

#### Scoring Methodology

- **100%**: All validation criteria met
- **90-99%**: Minor improvements needed
- **80-89%**: Some improvements needed
- **70-79%**: Significant improvements needed
- **<70%**: Major refactoring required

#### Scoring Categories

1. **Architecture Compliance**: Thin orchestrator pattern, centralized utilities
2. **Code Quality**: Type hints, docstrings, error handling
3. **Module Structure**: Directory structure, __init__.py, MCP integration
4. **Documentation**: AGENTS.md, README.md, API documentation
5. **Testing**: No-mock policy, real data, integration tests
6. **Pipeline Integration**: Dependencies, data flow, output consistency
7. **Performance**: Execution time, memory usage, success rates
8. **Security**: Input validation, error recovery, security patterns
9. **Naming Conventions**: Consistent naming across codebase
10. **Dependency Management**: Import patterns, module coupling

### 11.3 Action Item Format

Each action item should include:

- **File Path**: Specific file to modify
- **Line Numbers**: Specific lines (if applicable)
- **Issue Description**: What needs to be fixed
- **Recommended Fix**: How to fix it
- **Priority**: Critical, High, Medium, Low

#### Action Item Example

```markdown
### Action Item: Add Type Hints to process_validation

- **File**: `src/validation/processor.py`
- **Lines**: 45-60
- **Issue**: Function `process_validation` missing type hints
- **Recommended Fix**: Add type hints for all parameters and return type
- **Priority**: High

```python
# Current (missing type hints)
def process_validation(target_dir, output_dir, verbose=False, **kwargs):
    ...

# Recommended (with type hints)
def process_validation(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs: Any
) -> bool:
    ...
```
```

---

## 12. Usage Instructions

### 12.1 Manual Review Process

1. **Start with Executive Summary**: Review high-level coherence metrics
2. **Architecture Compliance**: Validate thin orchestrator pattern for all 24 steps
3. **Code Quality**: Check type hints, docstrings, error handling
4. **Module Structure**: Verify directory structure and __init__.py patterns
5. **Documentation**: Validate AGENTS.md and README.md completeness
6. **Testing**: Verify no-mock policy and real data usage
7. **Integration**: Check pipeline dependencies and data flow
8. **Performance**: Review execution time and memory usage
9. **Security**: Validate input validation and error recovery
10. **Naming**: Check naming conventions consistency

### 12.2 Automated Validation Integration

#### Using Pipeline Validation Script

```bash
# Run pipeline validation
python src/pipeline/pipeline_validation.py --src-dir src --output-dir output

# Save detailed report
python src/pipeline/pipeline_validation.py --save-report validation_report.json
```

#### Using Test Suite

```bash
# Run all tests
python src/2_tests.py --comprehensive

# Run specific test module
pytest src/tests/test_validation.py -v
```

### 12.3 Generating Coherence Reports

1. **Run Validation Scripts**: Execute pipeline validation and test suite
2. **Review Output**: Review validation reports and test results
3. **Assess Compliance**: Score each validation area
4. **Create Action Items**: Generate specific action items with priorities
5. **Track Improvements**: Track improvements over time

### 12.4 AI-Assisted Review

This mega-prompt can be used with AI assistants to:

1. **Systematic Review**: Review entire codebase systematically
2. **Pattern Detection**: Detect anti-patterns and inconsistencies
3. **Compliance Scoring**: Calculate compliance scores for each area
4. **Action Item Generation**: Generate specific, actionable improvement items
5. **Documentation Updates**: Update documentation based on findings

#### AI Review Prompt Template

```
Using the REPO_COHERENCE_CHECK.md mega-prompt, review the following:

1. Architecture compliance for [specific module/step]
2. Code quality standards (type hints, docstrings, error handling)
3. Module structure consistency
4. Documentation completeness
5. Testing standards compliance

Provide:
- Compliance scores for each area
- Specific issues found with file paths and line numbers
- Recommended fixes with code examples
- Priority levels for each action item
```

---

## 13. Integration with Existing Validation Tools

### 13.1 Pipeline Validation Script

**Location**: `src/pipeline/pipeline_validation.py`

**Capabilities**:
- Validates module imports and centralized utilities usage
- Checks configuration consistency
- Validates output structure
- Checks argument consistency
- Detects dependency cycles
- Validates output naming conventions
- Checks performance tracking coverage

**Usage**:
```bash
python src/pipeline/pipeline_validation.py [--fix-issues]
```

### 13.2 Validation Module

**Location**: `src/validation/`

**Capabilities**:
- Model consistency checking
- Semantic validation
- Quality assessment
- Cross-format validation

### 13.3 Test Infrastructure

**Location**: `src/tests/`

**Capabilities**:
- Comprehensive test suite
- Integration tests
- Performance tests
- Regression tests

---

## 14. Examples and Anti-Patterns

### 14.1 Correct Patterns

#### Thin Orchestrator Pattern

```python
#!/usr/bin/env python3
"""
Step N: [Module Name] (Thin Orchestrator)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    create_standardized_pipeline_script
)
from module import process_module

run_script = create_standardized_pipeline_script(
    "N_module.py",
    process_module,
    "Module processing description"
)

def main() -> int:
    """Main entry point."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
```

#### Module Implementation Pattern

```python
"""
Module Processor

Core processing logic for [module name].
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

def process_module(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs: Any
) -> bool:
    """
    Process [module] functionality.
    
    Args:
        target_dir: Input directory
        output_dir: Output directory
        verbose: Enable verbose logging
        **kwargs: Additional options
        
    Returns:
        True if processing succeeded
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Core processing logic here
        result = perform_processing(target_dir, output_dir, **kwargs)
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False
```

### 14.2 Anti-Patterns to Avoid

#### Thick Orchestrator (Incorrect)

```python
# ❌ WRONG: Long implementation in numbered script
def generate_visualization(data):
    # 100+ lines of visualization code
    import matplotlib.pyplot as plt
    import numpy as np
    # ... extensive implementation ...
    return result
```

#### Missing Type Hints (Incorrect)

```python
# ❌ WRONG: Missing type hints
def process_validation(target_dir, output_dir, verbose=False):
    ...
```

#### Using Mocks in Tests (Incorrect)

```python
# ❌ WRONG: Using mocks
from unittest.mock import Mock, patch

@patch('module.process_function')
def test_processing(mock_process):
    mock_process.return_value = True
    ...
```

---

## 15. Conclusion

This mega-prompt provides a comprehensive framework for validating repo-wide coherence across all 24 pipeline steps and 28 specialized modules. Use it systematically to:

1. **Identify Issues**: Find architectural, code quality, and documentation issues
2. **Score Compliance**: Calculate compliance scores for each validation area
3. **Generate Action Items**: Create specific, actionable improvement tasks
4. **Track Progress**: Monitor improvements over time
5. **Maintain Standards**: Ensure ongoing compliance with established patterns

### Next Steps

1. Run automated validation scripts
2. Perform manual review using this checklist
3. Generate compliance scores for each area
4. Create prioritized action items
5. Implement improvements systematically
6. Re-validate after changes

---

**Status**: Production Ready
**Version**: 1.0.0
**Coverage**: 24 pipeline steps, 28 modules, 41 AGENTS.md files

---

## 16. Known Issues and Action Items

### 16.1 Testing Policy Violations

**Issue**: Some test files violate the no-mock policy by using `unittest.mock`.

**Affected Files**:
- `src/tests/test_pipeline_warnings_fix.py`
- `src/tests/test_pipeline_recovery.py`
- `src/tests/test_pipeline_error_scenarios.py`
- `src/tests/test_fast_suite.py`
- `src/tests/test_d2_visualizer.py`
- `src/tests/test_advanced_visualization_overall.py`
- `src/tests/conftest.py`

**Priority**: High
**Action**: Refactor these tests to use real code paths and real data. Tests may skip when external dependencies are unavailable, but must never replace dependencies with mocks.

### 16.2 Validation Priorities

When using this mega-prompt for code review, prioritize:

1. **Critical**: Architecture compliance (thin orchestrator pattern)
2. **High**: Code quality (type hints, docstrings, error handling)
3. **High**: Testing standards (no-mock policy compliance)
4. **Medium**: Documentation completeness (AGENTS.md, README.md)
5. **Medium**: Module structure consistency
6. **Low**: Performance optimization opportunities
7. **Low**: Naming convention improvements

### 16.3 Quick Validation Commands

```bash
# Check for unittest.mock usage
grep -r "from unittest.mock" src/tests/

# Check pipeline script lengths
find src -name "[0-9]_*.py" -exec wc -l {} \; | sort -n

# Check for missing type hints in public functions
# (requires type checking tool like mypy or pyright)

# Validate output directory structure
ls -la output/ | grep "_output"

# Check AGENTS.md coverage
find src -name "AGENTS.md" | wc -l
```

