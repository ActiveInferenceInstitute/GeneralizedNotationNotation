# Step 6: Validation

## Architectural Mapping

**Orchestrator**: `src/6_validation.py` (50 lines)
**Implementation Layer**: `src/validation/`

## Module Description

This module provides comprehensive validation capabilities for GNN models, including consistency checking, semantic validation, and quality assessment.


```
src/validation/
├── __init__.py                    # Module initialization, exports, and process_validation orchestrator
├── README.md                      # This documentation
├── AGENTS.md                      # Agent scaffolding documentation
├── SPEC.md                        # Module specification
├── consistency_checker.py         # Consistency checking (naming, style, structure, references)
├── semantic_validator.py          # Semantic validation (structure, state space, connections, math)
├── performance_profiler.py        # Performance profiling (complexity, memory, parallelization)
└── mcp.py                         # Model Context Protocol integration

## Agent Identity & Capabilities

# Validation Module - Agent Scaffolding

## Module Overview

**Purpose**: Advanced validation and consistency checking for GNN models and pipeline components

**Pipeline Step**: Step 6: Validation (6_validation.py)

**Category**: Validation / Quality Assurance

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-01-21

---

## Core Functionality

### Primary Responsibilities
1. Advanced validation and consistency checking
2. Model structure and semantic validation
3. Performance profiling and optimization
4. Cross-format consistency verification
5. Quality assurance and compliance checking

### Key Capabilities
- Comprehensive model validation
- Semantic consistency checking
- Performance profiling and analysis
- Cross-format validation
- Quality metrics and compliance

---

## API Reference

### Public Functions

#### `process_validation(target_dir, output_dir, verbose=False, logger=None, **kwargs) -> bool`
**Description**: Main validation processing function called by orchestrator (6_validation.py)

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to validate
- `output_dir` (Path): Output directory for validation results
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Logger, optional): Logger instance (default: None)
- `strict` (bool): Enable strict validation mode (default: False)
- `profile` (bool): Enable performance profiling (default: False)
- `**kwargs`: Additional validation options

**Returns**: `True` if validation succeeded

**Example**:
```python
from validation import process_validation

success = process_validation(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/6_validation_output"),
    verbose=True,
    strict=True,
    profile=True
)
```

#### `process_semantic_validation(model_data: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Perform semantic validation on model data

**Parameters**:
- `model_data` (Dict[str, Any]): Parsed GNN model data

**Returns**: Dictionary with semantic validation results

#### `profile_performance(model_data: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Profile model performance characteristics

**Parameters**:
- `model_data` (Dict[str, Any]): Parsed GNN model data

**Returns**: Dictionary with performance metrics

#### `check_consistency(model_data: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Check consistency of model data

**Parameters**:
- `model_data` (Dict[str, Any]): Parsed GNN model data

**Returns**: Dictionary with consistency results

---

## Dependencies

### Required Dependencies
- `pathlib` - Path manipulation
- `re` - Regular expressions for parsing

### Internal Dependencies
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### Validation Settings
```python
VALIDATION_CONFIG = {
    "strict_validation": False,
    "profile_performance": True,
    "check_consistency": True,
    "validate_semantics": True,
}
```

---

## Usage Examples

### Basic Validation
```python
from validation import process_validation

success = process_validation(
    target_dir="input/gnn_files", output_dir="output/6_validation_output"
)
```

### Semantic Validation and Consistency
```python
from validation import process_semantic_validation, check_consistency

semantic = process_semantic_validation("model.gnn")
consistency = check_consistency("model.gnn")
```

### Performance Profiling
```python
from validation import profile_performance

profile = profile_performance("model.gnn")
```

The exported surface is listed in `src/validation/__init__.py` (`__all__`):
`process_validation`, `process_semantic_validation`, `profile_performance`,
`check_consistency` plus the `SemanticValidator`, `PerformanceProfiler` and
`ConsistencyChecker` classes.

---

## Output Specification

### Output Products
- `validation_results.json` - Validation results
- `performance_profile.json` - Performance profiling
- `consistency_report.json` - Consistency checking
- `validation_summary.md` - Human-readable summary

### Output Directory Structure
```
output/6_validation_output/
├── validation_results.json
├── performance_profile.json
├── consistency_report.json
├── validation_summary.md
└── detailed_analysis/
    ├── structure_validation.json
    └── semantic_validation.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~1-5 seconds per model
- **Memory**: ~20-100MB
- **Status**: ✅ Production Ready

### Expected Performance
- **Basic Validation**: < 1 second
- **Structure Validation**: 1-3 seconds
- **Performance Profiling**: 2-5 seconds
- **Consistency Checking**: 1-4 seconds

---

## Error Handling

### Validation Errors
1. **Structure Errors**: Invalid model structure
2. **Semantic Errors**: Semantic inconsistencies
3. **Performance Issues**: Performance problems
4. **Consistency Errors**: Cross-format inconsistencies

### Recovery Strategies
- **Structure Repair**: Suggest structural fixes
- **Semantic Resolution**: Provide semantic guidance
- **Performance Optimization**: Suggest performance improvements
- **Consistency Reconciliation**: Resolve format differences

---

## Integration Points

### Orchestrated By
- **Script**: `6_validation.py` (Step 6)
- **Function**: `process_validation()`

### Imports From
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `main.py` - Pipeline orchestration
- `tests.test_validation_*` - Validation tests

### Data Flow
```
Model Content → Structure Validation → Semantic Validation → Performance Profiling → Consistency Checking
```

---

## Testing

### Test Files
- `src/tests/validation/test_validation_overall.py` - Module-level validation tests
- `src/tests/gnn/test_gnn_validation.py` - GNN validation-focused tests (shared)

### Test Coverage
- Measure: `uv run --extra dev python -m pytest src/tests/validation/ --cov=validation --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios
1. Model structure validation
2. Semantic consistency checking
3. Performance profiling accuracy
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `process_validation` - Run Step 6 over a directory
- `validate_gnn_file` - Validate one GNN file
- `get_validation_report` - Read the latest validation results
- `check_schema_compliance` - Check a file against the GNN schema

### Tool Endpoints
```python
@mcp_tool("validation.validate_structure")
def validate_structure_tool(content):
    """Validate model structure"""
    # Implementation
```

---

---
## Documentation
- **[README](../../../src/validation/README.md)**: Module Overview
- **[AGENTS](../../../src/validation/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/validation/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/validation/SKILL.md)**: Capability API


---

**Source Reference**: [src/validation](../../../src/validation)
