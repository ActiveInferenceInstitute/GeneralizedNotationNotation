# Validation Module - Agent Scaffolding

## Module Overview

**Purpose**: Advanced validation and consistency checking for GNN models and pipeline components

**Pipeline Step**: Step 6: Validation (6_validation.py)

**Category**: Validation / Quality Assurance

**Status**: Production Ready

**Version**: 3.2.0

**Last Updated**: 2026-09-02

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

#### `process_validation(target_dir, output_dir, verbose=False, **kwargs) -> bool`
**Description**: Main validation processing function called by orchestrator (6_validation.py)

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to validate
- `output_dir` (Path): Output directory for validation results
- `verbose` (bool): Enable verbose logging (default: False); logging otherwise goes to the module logger
- `**kwargs`: Additional validation options (accepted but not consumed; behavior is governed by validator defaults)

**Returns**: `True` if validation succeeded

**Example**:
```python
from validation import process_validation

success = process_validation(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/6_validation_output"),
    verbose=True,
)

```

#### `process_semantic_validation(model_data) -> Dict[str, Any]`
**Description**: Perform semantic validation on model data. Returns `{file_path, file_name, valid, errors, warnings, semantic_score}` (`status: error` dict on failure).

**Parameters**:
- `model_data` (Dict[str, Any]): Parsed GNN model data

**Returns**: Dictionary with semantic validation results

#### `profile_performance(model_data) -> Dict[str, Any]`
**Description**: Profile model performance characteristics. Returns `{file_path, file_name, metrics, warnings, performance_score}` (`status: error` dict on failure).

**Parameters**:
- `model_data` (str | Path | Dict[str, Any]): GNN content string, file path, or parsed model data

#### `check_consistency(model_data) -> Dict[str, Any]`
**Description**: Check consistency of model data. Returns `{file_path, file_name, consistent, warnings, checks, consistency_score}` (`status: error` dict on failure).

**Parameters**:
- `model_data` (str | Path | Dict[str, Any]): GNN file path or parsed model data

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

No module-level configuration file. Semantic validation depth is set via `SemanticValidator(validation_level=...)` (`basic`, `standard`, `strict`, `research`); the orchestrator runs all three validators with their defaults.

---

## Usage Examples

### Basic Validation
```python
from validation import process_validation

success = process_validation(
    target_dir="input/gnn_files", output_dir="output/6_validation_output"
)
```

### Semantic Validation
```python
from validation import process_semantic_validation

result = process_semantic_validation("model.gnn")
print(f"Valid: {result['valid']}, Score: {result['semantic_score']:.2f}")
```

### Performance Profiling
```python
from validation import profile_performance

result = profile_performance("model.gnn")
print(f"Memory estimate: {result['metrics']['estimated_memory_mb']:.2f} MB")
print(f"Score: {result['performance_score']:.2f}")
```

---

## Output Specification

### Output Products
- `validation_results.json` - Validation results
- `validation_summary.json` - Validation summary

### Output Directory Structure
```
output/6_validation_output/
├── validation_results.json
└── validation_summary.json
```

---

## Performance Characteristics

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
Measure on demand:

```bash
uv run --extra dev python -m pytest src/tests/test_validation*.py \
    --cov=src/validation --cov-report=term-missing
```
### Key Test Scenarios
1. Model structure validation
2. Semantic consistency checking
3. Performance profiling accuracy
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
Registered by `validation.mcp.register_tools(mcp_instance)` (4 tools):
- `process_validation` - Run full validation pipeline on a directory
- `validate_gnn_file` - Validate a single GNN file (basic/standard/strict level)
- `get_validation_report` - Read saved validation reports from a previous run
- `check_schema_compliance` - Check a GNN model string against canonical schema requirements

---

## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
