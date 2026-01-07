# Type Checker Module - Agent Scaffolding

## Module Overview

**Purpose**: GNN syntax validation, type checking, and resource estimation for the GNN processing pipeline

**Pipeline Step**: Step 5: Type checking (5_type_checker.py)

**Category**: Type Checking / Validation

**Status**: ✅ Production Ready

**Version**: 1.0.0

**Last Updated**: 2025-12-30

---

## Core Functionality

### Primary Responsibilities
1. GNN syntax validation and type checking
2. Resource estimation and optimization suggestions
3. Model structure validation
4. Performance prediction and analysis
5. Type safety verification

### Key Capabilities
- Comprehensive GNN syntax validation
- Type checking and inference
- Resource estimation and optimization
- Performance prediction modeling
- Model structure validation

---

## API Reference

### Public Classes

#### `GNNTypeChecker`
**Description**: Type checker class for GNN files called by orchestrator (5_type_checker.py)

**Methods**:
- `validate_gnn_files(target_dir, output_dir, verbose=False, **kwargs) -> bool` - Validate GNN files in target directory
  - `target_dir` (Path): Directory containing GNN files
  - `output_dir` (Path): Output directory for results
  - `verbose` (bool): Enable verbose output (default: False)
  - `**kwargs`: Additional validation options (strict, estimate_resources)
  - Returns: `True` if validation succeeded

**Example**:
```python
from type_checker import GNNTypeChecker

checker = GNNTypeChecker()
success = checker.validate_gnn_files(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/5_type_checker_output"),
    verbose=True,
    strict=True,
    estimate_resources=True
)
```

### Public Functions

#### `validate_gnn_files(target_dir, output_dir, **kwargs) -> bool`
**Description**: Convenience function for validating GNN files (creates GNNTypeChecker instance)

**Parameters**:
- `target_dir`: Directory containing GNN files
- `output_dir`: Output directory for results
- `**kwargs`: Additional validation options

**Returns**: `True` if validation succeeded

#### `estimate_file_resources(content: str) -> Dict[str, Any]`
**Description**: Estimate computational resources for GNN model from file content

**Parameters**:
- `content` (str): GNN file content to analyze

**Returns**: Dictionary with resource estimates including memory, execution time, and optimization suggestions

**Example**:
```python
from type_checker import estimate_file_resources

with open("model.md", "r") as f:
    content = f.read()

estimates = estimate_file_resources(content)
print(f"Estimated memory: {estimates.get('memory_mb', 0)} MB")
print(f"Estimated execution time: {estimates.get('execution_time', 0)} seconds")
```

#### `check_type_safety(gnn_content) -> Dict[str, Any]`
**Description**: Check type safety of GNN model

**Parameters**:
- `gnn_content`: GNN content to check

**Returns**: Dictionary with type safety results

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
TYPE_CHECKER_CONFIG = {
    'strict_validation': False,
    'estimate_resources': True,
    'check_performance': True,
    'validate_structure': True
}
```

---

## Usage Examples

### Basic Type Checking
```python
from type_checker import validate_gnn_files

success = validate_gnn_files(
    target_dir="input/gnn_files",
    output_dir="output/5_type_checker_output"
)
```

### Resource Estimation
```python
from type_checker import estimate_file_resources

with open("model.md", "r") as f:
    content = f.read()

estimates = estimate_file_resources(content)
print(f"Estimated memory: {estimates.get('memory_mb', 0)} MB")
print(f"Estimated execution time: {estimates.get('execution_time', 0)} seconds")
```

### Type Safety Check
```python
from type_checker import check_type_safety

safety = check_type_safety(content)
if safety['type_safe']:
    print("Model is type safe")
else:
    print("Type issues found:")
    for issue in safety['issues']:
        print(f"  - {issue}")
```

---

## Output Specification

### Output Products
- `type_check_results.json` - Type checking results
- `resource_estimates.json` - Resource estimation results
- `type_safety_report.md` - Human-readable type safety report
- `validation_summary.json` - Validation summary

### Output Directory Structure
```
output/5_type_checker_output/
├── type_check_results.json
├── resource_estimates.json
├── type_safety_report.md
├── validation_summary.json
└── detailed_analysis/
    ├── syntax_analysis.json
    └── structure_analysis.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~1-3 seconds per model
- **Memory**: ~20-50MB
- **Status**: ✅ Production Ready

### Expected Performance
- **Basic Validation**: < 1 second
- **Resource Estimation**: 1-2 seconds
- **Type Safety Check**: 1-3 seconds
- **Comprehensive Analysis**: 2-5 seconds

---

## Error Handling

### Validation Errors
1. **Syntax Errors**: Invalid GNN syntax
2. **Type Errors**: Type mismatches or inconsistencies
3. **Structure Errors**: Invalid model structure
4. **Resource Errors**: Resource estimation failures

### Recovery Strategies
- **Syntax Repair**: Suggest syntax fixes
- **Type Coercion**: Suggest type conversions
- **Structure Improvement**: Suggest structural improvements
- **Resource Optimization**: Provide optimization suggestions

---

## Integration Points

### Orchestrated By
- **Script**: `5_type_checker.py` (Step 5)
- **Function**: `validate_gnn_files()`

### Imports From
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `main.py` - Pipeline orchestration
- `tests.test_type_checker_*` - Type checker tests

### Data Flow
```
GNN Files → Syntax Validation → Type Checking → Resource Estimation → Optimization Suggestions
```

---

## Testing

### Test Files
- `src/tests/test_type_checker_integration.py` - Integration tests
- `src/tests/test_type_checker_validation.py` - Validation tests

### Test Coverage
- **Current**: 88%
- **Target**: 90%+

### Key Test Scenarios
1. GNN syntax validation
2. Type checking and inference
3. Resource estimation accuracy
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `type_checker.validate` - Validate GNN type safety
- `type_checker.estimate_file_resources` - Estimate computational resources
- `type_checker.check_structure` - Check model structure
- `type_checker.optimize` - Provide optimization suggestions

### Tool Endpoints
```python
@mcp_tool("type_checker.validate")
def validate_gnn_tool(file_path):
    """Validate GNN file type safety"""
    # Implementation
```

### MCP File Location
- `src/type_checker/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Type checking fails on valid GNN files
**Symptom**: Type checker reports errors on files that appear valid  
**Cause**: GNN syntax variations or missing sections  
**Solution**: 
- Check that all required sections are present (StateSpaceBlock, Connections, etc.)
- Verify syntax matches GNN specification
- Use `--verbose` flag for detailed error messages

#### Issue 2: Resource estimation returns zero values
**Symptom**: `estimate_file_resources()` returns all zeros  
**Cause**: File content parsing failed or empty content  
**Solution**:
- Verify file content is not empty
- Check file encoding (should be UTF-8)
- Ensure file contains valid GNN syntax

### Performance Issues

#### Slow Type Checking
**Symptoms**: Type checking takes longer than expected  
**Diagnosis**:
```bash
# Enable verbose logging
python src/5_type_checker.py --target-dir input/ --verbose
```

**Solutions**:
- Use `--fast` mode for basic validation only
- Disable resource estimation if not needed
- Process files in smaller batches

---

## Version History

### Current Version: 1.0.0

**Features**:
- GNN syntax validation
- Type checking and inference
- Resource estimation
- Performance prediction
- Type safety verification

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced type inference for complex models
- **Future**: Integration with static analysis tools

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [GNN Syntax Guide](../../doc/gnn/gnn_syntax.md)
- [Type System Documentation](../../doc/gnn/gnn_type_system.md)

### External Resources
- [Active Inference Documentation](https://activeinference.org)
- [Python Type Hints (PEP 484)](https://www.python.org/dev/peps/pep-0484/)

---

**Last Updated**: 2025-12-30
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: 1.0.0
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern