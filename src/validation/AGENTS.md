# Validation Module - Agent Scaffolding

## Module Overview

**Purpose**: Advanced validation and consistency checking for GNN models and pipeline components

**Pipeline Step**: Step 6: Validation (6_validation.py)

**Category**: Validation / Quality Assurance

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

#### `process_validation(target_dir, output_dir, **kwargs) -> bool`
**Description**: Process validation for GNN models

**Parameters**:
- `target_dir`: Directory containing files to validate
- `output_dir`: Output directory for validation results
- `**kwargs`: Additional validation options

**Returns**: `True` if validation succeeded

#### `validate_model_structure(content) -> Dict[str, Any]`
**Description**: Validate model structure and consistency

**Parameters**:
- `content`: Model content to validate

**Returns**: Dictionary with validation results

#### `profile_model_performance(content) -> Dict[str, Any]`
**Description**: Profile model performance characteristics

**Parameters**:
- `content`: Model content to profile

**Returns**: Dictionary with performance metrics

#### `check_cross_format_consistency(content) -> Dict[str, Any]`
**Description**: Check consistency across formats

**Parameters**:
- `content`: Content to check for consistency

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
    'strict_validation': False,
    'profile_performance': True,
    'check_consistency': True,
    'validate_semantics': True
}
```

---

## Usage Examples

### Basic Validation
```python
from validation import process_validation

success = process_validation(
    target_dir="input/gnn_files",
    output_dir="output/6_validation_output"
)
```

### Model Structure Validation
```python
from validation import validate_model_structure

with open("model.gnn", "r") as f:
    content = f.read()

validation = validate_model_structure(content)
if validation['valid']:
    print("Model structure is valid")
else:
    print("Validation issues:")
    for issue in validation['issues']:
        print(f"  - {issue}")
```

### Performance Profiling
```python
from validation import profile_model_performance

profile = profile_model_performance(content)
print(f"Estimated complexity: {profile['complexity_score']}")
print(f"Performance rating: {profile['performance_rating']}")
```

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
- `src/tests/test_validation_integration.py` - Integration tests
- `src/tests/test_validation_functionality.py` - Functionality tests

### Test Coverage
- **Current**: 82%
- **Target**: 85%+

### Key Test Scenarios
1. Model structure validation
2. Semantic consistency checking
3. Performance profiling accuracy
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `validation.validate_structure` - Validate model structure
- `validation.profile_performance` - Profile model performance
- `validation.check_consistency` - Check cross-format consistency
- `validation.analyze_quality` - Analyze model quality

### Tool Endpoints
```python
@mcp_tool("validation.validate_structure")
def validate_structure_tool(content):
    """Validate model structure"""
    # Implementation
```

---

**Last Updated: October 28, 2025
**Status**: ✅ Production Ready