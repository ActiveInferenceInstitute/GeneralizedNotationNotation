# [Module Name] - Agent Scaffolding

## Module Overview

**Purpose**: [Brief 1-2 sentence description of what this module does]

**Pipeline Step**: [Script number and name, e.g., "Step 11: Code Rendering (11_render.py)"]

**Category**: [Core/Processing/Integration/Utility]

---

## Core Functionality

### Primary Responsibilities
1. [Main responsibility 1]
2. [Main responsibility 2]
3. [Main responsibility 3]

### Key Capabilities
- [Capability 1]
- [Capability 2]
- [Capability 3]

---

## API Reference

### Public Functions

#### `primary_function(args) -> ReturnType`
**Description**: [What this function does]

**Parameters**:
- `arg1` (type): [description]
- `arg2` (type): [description]

**Returns**: [Return value description]

**Example**:
```python
result = primary_function(arg1, arg2)
```

#### `secondary_function(args) -> ReturnType`
**Description**: [What this function does]

**Parameters**:
- `arg1` (type): [description]

**Returns**: [Return value description]

---

### Public Classes

#### `MainClass`
**Description**: [What this class does]

**Methods**:
- `method1(args)` - [description]
- `method2(args)` - [description]

**Example**:
```python
instance = MainClass()
result = instance.method1(arg)
```

---

## Dependencies

### Required Dependencies
- `numpy` - [reason]
- `pathlib` - [reason]

### Optional Dependencies
- `matplotlib` - [reason, fallback behavior]
- `pandas` - [reason, fallback behavior]

### Internal Dependencies
- `utils.pipeline_template` - [what functions/classes used]
- `pipeline.config` - [what functions/classes used]

---

## Configuration

### Environment Variables
- `VAR_NAME` - [description, default value]

### Configuration Files
- `config.yaml` - [what settings]

### Default Settings
```python
DEFAULT_SETTING_1 = value
DEFAULT_SETTING_2 = value
```

---

## Usage Examples

### Basic Usage
```python
from [module] import primary_function

result = primary_function(
    arg1="value1",
    arg2="value2"
)
```

### Advanced Usage
```python
from [module] import MainClass

processor = MainClass(
    option1=True,
    option2="advanced"
)

result = processor.process(data)
```

### Pipeline Integration
```python
# Called from numbered script
from [module] import process_[module]_standardized

success = process_[module]_standardized(
    target_dir=Path("input"),
    output_dir=Path("output"),
    logger=logger,
    verbose=True
)
```

---

## Input/Output Specification

### Input Requirements
- **File Formats**: [list formats]
- **Directory Structure**: [expected structure]
- **Prerequisites**: [what must run before this]

### Output Products
- **Primary Outputs**: [list main outputs]
- **Metadata Files**: [JSON summaries, logs]
- **Artifacts**: [visualizations, reports, etc.]

### Output Directory Structure
```
output/[step]_[module]_output/
├── primary_output.ext
├── metadata.json
├── processing_summary.json
└── [subdirectories]/
```

---

## Error Handling

### Error Categories
1. **Dependency Errors**: [how handled]
2. **File Errors**: [how handled]
3. **Processing Errors**: [how handled]

### Fallback Strategies
- **Primary**: [main approach]
- **Fallback 1**: [if primary fails]
- **Fallback 2**: [if fallback 1 fails]
- **Final**: [minimal output/graceful degradation]

### Error Reporting
- **Logging Level**: [INFO/WARNING/ERROR]
- **User Messages**: [actionable error messages]
- **Recovery Suggestions**: [how to fix]

---

## Integration Points

### Orchestrated By
- **Script**: [numbered script name]
- **Function**: [which function calls this module]

### Imports From
- `utils.[utility]` - [what functionality]
- `[module].[submodule]` - [what functionality]

### Imported By
- `[other_module]` - [what they use]
- `tests.[test_module]` - [what tests use]

### Data Flow
```
[Previous Step] → [This Module] → [Next Step]
     ↓                  ↓                ↓
  input data      processing      output data
```

---

## Testing

### Test Files
- `src/tests/test_[module]_integration.py` - Integration tests
- `src/tests/test_[module]_unit.py` - Unit tests
- `src/tests/test_[module]_error_scenarios.py` - Error handling tests

### Test Coverage
- **Current**: [percentage]%
- **Target**: 90%+

### Key Test Scenarios
1. [Scenario 1]
2. [Scenario 2]
3. [Scenario 3]

### Test Commands
```bash
# Run module-specific tests
pytest src/tests/test_[module]*.py -v

# Run with coverage
pytest src/tests/test_[module]*.py --cov=src/[module] --cov-report=term-missing
```

---

## MCP Integration

### Tools Registered
- `[module]_process` - [description]
- `[module]_analyze` - [description]

### Tool Endpoints
```python
@mcp_tool("module_process")
def process_tool(args):
    """Tool description"""
    # Implementation
```

### MCP File Location
- `src/[module]/mcp.py` - MCP tool registrations

---

## Performance Characteristics

### Resource Requirements
- **Memory**: [typical usage]
- **CPU**: [typical usage]
- **Disk**: [typical usage]

### Execution Time
- **Fast Path**: [time for typical input]
- **Slow Path**: [time for large input]
- **Timeout**: [configured timeout]

### Scalability
- **Input Size Limits**: [max recommended]
- **Parallelization**: [supported/not supported]

---

## Development Guidelines

### Adding New Features
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Code Style
- Follow PEP 8
- Use type hints
- Document all public functions
- Include docstring examples

### Testing Requirements
- All new functions must have tests
- Coverage must remain >90%
- Error scenarios must be tested

---

## Troubleshooting

### Common Issues

#### Issue 1: [Problem description]
**Symptom**: [What user sees]  
**Cause**: [Why it happens]  
**Solution**: [How to fix]

#### Issue 2: [Problem description]
**Symptom**: [What user sees]  
**Cause**: [Why it happens]  
**Solution**: [How to fix]

### Debug Mode
```bash
# Run with verbose logging
python src/[N]_[module].py --verbose

# Check output directory
ls -la output/[N]_[module]_output/

# View processing summary
cat output/[N]_[module]_output/[module]_processing_summary.json
```

---

## Version History

### Current Version: 1.0.0

**Features**:
- [Feature 1]
- [Feature 2]

**Known Issues**:
- [Issue 1 - workaround]

### Roadmap
- **Next Version**: [Planned features]
- **Future**: [Long-term goals]

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [.cursorrules](../../.cursorrules)

### External Resources
- [Link to relevant papers/docs]
- [Link to dependency documentation]

---

**Last Updated**: [Date]  
**Maintainer**: GNN Pipeline Team  
**Status**: ✅ Production Ready / 🔄 In Development / ⚠️ Experimental



