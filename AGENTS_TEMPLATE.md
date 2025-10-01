# [Module Name] - Agent Scaffolding

## Module Overview

**Purpose**: [Brief 1-2 sentence description of what this module does. Focus on the core functionality and value proposition.]

**Pipeline Step**: [Script number and name, e.g., "Step 11: Code Rendering (11_render.py)"]

**Category**: [Core/Processing/Integration/Utility/Analysis/Visualization/Audio]

**Status**: [Production Ready / Beta / Experimental / Deprecated]

**Version**: [Current version, e.g., "1.0.0"]

**Last Updated**: [Date in ISO format, e.g., "2025-10-01"]

---

## Core Functionality

### Primary Responsibilities
1. [Main responsibility 1 - be specific about what the module actually does]
2. [Main responsibility 2 - focus on core value and outcomes]
3. [Main responsibility 3 - describe the key transformation or processing]

### Key Capabilities
- [Capability 1 - specific technical capability with concrete examples]
- [Capability 2 - integration or processing capability]
- [Capability 3 - output or analysis capability]

### Agent Capabilities
This module provides specialized agent capabilities for [specific domain]:

#### 🎯 [Agent Type 1] Agent
- **Core Function**: [What this agent does]
- **Input Processing**: [What inputs it handles]
- **Output Generation**: [What outputs it produces]
- **Decision Making**: [How it makes choices or optimizations]

#### 🤖 [Agent Type 2] Agent
- **Core Function**: [What this agent does]
- **Specialization**: [What makes this agent unique]
- **Integration Points**: [How it connects with other modules]

---

## API Reference

### Public Functions

#### `primary_function(args) -> ReturnType`
**Description**: [What this function does - be specific and include the core purpose]

**Parameters**:
- `arg1` (type): [description - include valid values and constraints]
- `arg2` (type): [description - mention any dependencies or requirements]

**Returns**: [Return value description - specify format and content]

**Raises**:
- `ValueError`: [When and why this exception is raised]
- `FileNotFoundError`: [When input files are missing]
- `TypeError`: [When parameters have incorrect types]

**Example**:
```python
from [module] import primary_function

# Basic usage
result = primary_function(
    arg1="input_value",
    arg2="configuration"
)

# Advanced usage with error handling
try:
    result = primary_function(
        arg1=Path("input/file.md"),
        arg2={"option": "value"}
    )
    print(f"Processing completed: {result}")
except ValueError as e:
    print(f"Configuration error: {e}")
```

#### `secondary_function(args) -> ReturnType`
**Description**: [What this function does - focus on specific functionality]

**Parameters**:
- `arg1` (type): [description with detailed constraints]

**Returns**: [Return value description with format specification]

**Example**:
```python
from [module] import secondary_function

# Usage example
result = secondary_function(
    input_data=data,
    config={"param": "value"}
)
```

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

# Simple processing
result = primary_function(
    input_path="data/input.md",
    output_dir="output/",
    verbose=True
)
print(f"Processing completed: {result}")
```

### Advanced Usage with Configuration
```python
from [module] import MainClass
from pathlib import Path

# Configure processing options
processor = MainClass(
    enable_feature_x=True,
    optimization_level="high",
    cache_results=True
)

# Process with custom parameters
result = processor.process(
    data=input_data,
    config={
        "param1": "value1",
        "param2": 42
    }
)
```

### Error Handling and Recovery
```python
from [module] import primary_function
import logging

logger = logging.getLogger(__name__)

try:
    result = primary_function(
        input_file=Path("input/model.md"),
        output_dir=Path("output/"),
        timeout=300
    )
except FileNotFoundError as e:
    logger.error(f"Input file not found: {e}")
    # Fallback processing or user notification
except TimeoutError as e:
    logger.warning(f"Processing timed out: {e}")
    # Retry with different parameters
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Graceful degradation
```

### Pipeline Integration
```python
# Called from numbered script (thin orchestrator pattern)
from [module] import process_[module]_standardized
from utils.pipeline_template import setup_step_logging
from pathlib import Path

def main():
    # Setup standardized logging
    logger = setup_step_logging("[module]", verbose=True)

    # Process with pipeline integration
    success = process_[module]_standardized(
        target_dir=Path("input/gnn_files"),
        output_dir=Path("output/[step]_output"),
        logger=logger,
        verbose=True,
        config={"option": "value"}
    )

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
```

### Batch Processing
```python
from [module] import batch_processor
from pathlib import Path

# Process multiple files
files_to_process = [
    Path("input/model1.md"),
    Path("input/model2.md"),
    Path("input/model3.md")
]

results = batch_processor.process_files(
    input_files=files_to_process,
    output_dir=Path("output/batch/"),
    parallel=True,
    max_workers=4
)

for file_path, result in results.items():
    print(f"{file_path}: {'SUCCESS' if result.success else 'FAILED'}")
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
- **Memory**: [typical usage pattern - e.g., "2-8GB for standard models, scales with model complexity"]
- **CPU**: [typical usage pattern - e.g., "Single core for basic processing, multi-core for batch operations"]
- **Disk**: [typical usage pattern - e.g., "100MB-1GB temporary storage during processing"]
- **Network**: [network requirements if any - e.g., "API calls to external services"]

### Execution Time Benchmarks
- **Fast Path**: [time for typical input - e.g., "30-60 seconds for standard 100KB GNN file"]
- **Slow Path**: [time for large input - e.g., "5-10 minutes for complex 1MB+ models"]
- **Timeout**: [configured timeout - e.g., "300 seconds default, configurable per operation"]

### Scalability Metrics
- **Input Size Limits**: [max recommended - e.g., "Tested up to 10MB GNN files, performance degrades beyond 50MB"]
- **Parallelization**: [supported/not supported - e.g., "Full batch processing support, limited by memory"]
- **Concurrent Users**: [multi-user capability - e.g., "Thread-safe for concurrent processing"]
- **Horizontal Scaling**: [scaling capability - e.g., "Can be distributed across multiple nodes"]

### Performance Optimization Tips
- [Specific optimization 1 - e.g., "Use caching for repeated model processing"]
- [Specific optimization 2 - e.g., "Batch similar models together for efficiency"]
- [Specific optimization 3 - e.g., "Configure memory limits based on available resources"]

---

## Development Guidelines

### Adding New Features
1. **Plan**: Define the feature requirements and integration points with existing modules
2. **Design**: Create interface specifications and update API documentation
3. **Implement**: Follow thin orchestrator pattern, add comprehensive tests
4. **Test**: Validate functionality with real data and edge cases
5. **Document**: Update AGENTS.md and add usage examples
6. **Review**: Ensure compliance with all coding standards and patterns

### Code Style and Standards
- **PEP 8 Compliance**: Follow Python style guidelines strictly
- **Type Hints**: All public functions must have complete type annotations
- **Documentation**: Every public function/class must have comprehensive docstrings
- **Examples**: Include practical examples in all docstrings
- **Error Handling**: Implement proper exception handling with meaningful messages
- **Resource Management**: Ensure proper cleanup of files, connections, and memory

### Testing Requirements
- **Unit Tests**: All new functions must have corresponding unit tests
- **Integration Tests**: Test module integration with pipeline and other modules
- **Coverage Target**: Maintain >95% test coverage for new code
- **Edge Cases**: Test all error scenarios and boundary conditions
- **Performance Tests**: Include timing and memory usage validation
- **Real Data**: Use actual representative data, not mocks or synthetic data

### Agent Development Best Practices
1. **Single Responsibility**: Each agent should have one clear, well-defined purpose
2. **Stateless Design**: Prefer stateless agents for better testability and reliability
3. **Configuration**: Use dependency injection for flexible configuration
4. **Monitoring**: Implement health checks and performance monitoring
5. **Error Recovery**: Design graceful degradation strategies for failures
6. **Resource Awareness**: Implement proper resource cleanup and limits

---

## Troubleshooting

### Common Issues

#### Issue 1: [Problem description - be specific about the actual error]
**Symptom**: [What user sees - include actual error messages or behaviors]  
**Cause**: [Why it happens - technical root cause explanation]  
**Solution**: [How to fix - step-by-step resolution with code examples]

**Example**:
```bash
# Diagnostic command
python src/[N]_[module].py --target-dir input/ --verbose --debug

# Check specific log files
tail -f output/[N]_[module]_output/*.log
```

#### Issue 2: [Problem description - focus on real issues users encounter]
**Symptom**: [What user sees - concrete symptoms and error patterns]  
**Cause**: [Why it happens - underlying technical cause]  
**Solution**: [How to fix - actionable steps with verification]

### Performance Issues

#### Slow Processing
**Symptoms**: Processing takes longer than expected timeouts
**Diagnosis**:
```bash
# Enable performance profiling
python src/[N]_[module].py --profile --verbose

# Check resource usage
python src/main.py --only-steps [N] --verbose
```

**Solutions**:
- [Specific optimization 1]
- [Specific optimization 2]
- [Configuration adjustment]

#### Memory Issues
**Symptoms**: OutOfMemory errors or excessive memory usage
**Diagnosis**:
```bash
# Monitor memory usage
python src/[N]_[module].py --monitor-memory --verbose
```

**Solutions**:
- [Memory optimization strategy 1]
- [Memory optimization strategy 2]

### Integration Issues

#### Pipeline Integration Failures
**Symptoms**: Module fails when called from pipeline but works standalone
**Diagnosis**:
```bash
# Test standalone
python src/[N]_[module].py --target-dir test_input/

# Test with pipeline
python src/main.py --only-steps [N] --target-dir test_input/
```

### Debug Mode Commands
```bash
# Run with comprehensive debugging
python src/[N]_[module].py --target-dir input/ --verbose --debug --profile

# Check all output files
find output/[N]_[module]_output/ -name "*.json" -exec cat {} \;

# View detailed logs
ls -la output/[N]_[module]_output/*.log
cat output/[N]_[module]_output/[module]_processing_summary.json
```

### Getting Help
1. **Check Documentation**: Review this AGENTS.md file for usage examples
2. **Enable Debug Mode**: Use `--verbose --debug` flags for detailed logging
3. **Examine Outputs**: Check `output/[N]_[module]_output/` for diagnostic files
4. **Community Support**: Open GitHub issues with complete error logs and reproduction steps

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

## Agent Architecture Patterns

This module implements several key agent architecture patterns that ensure robust, maintainable, and scalable functionality:

### 1. **Thin Orchestrator Pattern**
- **Script Level**: `src/N_[module].py` handles argument parsing, logging, and high-level flow control
- **Module Level**: `src/[module]/` contains all domain logic and implementation details
- **Separation**: Clear boundary between orchestration (what) and implementation (how)

### 2. **Configuration-Driven Design**
- **Environment Variables**: Runtime configuration via `.env` file
- **YAML Configuration**: Structured configuration for complex parameters
- **Runtime Validation**: Configuration validation with helpful error messages

### 3. **Resource Management Pattern**
- **Context Managers**: Proper resource cleanup using `with` statements
- **RAII Pattern**: Resource Acquisition Is Initialization for automatic cleanup
- **Monitoring**: Built-in resource usage tracking and limits

### 4. **Error Handling Strategy**
- **Graceful Degradation**: Continue operation with reduced functionality when possible
- **Detailed Diagnostics**: Comprehensive error information for debugging
- **Recovery Mechanisms**: Automatic retry and fallback strategies

### 5. **Performance Optimization**
- **Caching Strategy**: Intelligent caching of expensive operations
- **Batch Processing**: Efficient handling of multiple inputs
- **Resource Pooling**: Reuse of expensive resources (connections, models)

### 6. **Testing Strategy**
- **Unit Tests**: Individual function testing with mocked dependencies
- **Integration Tests**: End-to-end pipeline testing with real data
- **Performance Tests**: Timing and resource usage validation

### 7. **Documentation Pattern**
- **API Documentation**: Complete function and class documentation
- **Usage Examples**: Practical examples for all major use cases
- **Troubleshooting Guide**: Common issues and solutions
- **Performance Guidelines**: Optimization recommendations

### 8. **MCP Integration Pattern**
- **Tool Registration**: Proper tool registration in `mcp.py`
- **Standard Interface**: Consistent interface across all MCP tools
- **Error Handling**: MCP-compliant error responses

### 9. **Pipeline Integration**
- **Standardized Interface**: Consistent function signatures across modules
- **Logging Integration**: Structured logging compatible with pipeline requirements
- **Output Management**: Proper output directory structure and file management

### 10. **Security Pattern**
- **Input Validation**: Comprehensive validation of all inputs
- **Secure Defaults**: Safe default configurations
- **Access Control**: Proper file and resource access controls

---

**Last Updated**: [Date in ISO format, e.g., "2025-10-01"]
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready / 🔄 In Development / ⚠️ Experimental
**Version**: [Current version, e.g., "1.0.0"]
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern



