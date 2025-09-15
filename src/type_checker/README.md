# Type Checker Module

The Type Checker module provides comprehensive validation, analysis, and type checking capabilities for Generalized Notation Notation (GNN) files in the GNN Processing Pipeline.

## Overview

This module implements robust type checking and validation for GNN files, including syntax validation, type consistency checking, resource estimation, and performance analysis. It follows the thin orchestrator pattern and provides both programmatic APIs and MCP (Model Context Protocol) integration.

## Key Features

### Core Functionality
- **GNN File Validation**: Comprehensive syntax and semantic validation
- **Type Analysis**: Variable type checking and dimension validation
- **Connection Analysis**: Pattern analysis and complexity estimation
- **Resource Estimation**: Computational and memory requirement estimation
- **Performance Analysis**: Processing metrics and optimization suggestions

### Advanced Capabilities
- **Error Recovery**: Graceful handling of malformed input
- **Performance Monitoring**: Real-time metrics and performance tracking
- **MCP Integration**: External tool integration via Model Context Protocol
- **Comprehensive Logging**: Detailed logging with correlation IDs
- **Validation Modes**: Standard and strict validation modes

## Architecture

### Module Structure
```
src/type_checker/
├── __init__.py          # Module initialization and exports
├── processor.py         # Core type checking functionality
├── analysis_utils.py    # Analysis utilities and complexity estimation
├── mcp.py              # MCP tool registration and execution
└── README.md           # This documentation
```

### Key Components

#### 1. GNNTypeChecker (processor.py)
The main class providing comprehensive type checking capabilities:

```python
from src.type_checker import GNNTypeChecker

# Initialize with options
checker = GNNTypeChecker(strict_mode=True, verbose=True)

# Validate GNN files
success = checker.validate_gnn_files(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/type_check"),
    verbose=True
)

# Validate single file
result = checker.validate_single_gnn_file(
    file_path=Path("model.gnn"),
    verbose=True
)
```

#### 2. Analysis Utilities (analysis_utils.py)
Specialized functions for detailed analysis:

```python
from src.type_checker import (
    analyze_variable_types,
    analyze_connections,
    estimate_computational_complexity
)

# Analyze variable types
variables = [
    {"name": "state", "type": "belief", "data_type": "float", "dimensions": [10, 1]},
    {"name": "action", "type": "action", "data_type": "int", "dimensions": [5]}
]
type_analysis = analyze_variable_types(variables)

# Analyze connections
connections = [
    {"type": "transition", "source_variables": ["state", "action"], "target_variables": ["state"]}
]
conn_analysis = analyze_connections(connections)

# Estimate complexity
complexity = estimate_computational_complexity(type_analysis, conn_analysis)
```

#### 3. MCP Integration (mcp.py)
Model Context Protocol integration for external tools:

```python
from src.type_checker import (
    register_mcp_tools,
    execute_mcp_tool,
    list_available_tools
)

# Register MCP tools
tools = register_mcp_tools()

# Execute MCP tool
result = execute_mcp_tool("validate_gnn_file", {
    "file_path": "/path/to/model.gnn",
    "strict_mode": True
})

# List available tools
available_tools = list_available_tools()
```

## Usage Examples

### Basic Type Checking

```python
from pathlib import Path
from src.type_checker import GNNTypeChecker

# Initialize checker
checker = GNNTypeChecker(strict_mode=False, verbose=True)

# Validate directory of GNN files
success = checker.validate_gnn_files(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/type_check"),
    verbose=True
)

if success:
    print("Type checking completed successfully")
else:
    print("Type checking failed - check logs for details")
```

### Advanced Analysis

```python
from src.type_checker import (
    analyze_variable_types,
    analyze_connections,
    estimate_computational_complexity
)

# Sample GNN model data
variables = [
    {
        "name": "belief_state",
        "type": "belief",
        "data_type": "float",
        "dimensions": [10, 1],
        "description": "Agent belief state"
    },
    {
        "name": "actions",
        "type": "action",
        "data_type": "int",
        "dimensions": [5],
        "description": "Available actions"
    }
]

connections = [
    {
        "type": "transition",
        "source_variables": ["belief_state", "actions"],
        "target_variables": ["belief_state"],
        "description": "State transition function"
    }
]

# Perform analysis
type_analysis = analyze_variable_types(variables)
conn_analysis = analyze_connections(connections)
complexity = estimate_computational_complexity(type_analysis, conn_analysis)

# Print results
print(f"Total variables: {type_analysis['total_variables']}")
print(f"Total connections: {conn_analysis['total_connections']}")
print(f"Estimated memory: {complexity['resource_requirements']['ram_gb_recommended']} GB")
```

### MCP Tool Usage

```python
from src.type_checker import execute_mcp_tool

# Validate a single file via MCP
result = execute_mcp_tool("validate_gnn_file", {
    "file_path": "/path/to/model.gnn",
    "strict_mode": True,
    "verbose": True
})

if result["success"]:
    print("File validation successful")
    print(f"Validation result: {result['result']}")
else:
    print(f"Validation failed: {result['error']}")
```

## Configuration Options

### GNNTypeChecker Parameters

- **strict_mode** (bool): Enable strict validation rules
- **verbose** (bool): Enable verbose logging and output

### Validation Rules

The type checker supports various validation rules:

- **Type Validation**: Validates variable types against allowed types
- **Dimension Validation**: Ensures dimensions are positive integers
- **Name Validation**: Validates variable and connection names
- **Consistency Checking**: Checks for duplicate names and type consistency
- **Syntax Validation**: Validates GNN syntax and structure

### Supported File Types

- `.md` - Markdown files with GNN content
- `.gnn` - Native GNN files
- `.txt` - Text files with GNN content

## Output Structure

### Validation Results

The type checker generates comprehensive output including:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "processed_files": 3,
  "success": true,
  "errors": [],
  "warnings": [],
  "validation_results": [...],
  "type_analysis": [...],
  "performance_metrics": {
    "files_processed": 3,
    "total_processing_time": 1.23,
    "errors_encountered": 0,
    "warnings_generated": 2
  },
  "summary_statistics": {
    "total_variables": 15,
    "total_connections": 8,
    "valid_files": 3,
    "invalid_files": 0,
    "type_errors": 0,
    "warnings_count": 2
  }
}
```

### Analysis Results

Type analysis provides detailed metrics:

```json
{
  "total_variables": 5,
  "type_distribution": {
    "belief": 2,
    "action": 1,
    "observation": 1,
    "reward": 1
  },
  "dimension_analysis": {
    "max_dimensions": 2,
    "avg_dimensions": 1.4,
    "dimension_distribution": {
      "1D": 3,
      "2D": 2
    }
  },
  "complexity_metrics": {
    "total_elements": 25,
    "estimated_memory_bytes": 200,
    "estimated_memory_mb": 0.0002,
    "estimated_memory_gb": 0.0000002
  }
}
```

## Error Handling

The type checker provides comprehensive error handling:

### Error Types
- **FileNotFoundError**: Missing input files
- **SyntaxError**: Invalid GNN syntax
- **TypeError**: Invalid type definitions
- **ValueError**: Invalid values or dimensions
- **ValidationError**: Custom validation failures

### Error Recovery
- Graceful degradation for malformed input
- Detailed error messages with context
- Partial processing when possible
- Comprehensive logging for debugging

## Performance Considerations

### Optimization Features
- Efficient parsing algorithms
- Memory-conscious processing
- Parallel processing support
- Caching for repeated operations

### Performance Metrics
- Processing time per file
- Memory usage tracking
- Error rate monitoring
- Throughput measurements

### Scaling Guidelines
- Small models (< 100 variables): Any system
- Medium models (100-1000 variables): 4GB RAM, 2 CPU cores
- Large models (1000+ variables): 8GB+ RAM, 4+ CPU cores
- Very large models (10000+ variables): 16GB+ RAM, 8+ CPU cores

## Testing

The module includes comprehensive tests:

```bash
# Run all type checker tests
pytest src/tests/test_type_checker_overall.py -v

# Run specific test categories
pytest src/tests/test_type_checker_overall.py::TestTypeCheckerAnalysisUtils -v
pytest src/tests/test_type_checker_overall.py::TestTypeCheckerProcessor -v
pytest src/tests/test_type_checker_overall.py::TestTypeCheckerIntegration -v
```

### Test Coverage
- Unit tests for all functions
- Integration tests for workflows
- Performance tests for large datasets
- Error handling tests
- MCP integration tests

## Integration with Pipeline

The type checker integrates seamlessly with the GNN Processing Pipeline:

### Pipeline Step 5
The type checker is used in pipeline step 5 (`5_type_checker.py`):

```python
# Step 5 orchestrator imports from type_checker module
from type_checker.analysis_utils import (
    analyze_variable_types,
    analyze_connections,
    estimate_computational_complexity,
)
```

### Input/Output Flow
1. **Input**: Parsed GNN data from step 3
2. **Processing**: Type checking and analysis
3. **Output**: Validation results and analysis data
4. **Integration**: Results used by subsequent pipeline steps

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Ensure proper path setup
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent))
   ```

2. **File Not Found**
   ```python
   # Check file paths
   file_path = Path("model.gnn")
   if not file_path.exists():
       print(f"File not found: {file_path}")
   ```

3. **Memory Issues**
   ```python
   # Use strict mode for large files
   checker = GNNTypeChecker(strict_mode=True)
   ```

### Debug Mode

Enable verbose logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

checker = GNNTypeChecker(verbose=True)
```

## Contributing

When contributing to the type checker module:

1. Follow the thin orchestrator pattern
2. Add comprehensive tests for new functionality
3. Update documentation for API changes
4. Ensure MCP integration compatibility
5. Follow error handling best practices

## License

This module is part of the GNN Processing Pipeline and follows the same license terms.