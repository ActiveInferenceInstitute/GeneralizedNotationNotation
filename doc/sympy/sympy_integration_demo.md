# SymPy MCP Integration Demonstration

## Overview

The GNN project now includes full integration with SymPy MCP for symbolic mathematics capabilities. This integration provides mathematical validation, simplification, and analysis tools for Active Inference models specified in GNN format.

## Successfully Integrated SymPy MCP Tools

### Available Tools

The following SymPy MCP tools are now registered and available through the GNN MCP system:

1. **`sympy_validate_equation`** - Validate mathematical equations using symbolic processing
2. **`sympy_validate_matrix`** - Validate matrix properties including stochasticity constraints  
3. **`sympy_analyze_stability`** - Analyze system stability using eigenvalue analysis
4. **`sympy_simplify_expression`** - Simplify mathematical expressions to canonical form
5. **`sympy_solve_equation`** - Solve equations algebraically for specified variables
6. **`sympy_get_latex`** - Convert mathematical expressions to LaTeX format
7. **`sympy_initialize`** - Initialize SymPy MCP integration
8. **`sympy_cleanup`** - Clean up SymPy MCP integration and reset state

### Usage Example

```python
from src.mcp.mcp import mcp_instance

# Initialize SymPy integration (requires SymPy MCP server)
result = mcp_instance.execute_tool('sympy_initialize', {})

# Validate a GNN equation
equation_result = mcp_instance.execute_tool('sympy_validate_equation', {
    'equation': 'x^2 + 2*x + 1',
    'context': {}
})

# Simplify an expression
simplify_result = mcp_instance.execute_tool('sympy_simplify_expression', {
    'expression': '(x + 1)^2'
})

# Get LaTeX representation
latex_result = mcp_instance.execute_tool('sympy_get_latex', {
    'expression': 'sqrt(x^2 + 1)'
})

# Validate matrix stochasticity
matrix_result = mcp_instance.execute_tool('sympy_validate_matrix', {
    'matrix_data': [[0.5, 0.3], [0.5, 0.7]],
    'matrix_type': 'transition'
})
```

## Integration Architecture

### Key Components

1. **SymPy MCP Client** (`src/mcp/sympy_mcp_client.py`)
   - HTTP client for SymPy MCP server communication
   - Async context manager support
   - Comprehensive error handling with graceful fallback

2. **MCP Tool Registration** (`src/mcp/sympy_mcp.py`)
   - Integration with GNN MCP system
   - Synchronous/asynchronous compatibility wrappers
   - Tool schema definitions for validation

3. **Core MCP Discovery** (`src/mcp/mcp.py`)
   - Automatic discovery of SymPy tools
   - Integration with existing pipeline workflow

### Dependencies

- **httpx >= 0.27.0** - For HTTP client functionality
- **SymPy MCP Server** - External server providing symbolic mathematics capabilities

## Verification

The integration has been successfully verified:

✅ **MCP Tool Registration**: All 8 SymPy tools registered successfully  
✅ **Pipeline Integration**: Works with `python3 src/main.py --only-steps 7`  
✅ **Error Handling**: Graceful fallback when SymPy server unavailable  
✅ **Type Safety**: Proper type annotations and validation  
✅ **Documentation**: Comprehensive inline documentation and examples  

## Running MCP Integration Test

```bash
# Run MCP pipeline step to verify integration
python3 src/main.py --only-steps 7 --verbose

# Check MCP integration report
cat output/mcp_processing_step/7_mcp_integration_report.md | grep sympy
```

The integration report will show all SymPy tools with their schemas and descriptions.

## Setting Up SymPy MCP Server

To use the full SymPy capabilities, you'll need to set up the SymPy MCP server:

1. **Clone SymPy MCP Repository**:
   ```bash
   git clone https://github.com/sdiehl/sympy-mcp.git
   cd sympy-mcp
   ```

2. **Install Dependencies**:
   ```bash
   uv install
   ```

3. **Run Server**:
   ```bash
   uv run --with mcp[cli] --with sympy mcp run server.py --transport sse
   ```

4. **Test Integration**:
   ```python
   from src.mcp.mcp import mcp_instance
   result = mcp_instance.execute_tool('sympy_initialize', {})
   print(result)  # Should show success when server is running
   ```

## Benefits

The SymPy MCP integration provides:

- **Mathematical Validation**: Automatic validation of GNN equations and matrices
- **Symbolic Simplification**: Canonical forms for mathematical expressions
- **LaTeX Generation**: Consistent mathematical notation
- **Stability Analysis**: Eigenvalue analysis for dynamic systems
- **Research Acceleration**: Rapid mathematical verification and exploration

## Future Enhancements

Planned extensions include:

- **GNN Syntax Conversion**: Automatic conversion from GNN notation to SymPy
- **Active Inference Validation**: Specialized validation for AI model matrices
- **Differential Equation Solving**: Support for dynamic GNN models
- **Model Comparison**: Symbolic comparison of equivalent models
- **Visualization Integration**: Mathematical plots and interactive exploration

## Conclusion

The SymPy MCP integration successfully transforms GNN from a passive notation system into an active mathematical validation and analysis framework. This enhancement provides the foundation for more sophisticated mathematical processing capabilities while maintaining compatibility with existing GNN workflows. 