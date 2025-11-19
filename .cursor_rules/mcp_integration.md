# MCP (Model Context Protocol) Integration

> **Environment Note**: MCP workflows assume the `uv` environment context. Launch MCP tooling via `uv` (e.g., `uv run src/21_mcp.py`, `uv pip install -e .`) to keep environment, dependencies, and MCP registration consistent.

## Overview

The Model Context Protocol (MCP) enables standardized tool registration and discovery across the GNN pipeline. Every applicable module exposes its core functionality through MCP tools, creating a unified interface for external systems and AI assistance.

## Universal MCP Pattern

Every applicable module includes `mcp.py` with:
- **Tool Registration**: Module-specific MCP tool definitions with complete metadata
- **Function Exposure**: Key module functions exposed as MCP tools with parameter documentation
- **Standardized Interface**: Consistent MCP API across all modules following protocol specifications
- **Real Integration**: Functional MCP tools with genuine implementations, not placeholders
- **Error Handling**: Comprehensive error handling with clear error types and messages
- **Type Safety**: Full type hints for all MCP tool parameters and responses

## MCP System Architecture

### Central Registry
- **Core Implementation**: `src/mcp/` contains MCP server and registry implementation
- **Centralized Server**: Unified MCP server managing all module tool registrations
- **Protocol Compliance**: Full Model Context Protocol specification compliance
- **Tool Catalog**: Complete tool catalog with descriptions and metadata

### Module Integration
- **Module Exports**: Each module's `mcp.py` registers with central system
- **Tool Discovery**: Automatic tool discovery via centralized registry
- **Dynamic Loading**: Dynamic module loading and tool registration on startup
- **Namespace Isolation**: Proper namespace isolation for module-specific tools

### Tool Discovery
- **Automatic Discovery**: Tools automatically discovered and registered from all modules
- **Dynamic Registration**: Registration happens at server startup without manual configuration
- **Tool Catalog Queries**: External systems can query available tools and parameters
- **Version Management**: Tool versioning and backward compatibility support

### API Consistency
- **Standardized Requests**: All tools follow standard MCP request/response format
- **Parameter Validation**: Comprehensive parameter validation with clear error messages
- **Return Format**: Standardized JSON response format across all tools
- **Error Responses**: Consistent error response format with error types and context

## MCP Tool Categories

### GNN Processing Tools
- **gnn_parse**: Parse GNN files and return structured model
- **gnn_validate**: Validate GNN syntax and structure
- **gnn_serialize**: Serialize model to specified format
- **gnn_discover**: Discover GNN files in directory

### Rendering and Execution Tools
- **render_to_pymdp**: Generate PyMDP code from GNN
- **render_to_rxinfer**: Generate RxInfer.jl code from GNN
- **render_to_activeinference**: Generate ActiveInference.jl code
- **execute_simulation**: Execute rendered simulation script
- **validate_execution**: Validate simulation results

### Visualization Tools
- **visualize_graph**: Generate network graph visualization
- **visualize_matrices**: Generate matrix heatmap visualization
- **visualize_ontology**: Generate ontology diagram
- **export_visualization**: Export visualization to format

### LLM and Analysis Tools
- **analyze_model**: AI-powered model analysis
- **explain_model**: Generate natural language explanation
- **suggest_improvements**: Suggest model improvements
- **validate_semantic**: Semantic validation analysis

### Export and Conversion Tools
- **export_to_json**: Export model to JSON format
- **export_to_xml**: Export model to XML format
- **export_to_yaml**: Export model to YAML format
- **convert_formats**: Convert between model formats

### Audio Generation Tools
- **generate_sapf**: Generate SAPF audio representation
- **sonify_model**: Convert model structure to audio
- **analyze_audio**: Audio analysis and validation

### Website and Documentation Tools
- **generate_website**: Generate HTML website from artifacts
- **generate_report**: Generate comprehensive analysis report
- **export_documentation**: Export model documentation

### Utility Tools
- **get_module_info**: Get module metadata and capabilities
- **list_available_tools**: List all available MCP tools
- **validate_configuration**: Validate system configuration

## MCP Tool Implementation Pattern

### Standard Tool Structure
```python
def tool_name_mcp(mcp_instance_ref, **kwargs) -> Dict[str, Any]:
    """
    MCP tool for [function].
    
    Args:
        mcp_instance_ref: MCP instance reference
        **kwargs: Function-specific arguments
        
    Returns:
        Standardized MCP response dictionary
    """
    try:
        # Validate required arguments
        required_args = ['arg1', 'arg2']
        for arg in required_args:
            if arg not in kwargs:
                return {
                    "success": False,
                    "error": f"Missing required argument: {arg}",
                    "error_type": "missing_argument"
                }
        
        # Call core function with proper error handling
        result = core_function(**kwargs)
        
        return {
            "success": True,
            "result": result,
            "operation": "[function_name]",
            "module": "[module_name]"
        }
        
    except ValueError as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": "validation_error"
        }
    except Exception as e:
        logger.error(f"MCP tool error: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "execution_error"
        }

# MCP tool registry with metadata
MCP_TOOLS = {
    "[tool_name]": {
        "function": tool_name_mcp,
        "description": "[Function description]",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {
                    "type": "string",
                    "description": "Argument description"
                },
                "arg2": {
                    "type": "boolean",
                    "description": "Argument description"
                }
            },
            "required": ["arg1"]
        }
    }
}
```

## MCP Error Handling

### Error Types
- **missing_argument**: Required argument not provided
- **validation_error**: Parameter validation failed
- **execution_error**: Tool execution failed
- **dependency_error**: Required dependency not available
- **not_found**: Resource not found
- **permission_error**: Access permission denied
- **timeout_error**: Operation timed out

### Error Response Format
```python
{
    "success": False,
    "error": "Human-readable error message",
    "error_type": "error_type_name",
    "details": {
        "argument": "arg_name",
        "expected": "expected_type",
        "provided": "provided_value"
    }
}
```

## MCP Server Configuration

### Server Setup
- **Port Configuration**: Configurable MCP server port (default: 8000)
- **Host Configuration**: Configurable host (default: localhost)
- **SSL/TLS Support**: Optional SSL/TLS for secure communication
- **Authentication**: Optional authentication for secure tool access
- **Rate Limiting**: Rate limiting for public endpoints
- **Logging**: Comprehensive server logging and monitoring

### Tool Registration
- **Automatic Discovery**: Tools discovered at server startup
- **Dynamic Loading**: Module loading and registration on demand
- **Tool Metadata**: Complete metadata for each tool with descriptions
- **Parameter Documentation**: Full parameter documentation for tool discovery
- **Version Information**: Tool version and compatibility information

## MCP Integration Points

### Module Exports
Each module provides MCP integration via:
1. **mcp.py file**: Contains MCP tool definitions
2. **Tool registration**: Register tools with central MCP server
3. **Tool implementation**: Delegate to core module functions
4. **Error handling**: Comprehensive error handling for tool execution

### Server Integration
- **Central Registry**: All tools registered with central MCP server
- **Tool Discovery**: Tools discoverable via protocol queries
- **Request Routing**: Route tool requests to appropriate modules
- **Response Handling**: Standardized response handling and formatting

## Usage Examples

### Querying Available Tools
```bash
# List all available MCP tools
curl http://localhost:8000/tools

# Get tool metadata
curl http://localhost:8000/tools/gnn_parse
```

### Calling MCP Tools
```bash
# Parse GNN file via MCP
curl -X POST http://localhost:8000/tools/gnn_parse \
  -H "Content-Type: application/json" \
  -d '{"file_path": "model.md"}'

# Generate PyMDP code via MCP
curl -X POST http://localhost:8000/tools/render_to_pymdp \
  -H "Content-Type: application/json" \
  -d '{"model_data": {...}}'
```

### Tool Response Format
```json
{
    "success": true,
    "result": {
        "parsed_model": {...},
        "metadata": {...},
        "validation_status": "valid"
    },
    "operation": "gnn_parse",
    "module": "gnn",
    "execution_time_ms": 123
}
```

## MCP Tool Categories by Module

### Template Module (Step 0)
- `get_template_info`: Get template module information
- `list_template_options`: List available pipeline templates

### Setup Module (Step 1)
- `validate_environment`: Validate system environment
- `check_dependencies`: Check package dependencies
- `install_dependencies`: Install required dependencies

### Tests Module (Step 2)
- `run_tests`: Run test suite
- `check_coverage`: Check test coverage
- `list_test_categories`: List available test categories

### GNN Module (Step 3)
- `gnn_parse`: Parse GNN files
- `gnn_validate`: Validate GNN structure
- `gnn_serialize`: Serialize to format
- `gnn_discover`: Discover GNN files

### Type Checker Module (Step 5)
- `validate_gnn_types`: Validate type consistency
- `check_syntax`: Check GNN syntax
- `estimate_resources`: Estimate computational resources

### Export Module (Step 7)
- `export_to_json`: Export to JSON format
- `export_to_xml`: Export to XML format
- `export_to_yaml`: Export to YAML format
- `export_to_graphml`: Export to GraphML format

### Visualization Module (Step 8)
- `visualize_graph`: Generate graph visualization
- `visualize_matrices`: Generate matrix visualization
- `export_visualization`: Export visualization

### Render Module (Step 11)
- `render_pymdp`: Generate PyMDP code
- `render_rxinfer`: Generate RxInfer code
- `render_activeinference`: Generate ActiveInference code
- `render_discopy`: Generate DisCoPy code
- `render_jax`: Generate JAX code

### LLM Module (Step 13)
- `analyze_gnn`: AI-powered GNN analysis
- `explain_model`: Generate model explanation
- `suggest_improvements`: Suggest model improvements

### Audio Module (Step 15)
- `generate_sapf`: Generate SAPF audio
- `sonify_model`: Convert model to audio

## MCP Protocol Compliance

### Protocol Version
- **Version**: Model Context Protocol 1.0+
- **Compliance**: Full protocol compliance with standard tools and features
- **Extensions**: Support for protocol extensions and custom tools

### Standard Features
- **Tool Discovery**: Complete tool discovery via standard protocol
- **Parameter Validation**: Parameter validation per protocol specification
- **Error Handling**: Error handling per protocol specification
- **Authentication**: Authentication support per protocol specification

## Performance Considerations

### Tool Execution
- **Timeout**: Default 30-second timeout per tool execution
- **Resource Limits**: Memory and CPU limits per tool
- **Connection Pooling**: Connection pooling for efficiency
- **Caching**: Result caching for repeated tool calls

### Monitoring
- **Tool Metrics**: Execution time, success rate, error rate
- **Server Metrics**: Connection count, request rate, resource usage
- **Performance Tracking**: Detailed performance logging

## Security and Access Control

### Access Control
- **Tool Permissions**: Permission-based tool access
- **Rate Limiting**: Per-client rate limiting
- **Authentication**: Optional API key authentication
- **HTTPS Support**: TLS/SSL for secure communication

### Input Validation
- **Parameter Validation**: Comprehensive input validation
- **Type Checking**: Runtime type checking for parameters
- **Sanitization**: Input sanitization for security

---

**Status**: ✅ Production Ready - Full MCP integration with comprehensive tool support across all pipeline modules.
