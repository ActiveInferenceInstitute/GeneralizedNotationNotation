# Model Context Protocol (MCP) Integration

## Overview
The GNN project implements Model Context Protocol (MCP) to provide structured APIs for AI assistants and LLM integrations. MCP enables external tools and AI systems to interact with GNN processing capabilities through standardized interfaces.

## Architecture

### Core MCP System
- **Location**: `src/mcp/`
- **Main Module**: `mcp.py` - Central MCP instance and tool registration
- **Server Components**: HTTP and STDIO server implementations
- **Tool Discovery**: Automatic registration from functional modules

### Distributed MCP Modules
Each functional module includes its own `mcp.py` file that registers domain-specific tools:

```
src/
├── mcp/               # Core MCP infrastructure
├── export/mcp.py      # Export format tools
├── gnn/mcp.py         # GNN parsing and validation tools
├── ontology/mcp.py    # Ontology processing tools
├── visualization/mcp.py # Visualization generation tools
└── llm/mcp.py         # LLM integration tools
```

## Available MCP Tools

### Core GNN Processing (`gnn/mcp.py`)
- `parse_gnn_file` - Parse and validate GNN files
- `validate_gnn_syntax` - Syntax checking and error reporting
- `extract_model_metadata` - Extract model information
- `list_gnn_examples` - Discover available example models

### Export Tools (`export/mcp.py`)
- `export_to_json` - Convert GNN to JSON format
- `export_to_xml` - Convert GNN to XML format  
- `export_to_graphml` - Generate GraphML for network analysis
- `list_export_formats` - Show available export formats

### Visualization Tools (`visualization/mcp.py`)
- `generate_factor_graph` - Create factor graph diagrams
- `visualize_model_structure` - Generate model structure plots
- `create_connection_diagram` - Visualize variable connections
- `export_visualization_formats` - List available image formats

### Ontology Tools (`ontology/mcp.py`)
- `validate_ontology_terms` - Check Active Inference ontology compliance
- `map_gnn_to_ontology` - Create ontology mappings
- `query_ontology_database` - Search ontology terms
- `generate_ontology_report` - Create ontology analysis reports

### LLM Integration Tools (`llm/mcp.py`)
- `analyze_model_with_llm` - AI-powered model analysis
- `generate_model_explanation` - Natural language model descriptions
- `suggest_model_improvements` - AI recommendations for model enhancement
- `compare_models` - LLM-based model comparison

## Using MCP Tools

### Direct Python Access
```python
from src.mcp import mcp_instance

# List all available tools
tools = mcp_instance.tools
print(f"Available tools: {list(tools.keys())}")

# Use a specific tool
result = mcp_instance.call_tool("parse_gnn_file", {
    "file_path": "examples/basic_model.md"
})
```

### HTTP Server
```bash
# Start MCP HTTP server
python src/mcp/server_http.py --port 8000

# Make HTTP requests
curl -X POST http://localhost:8000/tools/parse_gnn_file \
  -H "Content-Type: application/json" \
  -d '{"file_path": "examples/basic_model.md"}'
```

### STDIO Server (for AI assistants)
```bash
# Start STDIO server for AI assistant integration
python src/mcp/server_stdio.py
```

### Command Line Interface
```bash
# Use MCP CLI tool
python src/mcp/cli.py list-tools
python src/mcp/cli.py call parse_gnn_file examples/basic_model.md
```

## Tool Schema Examples

### GNN File Parsing Tool
```json
{
  "name": "parse_gnn_file",
  "description": "Parse and validate a GNN file, extracting model components",
  "inputSchema": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "string",
        "description": "Path to the GNN file to parse"
      },
      "validate_syntax": {
        "type": "boolean", 
        "description": "Whether to perform syntax validation",
        "default": true
      }
    },
    "required": ["file_path"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "model_name": {"type": "string"},
      "state_space": {"type": "object"},
      "connections": {"type": "array"},
      "validation_errors": {"type": "array"}
    }
  }
}
```

### Model Visualization Tool
```json
{
  "name": "generate_factor_graph",
  "description": "Generate a factor graph visualization of a GNN model",
  "inputSchema": {
    "type": "object",
    "properties": {
      "gnn_file": {"type": "string"},
      "output_format": {
        "type": "string",
        "enum": ["png", "svg", "pdf"],
        "default": "png"
      },
      "layout": {
        "type": "string", 
        "enum": ["spring", "circular", "hierarchical"],
        "default": "spring"
      }
    },
    "required": ["gnn_file"]
  }
}
```

## Pipeline Integration

### Step 7: MCP Integration Analysis
The pipeline includes dedicated MCP analysis:

```bash
# Run MCP integration check
python src/main.py --only-steps 7

# Generate MCP integration report
python src/7_mcp.py --output-dir output/
```

The MCP step generates comprehensive reports including:
- Available MCP tools across all modules
- Tool schemas and documentation
- Integration status and health checks
- API usage examples

## Development Guidelines

### Adding New MCP Tools

1. **Create tool function**:
```python
# In your module's mcp.py file
def my_new_tool(param1: str, param2: int = 10) -> dict:
    """
    Description of what this tool does.
    
    Args:
        param1: Description of parameter
        param2: Optional parameter with default
        
    Returns:
        Dictionary with results
    """
    # Implementation here
    return {"result": "success"}
```

2. **Register the tool**:
```python
def register_tools():
    """Register all tools from this module."""
    tools = {
        "my_new_tool": my_new_tool
    }
    
    from src.mcp import mcp_instance
    for name, func in tools.items():
        mcp_instance.register_tool(name, func)
```

3. **Test the tool**:
```bash
python src/mcp/cli.py call my_new_tool '{"param1": "test"}'
```

### Tool Design Principles

- **Atomic Functions**: Each tool should do one thing well
- **Clear Schemas**: Provide complete input/output schemas
- **Error Handling**: Return structured error information
- **Documentation**: Include comprehensive docstrings
- **Type Hints**: Use proper type annotations for automatic schema generation

### Security Considerations

- **Input Validation**: Validate all inputs against schemas
- **File Access**: Restrict file operations to safe directories
- **API Keys**: Handle sensitive credentials securely
- **Resource Limits**: Implement timeouts and resource constraints

## Troubleshooting

### Common Issues

1. **Tool Not Found**
   - Check if module's `register_tools()` was called
   - Verify tool is in the module's MCP registration

2. **Schema Validation Errors**
   - Ensure input matches the tool's schema
   - Check required parameters are provided

3. **Import Errors**
   - Verify all dependencies are installed
   - Check Python path includes src/ directory

### Debug Mode

```bash
# Run with verbose MCP logging
python src/mcp/cli.py --verbose list-tools

# Test tool with debugging
python src/mcp/cli.py --debug call tool_name parameters
```

## API Reference

### Core MCP Instance
```python
from src.mcp import mcp_instance

# Tool management
mcp_instance.register_tool(name, function)
mcp_instance.list_tools()
mcp_instance.call_tool(name, parameters)

# Schema inspection  
mcp_instance.get_tool_schema(name)
mcp_instance.validate_input(name, parameters)
```

### Server Components
- `server_http.py` - HTTP REST API server
- `server_stdio.py` - STDIO protocol server for AI assistants
- `cli.py` - Command-line interface

### Configuration
- Tool timeout settings
- Server port configuration
- API authentication (if enabled)
- Resource limits and constraints

## Integration Examples

### LLM Assistant Integration
```python
# Example of AI assistant using MCP tools
from src.mcp import mcp_instance

def analyze_user_model(file_path: str) -> str:
    # Parse the model
    parse_result = mcp_instance.call_tool("parse_gnn_file", {
        "file_path": file_path
    })
    
    # Generate visualization
    viz_result = mcp_instance.call_tool("generate_factor_graph", {
        "gnn_file": file_path,
        "output_format": "png"
    })
    
    # Get AI analysis
    analysis = mcp_instance.call_tool("analyze_model_with_llm", {
        "model_data": parse_result,
        "analysis_type": "comprehensive"
    })
    
    return analysis["explanation"]
```

### External Tool Integration
```bash
# External Python script using MCP HTTP API
import requests

response = requests.post("http://localhost:8000/tools/validate_gnn_syntax", 
                        json={"file_path": "my_model.md"})
result = response.json()
print(f"Validation result: {result}")
``` 