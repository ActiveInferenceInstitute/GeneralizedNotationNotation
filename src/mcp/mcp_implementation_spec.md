# GNN Model Context Protocol (MCP) Implementation Specification

This document provides the technical specification for the Model Context Protocol (MCP) implementation in the GeneralizedNotationNotation (GNN) project. This implementation enables AI assistants and other LLM-powered applications to interact with the GNN toolkit through a standardized protocol.

## 1. Architecture Overview

The GNN MCP implementation follows a modular, discoverable architecture with the following components:

### 1.1 Core Components
- **Core MCP Module** (`mcp.py`): Central implementation that handles tool registration, resource management, and module discovery.
- **MCP Servers**: Implementations for stdio and HTTP transports (`server_stdio.py`, `server_http.py`).
- **Command-Line Interface** (`cli.py`): Tools to start servers and execute commands.
- **Module Integrations**: MCP modules (`mcp.py` files) in various GNN components that expose domain-specific functionality.
- **Meta MCP Module** (`meta_mcp.py`): An MCP module that exposes the server's own metadata and status.

### 1.2 Operational Flow
1. LLM client connects to MCP server (stdio or HTTP).
2. MCP server (via `mcp.py`) discovers and loads available modules, including `meta_mcp.py` and other functional modules (e.g., `visualization/mcp.py`, `export/mcp.py`).
3. Client requests capabilities (e.g., by calling `get_mcp_server_capabilities` tool) to learn available tools and resources.
4. Client executes tools or requests resources.
5. Server routes requests to appropriate modules and returns results.

## 2. Core MCP Implementation (`mcp.py`)

### 2.1 Classes

#### `MCP` Class
The main class that implements the core MCP functionality:
```python
class MCP:
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.modules: Dict[str, Any] = {}
        
    def discover_modules(self):
        # Discover and load MCP modules (including meta_mcp.py)
        
    def register_tool(self, name, func, schema, description):
        # Register a new tool
        
    def register_resource(self, uri_template, retriever, description):
        # Register a new resource
        
    def execute_tool(self, tool_name, params):
        # Execute a registered tool
        
    def get_resource(self, uri):
        # Retrieve a resource
        
    def get_capabilities(self) -> Dict[str, Any]: # This is also exposed as a tool by meta_mcp.py
        # Return available capabilities
```

#### `MCPTool` Class
Represents a callable tool:
```python
class MCPTool:
    def __init__(self, name, func, schema, description):
        self.name = name
        self.func = func
        self.schema = schema
        self.description = description
```

#### `MCPResource` Class
Represents a retrievable resource:
```python
class MCPResource:
    def __init__(self, uri_template, retriever, description):
        self.uri_template = uri_template
        self.retriever = retriever
        self.description = description
```

### 2.2 Module Discovery
The MCP implementation uses a module discovery mechanism to dynamically load MCP capabilities:

1. Searches all relevant directories in the `src/` folder (including `src/mcp` itself for `meta_mcp.py`).
2. Looks for `mcp.py` (or `meta_mcp.py`) files in each directory.
3. Imports the module and calls its `register_tools(mcp_instance)` function.
4. Maintains a registry of all loaded modules.

## 3. Server Implementations (`server_stdio.py`, `server_http.py`)

Details as previously specified, handling JSON-RPC 2.0 messages over their respective transports.

## 4. Integrated Modules and Tools

### 4.1 Meta MCP Module (`meta_mcp.py`)
Exposes the server's own operational data:
```python
# Tools
- get_mcp_server_capabilities(): Returns full server capabilities.
- get_mcp_server_status(): Returns server uptime, loaded modules, etc.
- get_mcp_server_auth_status(): Returns current authentication status.
- get_mcp_server_encryption_status(): Returns current transport encryption status.
```

### 4.2 GNN Documentation Module (`gnn/mcp.py`)
Exposes GNN core documentation files:
```python
# Tools
- get_gnn_documentation(doc_name: Literal["file_structure", "punctuation"])

# Resources
- gnn://documentation/{doc_name}
```

### 4.3 GNN Type Checker Module (`gnn_type_checker/mcp.py`)
Provides tools for type checking and resource estimation:
```python
# Tools
- type_check_gnn_file(file_path)
- type_check_gnn_directory(dir_path, recursive, report_file)
- estimate_resources_for_gnn_file(file_path)
- estimate_resources_for_gnn_directory(dir_path, recursive)
```

### 4.4 Export Module (`export/mcp.py`)
Provides tools for exporting GNN models and reports:
```python
# Tools
- export_gnn_model(gnn_file_path, export_format, output_file_path): Exports GNN models to various non-visual formats (JSON, XML, GEXF, GraphML, Pickle, text).
- export_gnn_type_check_report(type_check_data_json_path, export_format, output_file_path)
- export_gnn_resource_estimation_report(resource_data_json_path, export_format, output_file_path)
```

### 4.5 Setup Utilities Module (`setup/mcp.py`)
Exposes general file and directory utilities:
```python
# Tools
- ensure_directory_exists(directory_path)
- find_project_gnn_files(search_directory, recursive)
- get_standard_output_paths(base_output_directory)
```

### 4.6 Tests Module (`tests/mcp.py`)
Exposes GNN testing functionalities (distinct from the type checker's direct tools):
```python
# Tools
- run_gnn_type_checker(file_path) # Runs type checker via test framework
- run_gnn_type_checker_on_directory(dir_path, report_file) # Runs via test framework
- run_gnn_unit_tests()

# Resources
- test-report://{report_file}
```

### 4.7 Visualization Module (`visualization/mcp.py`)
Provides tools for GNN model visualization:
```python
# Tools
- visualize_gnn_file(file_path, output_dir)
- visualize_gnn_directory(dir_path, output_dir)
- parse_gnn_file(file_path)

# Resources
- visualization://{output_directory}
```

## 5. Extensibility

### 5.1 Adding New MCP Modules
To add a new MCP module to GNN:

1. Create an `mcp.py` file in the module directory
2. Implement tool functions and resource retrievers
3. Create a `register_tools(mcp)` function:
   ```python
   def register_tools(mcp):
       # Register tools
       mcp.register_tool(
           "tool_name",
           tool_function,
           {
               "param1": {"type": "string", "description": "Parameter 1"},
               "param2": {"type": "integer", "description": "Parameter 2"}
           },
           "Tool description"
       )
       
       # Register resources
       mcp.register_resource(
           "resource://{identifier}",
           resource_retriever,
           "Resource description"
       )
   ```

### 5.2 Handling Complex Data Types
For complex data types or domain-specific objects:

1. Use JSON serialization for inputs and outputs
2. Provide schema definitions that can be understood by LLMs
3. Include clear documentation in tool and parameter descriptions

## 6. GNN-Specific Considerations

### 6.1 GNN Model Representation
When working with GNN models through MCP:

1. Models are primarily represented as Markdown files
2. Visualizations generate multiple output formats (PNG, SVG, HTML)
3. Type checking provides structured validation results
4. Resources can be used to access generated artifacts

### 6.2 Integration with LLMs
The MCP implementation enables LLMs to:

1. Parse and validate GNN models
2. Generate visualizations to understand model structure
3. Run type checking to ensure model validity
4. Access rich metadata about models
5. Programmatically interact with the GNN ecosystem

## 7. Testing and Quality Assurance (`test_mcp.py`)

The `test_mcp.py` script tests the core MCP framework, CLI, server instantiation, and integration with discovered modules (including `meta_mcp.py` if it can be discovered like other modules).

## 8. Usage Examples

### 8.1 Starting the MCP Server

```bash
# Start stdio server
python -m src.mcp.cli server --transport stdio

# Start HTTP server
python -m src.mcp.cli server --transport http --host 127.0.0.1 --port 8080
```

### 8.2 Using the CLI for Testing

```bash
# List capabilities
python -m src.mcp.cli list

# Execute a tool
python -m src.mcp.cli execute visualize_gnn_file --params '{"file_path": "path/to/model.md"}'

# Get a resource
python -m src.mcp.cli resource "visualization:///path/to/output/dir"
```

### 8.3 CLI Calls to Meta-Tools

```bash
python -m src.mcp.cli execute get_mcp_server_status
```

## 9. Security Considerations

### 9.1 Filesystem Access
MCP tools that access the filesystem should:

1. Validate paths to prevent directory traversal
2. Restrict operations to the project directory
3. Handle errors gracefully and provide clear error messages

### 9.2 Remote Access
When using the HTTP server:

1. By default, only localhost connections are accepted
2. No authentication is implemented (intended for local use only)
3. Production deployments should add proper authentication and TLS
