# Model Context Protocol (MCP) for GeneralizedNotationNotation

This module implements the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for the GeneralizedNotationNotation project. MCP is an open standard that enables seamless integration between Large Language Model (LLM) applications and external tools, developed by Anthropic.

## Overview

The MCP module provides a standardized way for LLM applications to interact with the GNN toolkit, allowing AI assistants to:

1. Run type checks on GNN models
2. Execute tests on the codebase
3. Generate visualizations of GNN models
4. Export GNN models and reports to various formats
5. Access resources and metadata about the project
6. Query the MCP server itself for its status and capabilities

## Architecture

The MCP implementation consists of:

- Core MCP functionality (`mcp.py`) that discovers and integrates with other modules
- Module-specific MCP implementations in each subdirectory (e.g., `tests/mcp.py`, `visualization/mcp.py`, `export/mcp.py`, `gnn_type_checker/mcp.py`, `setup/mcp.py`, `gnn/mcp.py`)
- A meta-module (`meta_mcp.py`) for server self-reflection
- Transport implementations for stdio and HTTP protocols
- CLI tools for starting MCP servers and executing MCP commands

## Available Tools

The MCP module provides the following tools, categorized by the module that provides them:

### MCP Server (`meta_mcp.py`)
- `get_mcp_server_capabilities`: Retrieves the full capabilities description of this MCP server.
- `get_mcp_server_status`: Provides the current operational status of the MCP server (uptime, loaded modules).
- `get_mcp_server_auth_status`: Describes the current authentication mechanisms and status.
- `get_mcp_server_encryption_status`: Describes the current encryption status for server transport.

### GNN Documentation (`gnn/mcp.py`)
- `get_gnn_documentation`: Retrieve the content of a GNN core documentation file (e.g., syntax, file structure).

### GNN Type Checker (`gnn_type_checker/mcp.py`)
- `type_check_gnn_file`: Runs the GNN type checker on a specified GNN model file.
- `type_check_gnn_directory`: Runs the GNN type checker on all GNN files in a specified directory.
- `estimate_resources_for_gnn_file`: Estimates computational resources for a GNN model file.
- `estimate_resources_for_gnn_directory`: Estimates computational resources for all GNN files in a directory.

### Export (`export/mcp.py`)
- `export_gnn_model`: Exports a GNN model to various non-visual file formats (e.g., JSON, XML, GEXF, GraphML, Pickle, plain text).
- `export_gnn_type_check_report`: Exports GNN Type Check results/report to a specified format.
- `export_gnn_resource_estimation_report`: Exports GNN Resource Estimation results/report to a specified format.

### Setup Utilities (`setup/mcp.py`)
- `ensure_directory_exists`: Ensures a directory exists, creating it if necessary.
- `find_project_gnn_files`: Finds all GNN (.md) files in a specified directory.
- `get_standard_output_paths`: Gets a dictionary of standard output directory paths.

### Tests (`tests/mcp.py`)
- `run_gnn_type_checker`: Run the GNN type checker on a specific file (via test module).
- `run_gnn_type_checker_on_directory`: Run the GNN type checker on all GNN files in a directory (via test module).
- `run_gnn_unit_tests`: Run the GNN unit tests and return results.

### Visualization (`visualization/mcp.py`)
- `visualize_gnn_file`: Generate visualizations for a specific GNN file.
- `visualize_gnn_directory`: Generate visualizations for all GNN files in a directory.
- `parse_gnn_file`: Parse a GNN file without generating visualizations.

## Resources

The MCP module also provides access to resources:

- `gnn://documentation/{doc_name}`: Access GNN core documentation files.
- `visualization://{output_directory}`: Retrieve visualization results by output directory.
- `test-report://{report_file}`: Retrieve a test report by file path.

## Usage

### Starting the MCP Server

To start an MCP server using stdio transport (for Claude Desktop or other LLM clients):

```bash
python -m src.mcp.cli server --transport stdio
```

To start an MCP server using HTTP transport:

```bash
python -m src.mcp.cli server --transport http --host 127.0.0.1 --port 8080
```

### Using the CLI to Execute Tools

List available MCP capabilities:

```bash
python -m src.mcp.cli list
# or use the meta-tool:
python -m src.mcp.cli execute get_mcp_server_capabilities
```

Execute a tool:

```bash
python -m src.mcp.cli execute visualize_gnn_file --params '{"file_path": "path/to/model.md"}'
```

Get a resource:

```bash
python -m src.mcp.cli resource "visualization:///path/to/output/dir"
```

## Integration with LLM Applications

### Claude Desktop

To use this MCP server with Claude Desktop, add the following to your Claude Desktop configuration file:

**On MacOS:**
```json
// ~/Library/Application\ Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "gnn-tools": {
      "command": "python3",
      "args": ["-m", "src.mcp.cli", "server", "--transport", "stdio"]
    }
  }
}
```

**On Windows:**
```json
// %APPDATA%\Claude\claude_desktop_config.json
{
  "mcpServers": {
    "gnn-tools": {
      "command": "python",
      "args": ["-m", "src.mcp.cli", "server", "--transport", "stdio"]
    }
  }
}
```

**On Linux:**
```json
// ~/.config/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "gnn-tools": {
      "command": "python3",
      "args": ["-m", "src.mcp.cli", "server", "--transport", "stdio"]
    }
  }
}
```

### Other LLM Applications

For other LLM applications that support MCP, follow their documentation for adding external MCP servers. You can typically point them to either:

1. The stdio server executable: `python3 -m src.mcp.cli server --transport stdio`
2. The HTTP server URL: `http://127.0.0.1:8080` (after starting the HTTP server)

## Development

### Adding New MCP Capabilities

To add new MCP capabilities to a module:

1. Create an `mcp.py` file in your module directory.
2. Implement tools and resources as functions.
3. Add a `register_tools(mcp_instance)` function that registers your tools and resources with the MCP instance.

Example:

```python
def my_tool(param1, param2):
    # Implement your tool functionality
    return {"result": "success"}

def my_resource_retriever(uri):
    # Implement your resource retrieval
    return {"content": "resource data"}

def register_tools(mcp_instance):
    mcp_instance.register_tool(
        "my_module_tool",
        my_tool,
        {
            "param1": {"type": "string", "description": "Parameter 1"},
            "param2": {"type": "integer", "description": "Parameter 2"}
        },
        "Description of my tool"
    )
    
    mcp_instance.register_resource(
        "my-resource://{resource_id}",
        my_resource_retriever,
        "Description of my resource"
    )
```

## Dependencies

The MCP module requires:

- Python 3.7+
- Standard library modules: json, sys, os, logging, threading, etc.
- Module-specific dependencies (imported from other GNN modules)

## License

This MCP implementation is part of the GeneralizedNotationNotation project and is released under the same license as the main project. 