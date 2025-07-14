# Model Context Protocol (MCP) Integration in GNN

This document provides a comprehensive overview of how the Model Context Protocol (MCP) is integrated and utilized within the Generalized Notation Notation (GNN) project. MCP serves as a standardized communication layer, enabling AI models, IDEs, and other software systems to interact with the GNN toolkit's diverse functionalities as callable "tools."

## 1. Purpose of MCP in the GNN Project

The primary goal of integrating MCP into the GNN project is to expose its rich set of functionalities—such as parsing, type checking, resource estimation, exporting, and rendering GNN files—in a standardized, programmatic way. This allows various MCP "hosts" (e.g., AI assistants, integrated development environments, automated research pipelines) to:

*   Programmatically access and utilize GNN's capabilities.
*   Automate workflows involving GNN model definition, analysis, and transformation.
*   Integrate GNN functionalities into larger AI systems and research environments.
*   Enhance interoperability with other MCP-compatible tools and platforms.

## 2. Core MCP Concepts Leveraged

The GNN project primarily leverages the following MCP concepts, as detailed in the general MCP specification (`model_context_protocol.md` or `doc/mcp/gnn_mcp_model_context_protocol.md`):

*   **MCP Server**: The GNN project implements an MCP server that listens for requests from MCP clients.
*   **MCP Tools**: GNN's core functionalities (e.g., `export_gnn_to_json`, `render_gnn_to_pymdp`, `type_check_gnn_file`) are exposed as "tools" that MCP clients can invoke. Each tool has a defined schema specifying its inputs and outputs.
*   **MCP Resources**: GNN files themselves, adhering to the GNN DSL specification (`doc/gnn_dsl_manual.md`), are the primary "resources" upon which these tools operate. Tool invocations typically reference these GNN files by their path.
*   **JSON-RPC 2.0**: This is the underlying protocol used for communication between MCP clients and the GNN MCP server.

## 3. Overview of GNN's MCP Implementation

The MCP implementation within the GNN project is primarily located in the `src/mcp/` directory, with extensions in various functional subdirectories.

*   **Central MCP Server Logic (`src/mcp/`)**:
    *   `mcp.py`: Defines the core `MCP` class, responsible for discovering and registering tools from different GNN modules, and dispatching incoming requests to the appropriate tool handlers.
    *   `server_http.py`: Implements an HTTP-based MCP server, allowing clients to communicate with GNN tools over HTTP.
    *   `server_stdio.py`: Implements an MCP server that communicates over standard input/output, suitable for direct integration with command-line tools or tightly coupled processes.
    *   `cli.py`: Provides a command-line interface for interacting with the GNN MCP server, allowing users to list available tools, execute them, or start the server.
    *   `meta_mcp.py`: Exposes MCP tools related to the MCP server itself (e.g., status, capabilities).

*   **Exposing GNN Functionalities as MCP Tools**:
    Various GNN functionalities are wrapped and exposed as MCP tools through dedicated `mcp.py` files in their respective modules:
    *   `src/export/mcp.py`: Registers tools for exporting GNN files to different formats (JSON, XML, GraphML, etc.).
    *   `src/render/mcp.py`: Registers tools for rendering GNN models into executable formats (e.g., PyMDP, RxInfer).
    *   `src/type_checker/mcp.py`: Registers tools for validating GNN file syntax and structure, and for estimating computational resources.
    *   `src/visualization/mcp.py` (if present, or logic integrated elsewhere): Would register tools for generating visualizations from GNN files.
    *   `src/ontology/mcp.py`: Registers tools for processing and validating GNN ontology annotations.

    Each of these modules contains a `register_tools(mcp_instance)` function, which is called by the main `MCP` class in `src/mcp/mcp.py` during initialization to make all GNN tools discoverable and callable.

## 4. Benefits of MCP Integration for GNN

Integrating MCP provides several key advantages:

*   **Standardization & Interoperability**: Adheres to an open standard, allowing any MCP-compatible client to interact with GNN.
*   **Programmatic Access & Automation**: Enables scripts, AI agents, and other programs to use GNN functionalities as part of automated workflows.
*   **Modularity & Extensibility**: GNN tools are organized modularly. New functionalities can be added and exposed as new MCP tools without altering the core MCP server logic.
*   **Support for Advanced AI Systems**: Facilitates the development of sophisticated AI agents (e.g., LLM-based agents as described in `doc/gnn_llm_neurosymbolic_active_inference.md`) that can reason about, manipulate, and utilize GNN models by calling upon GNN tools.
*   **Decoupling**: The GNN core logic is decoupled from the way it's accessed, allowing for different communication protocols (HTTP, stdio) and potential future protocols.

## 5. Typical Workflow Example

A common interaction between an MCP Host (e.g., an AI assistant or an IDE plugin) and the GNN MCP Server might proceed as follows:

1.  **Initialization**: The GNN MCP Server is started (e.g., via `python -m src.mcp.cli start_server http`). It discovers and registers all available GNN tools.
2.  **Tool Discovery (Optional)**: The MCP Host can send a `mcp/discover` request (or a similar capabilities request) to the GNN MCP Server to get a list of available tools and their JSON schemas. This allows the host to understand how to call each tool.
3.  **Tool Invocation**:
    *   The MCP Host wishes to validate a GNN file named `example.gnn`.
    *   It constructs a JSON-RPC request:
        ```json
        {
            "jsonrpc": "2.0",
            "method": "type_checker.type_check_gnn_file",
            "params": {"file_path": "path/to/example.gnn"},
            "id": "request-123"
        }
        ```
    *   This request is sent to the GNN MCP Server (e.g., via HTTP POST).
4.  **Server-Side Processing**:
    *   The GNN MCP Server receives the request.
    *   The `MCP` class in `src/mcp/mcp.py` routes the request to the `type_check_gnn_file_mcp` tool handler, which is defined in `src/type_checker/mcp.py`.
    *   The tool handler executes the GNN type-checking logic on `example.gnn`.
5.  **Response**:
    *   The GNN MCP Server sends a JSON-RPC response back to the Host:
        ```json
        {
            "jsonrpc": "2.0",
            "result": {
                "file_path": "path/to/example.gnn",
                "is_valid": true,
                "errors": [],
                "warnings": [],
                "resource_estimates": {
                    "memory_estimate_kb": 10.5,
                    "inference_estimate_units": 150
                }
            },
            "id": "request-123"
        }
        ```
6.  **Host Processing**: The MCP Host receives the response and can then use the validation results and resource estimates.

## 6. Relationship to GNN Documentation

*   **GNN Language and Structure**: Documents like `doc/gnn_dsl_manual.md`, `doc/gnn_file_structure_doc.md`, and `doc/gnn_syntax.md` define the GNN "language" that the MCP tools operate on. The GNN files are the primary "resources."
*   **GNN Tool Functionality**: The specific actions performed by the MCP tools (e.g., export formats, rendering targets, type-checking rules, resource metrics) are implicitly described by the GNN project's overall documentation (`doc/about_gnn.md`, `doc/gnn_implementation.md`, `doc/resource_metrics.md`, etc.).
*   **Active Inference Context**: The broader purpose and application of GNN models, which these tools help manage, are detailed in documents like `doc/gnn_overview.md`, `doc/gnn_paper.md`, and `doc/gnn_llm_neurosymbolic_active_inference.md`.

## 7. Key Files and Directories in `src/mcp/`

*   `README.md` (this file): Overview of MCP integration.
*   `mcp.py`: Core MCP server class, tool registration, request dispatching.
*   `cli.py`: Command-line interface for MCP server and tools.
*   `server_http.py`: HTTP server implementation for MCP.
*   `server_stdio.py`: Standard I/O server implementation for MCP.
*   `meta_mcp.py`: MCP tools for MCP server introspection.
*   `model_context_protocol.md`: A copy or link to the general MCP specification document.
*   `test_mcp.py`: Unit and integration tests for the MCP components.

By leveraging MCP, the GNN project significantly enhances its usability, interoperability, and potential for integration into sophisticated AI-driven research and development workflows. 