# GNN Model Context Protocol (MCP) Implementation Specification

## 1. Introduction

This document provides a detailed technical specification of the Model Context Protocol (MCP) server implementation within the GeneralizedNotationNotation (GNN) project. It is intended for developers working on or extending the GNN MCP server, and for those who require a deep understanding of its internal mechanisms, such as how tools are registered, discovered, and executed, and how different transport layers (HTTP, stdio) operate.

This specification complements the higher-level overview found in `src/mcp/README.md`, which describes the purpose and benefits of MCP integration in GNN. It also assumes familiarity with the general Model Context Protocol standard, as described in `src/mcp/model_context_protocol.md` (or `doc/mcp/gnn_mcp_model_context_protocol.md`).

The GNN MCP server acts as a bridge, exposing the rich functionalities of the GNN toolkit (parsing, type-checking, rendering, export, etc.) as standardized, callable "tools" that MCP-compliant clients (e.g., AI assistants, IDEs) can consume.

## 2. Core MCP Server Architecture (`src/mcp/mcp.py`)

The primary logic for the GNN MCP server resides in `src/mcp/mcp.py`. This file defines the central `MCP` class, which orchestrates tool discovery, registration, and execution, along with helper classes and an initialization routine.

### 2.1. `MCP` Class Detailed Description

The `MCP` class is the heart of the server. It maintains registries for available tools and resources and handles the dispatch of incoming requests to the appropriate handlers.

#### 2.1.1. Initialization (`__init__`)

When an `MCP` instance is created:
```python
class MCP:
    def __init__(self):
        self.tool_registry: Dict[str, MCPTool] = {}
        self.resource_registry: Dict[str, MCPResource] = {}
        self.discovered_modules: Dict[str, types.ModuleType] = {}
        self.logger = logging.getLogger(__name__)
        # Potentially other initializations like loading base configuration
```
-   `tool_registry`: A dictionary to store registered `MCPTool` objects, keyed by their unique tool name (e.g., `"export.export_gnn_to_json_mcp"`).
-   `resource_registry`: A dictionary to store registered `MCPResource` objects, keyed by their URI template.
-   `discovered_modules`: A dictionary to keep track of Python modules that have been successfully discovered and from which tools have been registered. This helps in debugging and understanding the server's current capabilities.
-   `logger`: A standard Python logger instance for logging server activities and errors.

#### 2.1.2. Module Discovery (`discover_modules`)

This method is responsible for finding and loading MCP tool definitions from various GNN sub-modules.
```python
    def discover_modules(self) -> bool:
        # ... (logic to find src_dir and project_root) ...
        possible_module_paths = []
        # Iterates through subdirectories of src/
        # Looks for 'mcp.py' in each subdirectory
        # Also specifically looks for 'meta_mcp.py' in src/mcp/

        for module_file_path in possible_module_paths:
            module_name = # ... (derived from path, e.g., src.export.mcp) ...
            try:
                spec = importlib.util.spec_from_file_location(module_name, module_file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "register_tools") and callable(module.register_tools):
                    module.register_tools(self) # Pass the MCP instance for registration
                    self.discovered_modules[module_name] = module
                    self.logger.info(f"Successfully registered tools from {module_name}")
                else:
                    self.logger.warning(f"Module {module_name} found but has no register_tools function.")
            except Exception as e:
                self.logger.error(f"Failed to load or register tools from {module_file_path}: {e}")
        return bool(self.tool_registry) # Returns true if at least one tool was registered
```
-   **Scanning**: It systematically scans subdirectories within the GNN project's `src/` directory (and potentially other configured locations).
-   **Identifying MCP Files**: It looks for files named `mcp.py` (or `meta_mcp.py` specifically for the `src/mcp` directory itself).
-   **Dynamic Importing**: Each found `mcp.py` file is dynamically imported as a Python module.
-   **Invoking `register_tools`**: If the imported module has a callable function named `register_tools`, this function is executed, passing the current `MCP` instance (`self`) as an argument. This allows the module to call `self.register_tool()` and `self.register_resource()` to make its capabilities known to the central server.
-   **Error Handling**: Includes mechanisms to log errors if a module fails to load or if its `register_tools` function is problematic, ensuring that the failure of one module doesn't necessarily stop the entire server initialization.

#### 2.1.3. Tool Registration (`register_tool`)

Modules call this method (via the `MCP` instance passed to their `register_tools` function) to make a specific function available as an MCP tool.
```python
    def register_tool(self, name: str, func: Callable, schema: Dict, description: str):
        if name in self.tool_registry:
            self.logger.warning(f"Tool {name} already registered. Overwriting.")
        self.tool_registry[name] = MCPTool(name=name, func=func, schema=schema, description=description)
        self.logger.info(f"Registered tool: {name}")
```
-   **Storage**: An `MCPTool` object (see Section 2.2) is created, encapsulating the tool's name, the actual Python function to execute, its parameter schema (JSON schema format), and a human-readable description.
-   **Registry**: This `MCPTool` object is stored in the `tool_registry` dictionary, keyed by the tool's unique `name` (e.g., `"export.export_gnn_to_json_mcp"`).
-   **Schema**: The `schema` parameter is crucial. It's a dictionary representing a JSON Schema that defines the expected parameters for the tool, their types, and whether they are required or optional. This schema is used for validating incoming requests and is also provided to clients during capability discovery.

#### 2.1.4. Resource Registration (`register_resource`)

Similar to tool registration, this method allows modules to expose retrievable resources.
```python
    def register_resource(self, uri_template: str, retriever: Callable, description: str):
        if uri_template in self.resource_registry:
            self.logger.warning(f"Resource URI template {uri_template} already registered. Overwriting.")
        self.resource_registry[uri_template] = MCPResource(uri_template=uri_template, retriever=retriever, description=description)
        self.logger.info(f"Registered resource: {uri_template}")
```
-   **Storage**: An `MCPResource` object is created, containing the `uri_template` (which might include placeholders like `{resource_id}`), the `retriever` function that fetches the resource content, and a `description`.
-   **Registry**: This `MCPResource` object is stored in the `resource_registry`.

#### 2.1.5. Request Handling and Dispatching

These methods are called by the transport layers (HTTP/stdio servers) to process client requests.

##### 2.1.5.1. `execute_tool` Method

This method handles invocations of registered tools.
```python
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Attempting to execute tool: {tool_name} with params: {params}")
        if tool_name not in self.tool_registry:
            self.logger.error(f"Tool not found: {tool_name}")
            raise MCPToolNotFoundError(f"Tool '{tool_name}' not found.")
        
        tool = self.tool_registry[tool_name]
        
        # Basic schema validation (can be enhanced with a proper JSON schema validator)
        # For simplicity, this example just checks for required parameters.
        # A real implementation should use a library like jsonschema.
        if tool.schema and tool.schema.get('properties'):
            for param_name, param_props in tool.schema['properties'].items():
                if tool.schema.get('required') and param_name in tool.schema['required'] and param_name not in params:
                    self.logger.error(f"Missing required parameter for {tool_name}: {param_name}")
                    raise ValueError(f"Missing required parameter: {param_name}")
        
        try:
            result = tool.func(**params) # Execute the tool's function
            self.logger.info(f"Tool {tool_name} executed successfully.")
            return result
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            # Depending on the error, re-raise or return a structured error
            raise # Or wrap in a specific MCP execution error
```
-   **Lookup**: It first checks if the requested `tool_name` exists in the `tool_registry`.
-   **Parameter Validation**: (Ideally) It validates the provided `params` against the tool's registered JSON schema. Missing required parameters or type mismatches should result in an error.
-   **Execution**: If the tool is found and parameters are valid, the associated Python function (`tool.func`) is called with the provided parameters unpacked (`**params`).
-   **Response/Error**: Returns the result from the tool function or raises an appropriate exception if an error occurs during lookup, validation, or execution.

##### 2.1.5.2. `get_resource` Method

This method handles requests to retrieve registered resources.
```python
    def get_resource(self, uri: str) -> Dict[str, Any]:
        self.logger.info(f"Attempting to retrieve resource: {uri}")
        # This is a simplified matching. A more robust implementation would handle URI templates.
        for template, resource_obj in self.resource_registry.items():
            # Simplified check: exact match or basic template match if implemented
            # A proper implementation would use a URI template matching library or regex
            if uri == template: # Placeholder for actual template matching logic
                try:
                    content = resource_obj.retriever(uri) # Call the retriever function
                    self.logger.info(f"Resource {uri} retrieved successfully.")
                    return content
                except Exception as e:
                    self.logger.error(f"Error retrieving resource {uri}: {e}", exc_info=True)
                    raise # Or wrap in a specific MCP resource error
        
        self.logger.error(f"Resource not found: {uri}")
        raise MCPResourceNotFoundError(f"Resource '{uri}' not found.")
```
-   **URI Matching**: It attempts to match the requested `uri` against the `uri_template`s in the `resource_registry`. This might involve simple string matching or more complex URI template parsing.
-   **Retrieval**: If a match is found, the associated `retriever` function is called with the URI (and potentially extracted parameters from the URI).
-   **Response/Error**: Returns the content from the retriever or raises an exception.

##### 2.1.5.3. `get_capabilities` Method

This method compiles and returns a description of all registered tools and resources, allowing clients to discover what the server can do.
```python
    def get_capabilities(self) -> Dict[str, Any]:
        self.logger.info("Retrieving server capabilities.")
        tools_list = []
        for tool_name, tool_obj in self.tool_registry.items():
            tools_list.append({
                "name": tool_obj.name,
                "description": tool_obj.description,
                "schema": tool_obj.schema
            })
        
        resources_list = []
        for uri_template, resource_obj in self.resource_registry.items():
            resources_list.append({
                "uri_template": resource_obj.uri_template,
                "description": resource_obj.description
            })
            
        return {
            "mcp_version": "1.0-gnn", # Example version
            "server_description": "GNN Project MCP Server",
            "tools": tools_list,
            "resources": resources_list
        }
```
-   **Compilation**: It iterates through `tool_registry` and `resource_registry`.
-   **Formatting**: For each tool, it includes its name, description, and parameter schema. For each resource, it includes its URI template and description.
-   **Output**: Returns a dictionary structured according to MCP guidelines for capability discovery, typically including server information, a list of tools, and a list of resources.

### 2.2. `MCPTool` and `MCPResource` Helper Classes

These are simple data classes (or named tuples) used to structure information about tools and resources within the registries.

```python
# (Assuming these are defined near the MCP class or imported)
from typing import Callable, Dict, Any

@dataclasses.dataclass # Or a simple class
class MCPTool:
    name: str
    func: Callable[..., Any]  # The actual Python function
    schema: Dict[str, Any]    # JSON schema for parameters
    description: str

@dataclasses.dataclass # Or a simple class
class MCPResource:
    uri_template: str
    retriever: Callable[[str], Any] # Function to get the resource content
    description: str
```
-   `MCPTool`: Holds the `name`, executable `func`, parameter `schema`, and `description` for a tool.
-   `MCPResource`: Holds the `uri_template`, `retriever` function, and `description` for a resource.

### 2.3. Initialization Function (`initialize`)

This standalone function (often at the end of `src/mcp/mcp.py`) is typically the main entry point for setting up and getting an `MCP` instance ready for use by the server transport layers.

```python
# (Global variable to hold the singleton MCP instance)
_mcp_instance = None

def initialize(halt_on_missing_sdk: bool = True, force_proceed_flag: bool = False) -> Tuple[MCP, bool, bool]:
    global _mcp_instance
    if _mcp_instance is None:
        _mcp_instance = MCP()
        # Optionally, check for SDKs or critical dependencies here if halt_on_missing_sdk is True
        # sdk_found = check_some_sdk()
        # if not sdk_found and halt_on_missing_sdk and not force_proceed_flag:
        #     raise MCPSDKNotFoundError("Critical SDK not found.")
        
        _mcp_instance.discover_modules() # Crucial step
    
    # did_force_proceed could be based on checks above
    return _mcp_instance, True # (Assuming sdk_found is true for this example), False # (Assuming not forced)

def get_mcp_instance() -> MCP:
    # Ensures that initialize() is called if instance doesn't exist yet
    if _mcp_instance is None:
        initialize()
    return _mcp_instance
```
-   **Singleton Pattern (Optional but common)**: Often, this function ensures that only one `MCP` instance is created and used throughout the application (e.g., by storing it in a global variable `_mcp_instance`).
-   **MCP Instantiation**: Creates an instance of the `MCP` class.
-   **Dependency Checks (Optional)**: Might include checks for essential SDKs or dependencies, controlled by `halt_on_missing_sdk` and `force_proceed_flag`.
-   **Module Discovery**: Critically, it calls `_mcp_instance.discover_modules()` to populate the tool and resource registries.
-   **Return Value**: Returns the initialized `MCP` instance and potentially status flags regarding dependencies.
-   `get_mcp_instance()`: A helper function usually provided so other parts of the MCP system (like server handlers) can easily access the central `MCP` object.

This core architecture in `src/mcp/mcp.py` provides a flexible and extensible framework for exposing GNN's functionalities via the Model Context Protocol.

## 3. Transport Layers

The GNN MCP server can communicate with clients using different transport layers. Currently, Standard I/O (stdio) and HTTP are supported. These layers are responsible for receiving raw request data, passing it to the core `MCP` instance for processing, and sending the response back to the client.

### 3.1. Stdio Server (`src/mcp/server_stdio.py`)

The stdio server allows MCP communication over standard input and standard output streams. This is particularly useful for direct integration with command-line applications or local AI assistants (like Claude Desktop) that manage child processes.

#### 3.1.1. `StdioServer` Class

```python
import queue
import sys
import threading
import json
import logging
# from .mcp import get_mcp_instance # Assuming mcp.py is in the same directory or accessible

class StdioServer:
    def __init__(self):
        # self.mcp_instance = get_mcp_instance() # Get the initialized MCP core
        # Simplified for spec, actual instance obtained via get_mcp_instance() from mcp.py
        self.mcp_instance = None # Placeholder for the MCP core instance
        self.incoming_queue = queue.Queue()
        self.outgoing_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(__name__)
        self.reader_thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.processor_thread = threading.Thread(target=self._processor_thread, daemon=True)
        self.writer_thread = threading.Thread(target=self._writer_thread, daemon=True)

    def start(self, mcp_core_instance):
        self.mcp_instance = mcp_core_instance
        self.logger.info("Starting StdioServer...")
        self.reader_thread.start()
        self.processor_thread.start()
        self.writer_thread.start()
        self.logger.info("StdioServer started and listening on stdin/stdout.")
        # Keep main thread alive or join threads if this is a blocking call
        try:
            while not self.stop_event.is_set():
                # Or join threads if appropriate for the application structure
                self.reader_thread.join(timeout=0.1)
                self.processor_thread.join(timeout=0.1)
                self.writer_thread.join(timeout=0.1)
                if not (self.reader_thread.is_alive() and \
                        self.processor_thread.is_alive() and \
                        self.writer_thread.is_alive()):
                    break # Exit if any thread dies
        except KeyboardInterrupt:
            self.logger.info("StdioServer received KeyboardInterrupt. Shutting down.")
        finally:
            self.stop()

    def stop(self):
        self.logger.info("Stopping StdioServer...")
        self.stop_event.set()
        # Send sentinel values to unblock queues if threads are waiting
        self.incoming_queue.put(None) 
        self.outgoing_queue.put(None)
        if self.reader_thread.is_alive(): self.reader_thread.join(timeout=1)
        if self.processor_thread.is_alive(): self.processor_thread.join(timeout=1)
        if self.writer_thread.is_alive(): self.writer_thread.join(timeout=1)
        self.logger.info("StdioServer stopped.")

    def _reader_thread(self):
        self.logger.debug("Stdio reader thread started.")
        for line in sys.stdin:
            if self.stop_event.is_set(): break
            line = line.strip()
            if not line: continue
            self.logger.debug(f"Stdio IN: {line}")
            try:
                request = json.loads(line)
                self.incoming_queue.put(request)
            except json.JSONDecodeError:
                self.logger.error(f"Invalid JSON received: {line}")
                # Consider sending a JSON-RPC error response for parse error
                err_response = {
                    "jsonrpc": "2.0", 
                    "error": {"code": -32700, "message": "Parse error"}, 
                    "id": None
                }
                self.outgoing_queue.put(err_response)
            except Exception as e:
                self.logger.error(f"Error in reader thread: {e}")
        self.logger.debug("Stdio reader thread stopped.")

    def _processor_thread(self):
        self.logger.debug("Stdio processor thread started.")
        while not self.stop_event.is_set():
            try:
                request = self.incoming_queue.get(timeout=0.1) # Timeout to check stop_event
                if request is None: # Sentinel value to stop
                    break
                response = self._process_message(request)
                self.outgoing_queue.put(response)
            except queue.Empty:
                continue # Timeout, loop again
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                # Construct a generic error response if possible
                request_id = request.get("id") if isinstance(request, dict) else None
                err_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": f"Internal error: {e}"},
                    "id": request_id
                }
                self.outgoing_queue.put(err_response)
        self.logger.debug("Stdio processor thread stopped.")

    def _writer_thread(self):
        self.logger.debug("Stdio writer thread started.")
        while not self.stop_event.is_set():
            try:
                response = self.outgoing_queue.get(timeout=0.1)
                if response is None: # Sentinel value to stop
                    break
                json_response = json.dumps(response)
                self.logger.debug(f"Stdio OUT: {json_response}")
                print(json_response, flush=True) # Print to stdout, flush ensures it's sent
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in writer thread: {e}")
        self.logger.debug("Stdio writer thread stopped.")

    def _process_message(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request_id = request.get("id")
        try:
            # Validate basic JSON-RPC structure
            if not all(k in request for k in ["jsonrpc", "method"]):
                raise ValueError("Invalid JSON-RPC request object")
            if request["jsonrpc"] != "2.0":
                raise ValueError("Invalid jsonrpc version")

            method = request["method"]
            params = request.get("params", {})

            # Handle capabilities directly or delegate to MCP instance
            if method == "mcp/discover" or method == "get_mcp_server_capabilities": # Example name
                result = self.mcp_instance.get_capabilities()
            elif method.startswith("resource/") : # Example: resource/get
                 uri = params.get("uri")
                 if not uri: raise ValueError("Missing uri for resource request")
                 result = self.mcp_instance.get_resource(uri)
            else: # Assume it's a tool name
                result = self.mcp_instance.execute_tool(method, params)
            
            return {"jsonrpc": "2.0", "result": result, "id": request_id}
        
        except Exception as e:
            self.logger.error(f"Error processing method {request.get('method')}: {e}", exc_info=True)
            # Determine error code based on MCP spec (e.g., MethodNotFound, InvalidParams)
            # This is a simplified error handling
            error_code = -32600 # Invalid Request by default
            if isinstance(e, ValueError): error_code = -32602 # Invalid params
            # Add MCPToolNotFoundError, MCPResourceNotFoundError etc from mcp.py
            # if isinstance(e, MCPToolNotFoundError): error_code = -32601 # Method not found
            
            return {
                "jsonrpc": "2.0",
                "error": {"code": error_code, "message": str(e)},
                "id": request_id
            }
```
-   **`mcp_instance`**: Holds a reference to the singleton `MCP` core object, obtained via `get_mcp_instance()` from `src/mcp/mcp.py` (or passed during `start`).
-   **Threading Model**: Uses a multi-threaded approach:
    -   A **reader thread** (`_reader_thread`) continuously reads lines from `sys.stdin`. Each line is expected to be a JSON-RPC request. Valid JSON objects are parsed and put into an `incoming_queue`.
    -   A **processor thread** (`_processor_thread`) takes messages from the `incoming_queue`. It calls `_process_message` which, in turn, dispatches the request (e.g., tool execution, resource retrieval, capability discovery) to the `self.mcp_instance`.
    -   The result (or error) from the `MCP` instance is formatted as a JSON-RPC response and put into an `outgoing_queue`.
    -   A **writer thread** (`_writer_thread`) takes messages from the `outgoing_queue` and writes them as JSON strings to `sys.stdout`, followed by a newline character and a flush to ensure immediate sending.
-   **Queues**: `incoming_queue` and `outgoing_queue` (from Python's `queue` module) are used to decouple the I/O operations from message processing, allowing for concurrent handling.
-   **`_process_message`**: This internal method is responsible for interpreting the incoming JSON-RPC message. It validates the JSON-RPC structure, identifies the method (e.g., tool name, `mcp/discover`), extracts parameters, calls the appropriate method on `self.mcp_instance` (e.g., `execute_tool`, `get_capabilities`, `get_resource`), and then formats the reply as a standard JSON-RPC response or error object.
-   **Error Handling**: Catches exceptions during JSON parsing, message processing, and tool execution, formatting them as JSON-RPC error responses with appropriate error codes (e.g., -32700 for Parse error, -32600 for Invalid Request, -32601 for Method not found, -32602 for Invalid params, -32603 for Internal error).
-   **Shutdown**: A `stop_event` (threading.Event) is used to signal threads to terminate gracefully. Sentinel values (`None`) can be put into queues to help unblock waiting threads.

#### 3.1.2. `start_stdio_server()` Function

```python
# In server_stdio.py
# from .mcp import initialize # To get the MCP instance

def start_stdio_server():
    # mcp_instance, _, _ = initialize() # Initialize the MCP core and discover tools
    # server = StdioServer()
    # server.start(mcp_instance)
    # This function is typically called by the CLI (`src/mcp/cli.py`)
    pass # Actual implementation would create StdioServer and start it.
```
This function, usually called by the CLI, creates an instance of `StdioServer`, initializes the core `MCP` instance (which discovers all tools), and then calls the server's `start(mcp_core_instance)` method to begin listening for requests.

### 3.2. HTTP Server (`src/mcp/server_http.py`)

The HTTP server allows MCP communication over the HTTP protocol, making GNN tools accessible via web requests. This is suitable for clients that prefer or require HTTP, such as web applications or remote services.

#### 3.2.1. `MCPHTTPHandler` Class

This class extends `http.server.BaseHTTPRequestHandler` to handle incoming HTTP requests. Only POST requests to a specific endpoint (e.g., `/jsonrpc`) are typically processed for MCP tool calls.

```python
import http.server
import json
import logging
# from .mcp import get_mcp_instance, MCPToolNotFoundError, MCPResourceNotFoundError # For error types

class MCPHTTPHandler(http.server.BaseHTTPRequestHandler):
    # mcp_instance = get_mcp_instance() # Class variable, initialized once by main server startup
    # For spec, assume mcp_instance is set by the MCPHTTPServer
    mcp_instance = None 
    logger = logging.getLogger(__name__)

    def do_POST(self):
        if self.path == '/' or self.path == '/jsonrpc': # Typically a single endpoint for JSON-RPC
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_jsonrpc_error(None, -32600, "Invalid Request: No data received")
                return
            
            post_data = self.rfile.read(content_length)
            self.logger.debug(f"HTTP IN: {post_data.decode('utf-8')}")
            try:
                request_json = json.loads(post_data.decode('utf-8'))
                self._handle_jsonrpc(request_json)
            except json.JSONDecodeError:
                self._send_jsonrpc_error(None, -32700, "Parse error")
            except Exception as e:
                self.logger.error(f"Unexpected error in do_POST: {e}", exc_info=True)
                self._send_jsonrpc_error(None, -32603, f"Internal server error: {e}")
        else:
            self._send_error(404, "Not Found")

    def _handle_jsonrpc(self, request: Dict[str, Any]):
        request_id = request.get("id")
        try:
            if not all(k in request for k in ["jsonrpc", "method"]):
                raise ValueError("Invalid JSON-RPC request object")
            if request["jsonrpc"] != "2.0":
                raise ValueError("Invalid jsonrpc version")

            method = request["method"]
            params = request.get("params", {})

            if method == "mcp/discover" or method == "get_mcp_server_capabilities":
                result = self.mcp_instance.get_capabilities()
            # Add handling for resource URIs if they are also POSTed, or handle via GET
            else: # Assume tool name
                result = self.mcp_instance.execute_tool(method, params)
            
            self._send_jsonrpc_result(request_id, result)
        
        except ValueError as e: # Catches invalid params from MCP core or JSON-RPC validation
            self.logger.warning(f"Invalid parameters for method {request.get('method')}: {e}")
            self._send_jsonrpc_error(request_id, -32602, str(e))
        # except MCPToolNotFoundError as e:
        #     self.logger.warning(f"Method not found: {request.get('method')}")
        #     self._send_jsonrpc_error(request_id, -32601, str(e))
        # except MCPResourceNotFoundError as e:
        #     self.logger.warning(f"Resource not found: {params.get('uri') if params else 'N/A'}")
        #     self._send_jsonrpc_error(request_id, -32601, str(e)) # Or a different code for resources
        except Exception as e:
            self.logger.error(f"Error processing HTTP request for method {request.get('method')}: {e}", exc_info=True)
            self._send_jsonrpc_error(request_id, -32603, f"Internal error: {e}")

    def _send_jsonrpc_result(self, request_id: Optional[str], result: Any):
        response_payload = {"jsonrpc": "2.0", "result": result, "id": request_id}
        self._send_json_response(200, response_payload)

    def _send_jsonrpc_error(self, request_id: Optional[str], code: int, message: str, data: Optional[Any] = None):
        error_obj = {"code": code, "message": message}
        if data is not None:
            error_obj["data"] = data
        response_payload = {"jsonrpc": "2.0", "error": error_obj, "id": request_id}
        # Map JSON-RPC error codes to HTTP status codes if desired, or use a generic 200 for JSON-RPC spec compliance
        # For simplicity, often 200 is used for all valid JSON-RPC responses (even errors)
        # or 400 for client errors (-32600 to -32699), 500 for server errors (-32000 to -32099)
        http_status = 200 
        if -32700 <= code <= -32600: http_status = 400 # Client-side JSON-RPC errors
        if code == -32601: http_status = 404 # Method not found can be 404
        self._send_json_response(http_status, response_payload)

    def _send_json_response(self, status_code: int, data: Any):
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        response_body = json.dumps(data).encode('utf-8')
        self.logger.debug(f"HTTP OUT ({status_code}): {response_body.decode('utf-8')}")
        self.wfile.write(response_body)

    def _send_error(self, status_code: int, message: str):
        self.send_response(status_code)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(message.encode('utf-8'))
        self.logger.debug(f"HTTP Error OUT ({status_code}): {message}")

    def log_message(self, format, *args):
        # Suppress default http.server logging unless debug is on
        # Or integrate with self.logger
        if self.logger.isEnabledFor(logging.DEBUG):
             self.logger.debug(f"{self.address_string()} - {format % args}")
```
-   **`mcp_instance`**: A class variable that should be set to the singleton `MCP` core object when the server starts. This is typically handled by `MCPHTTPServer`.
-   **`do_POST`**: Handles HTTP POST requests. It reads the request body, expects a JSON-RPC payload, parses it, and passes it to `_handle_jsonrpc`. It usually only responds to a specific endpoint like `/` or `/jsonrpc`.
-   **`_handle_jsonrpc`**: This is the core dispatcher for HTTP JSON-RPC requests. It inspects the JSON-RPC `method` field:
    -   If it's a known capability discovery method (e.g., `mcp/discover`), it invokes `mcp_instance.get_capabilities()`.
    -   If it's identified as a tool execution, it calls `mcp_instance.execute_tool()`.
    -   Resource requests might also be routed here if they are sent as POST requests with a specific method, or handled by `do_GET` if applicable.
-   **Response Methods**: `_send_jsonrpc_result` and `_send_jsonrpc_error` are used to construct and send valid JSON-RPC responses over HTTP. This includes setting the `Content-Type: application/json` header and appropriate HTTP status codes (e.g., 200 OK, 400 Bad Request for parse/validation errors, 404 Not Found for unknown methods, 500 Internal Server Error for tool execution errors).
-   **Logging**: Customizes logging to integrate with the application's logger.

#### 3.2.2. `MCPHTTPServer` Class

This class sets up and manages the HTTP server.

```python
import threading
from http.server import HTTPServer

class MCPHTTPServer:
    def __init__(self, mcp_core_instance, host: str = "127.0.0.1", port: int = 8080):
        self.mcp_instance = mcp_core_instance
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)

    def start(self):
        self.logger.info(f"Starting HTTP MCP server on {self.host}:{self.port}...")
        # Pass the MCP instance to the handler if it's instance-based
        # Or set it as a class variable if MCPHTTPHandler accesses it that way
        MCPHTTPHandler.mcp_instance = self.mcp_instance 
        self.server = HTTPServer((self.host, self.port), MCPHTTPHandler)
        self.thread = threading.Thread(target=self._server_thread, daemon=True)
        self.thread.start()
        self.logger.info(f"HTTP MCP server started. Listening for requests.")

    def _server_thread(self):
        if self.server:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                self.logger.info("HTTP server thread received KeyboardInterrupt.")
            except Exception as e:
                self.logger.error(f"HTTP server thread error: {e}", exc_info=True)
            finally:
                if self.server: self.server.server_close()
                self.logger.info("HTTP server thread stopped.")

    def stop(self):
        self.logger.info("Stopping HTTP MCP server...")
        if self.server:
            self.server.shutdown() # Signal serve_forever to stop
            self.server.server_close() # Close the server socket
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        self.logger.info("HTTP MCP server stopped.")
```
-   **Initialization**: Takes the core `mcp_core_instance`, `host`, and `port` for the server to listen on.
-   **Server Instance**: Uses Python's built-in `http.server.HTTPServer` with `MCPHTTPHandler` as the request handler. It ensures the handler class has access to the core `MCP` instance.
-   **Threading**: The HTTP server runs in a separate daemon thread (`self.thread`) using `serve_forever()`. This allows the `start()` method to return and the main program to continue.
-   **`start()` and `stop()`**: Methods to initiate and gracefully terminate the HTTP server. `stop()` uses `server.shutdown()` which must be called from a different thread than `serve_forever()`.

#### 3.2.3. `start_http_server()` Function

```python
# In server_http.py
# from .mcp import initialize # To get the MCP instance

def start_http_server(host: str = "127.0.0.1", port: int = 8080):
    # mcp_instance, _, _ = initialize()
    # server = MCPHTTPServer(mcp_instance, host, port)
    # server.start()
    # return server # Returns the server instance for potential management by CLI
    pass # Actual implementation in CLI would call this.
```
This function, usually called by the CLI, initializes the core `MCP` instance, creates an `MCPHTTPServer` instance with it, and starts the server. It might return the server instance for further management if needed.

Both transport layers are designed to use the same core `MCP` instance (obtained via `get_mcp_instance()` or passed during initialization), ensuring that all registered tools and resources are consistently available regardless of the communication protocol used by the client.

## 4. Command-Line Interface (`src/mcp/cli.py`)

The GNN project provides a command-line interface (CLI) in `src/mcp/cli.py` for interacting with the MCP server and its tools. This CLI is built using Python's `argparse` module and allows users to start the MCP server, list available capabilities, and directly execute tools or retrieve resources. It serves as both a utility for developers/users and an example of an MCP client.

Key functions and their roles:

```python
import argparse
import json
import sys
import logging
# from .mcp import initialize, MCPToolNotFoundError, MCPResourceNotFoundError
# from .server_stdio import start_stdio_server
# from .server_http import start_http_server

# (Global mcp_instance, initialized by main() or relevant subcommand handlers)
# mcp_instance = None

def list_capabilities(args):
    # global mcp_instance
    # if mcp_instance is None: mcp_instance, _, _ = initialize()
    # capabilities = mcp_instance.get_capabilities()
    # print(json.dumps(capabilities, indent=2))
    pass # Placeholder for actual implementation

def execute_tool_cli(args):
    # global mcp_instance
    # if mcp_instance is None: mcp_instance, _, _ = initialize()
    # tool_name = args.tool_name
    # params = json.loads(args.params) if args.params else {}
    # try:
    #     result = mcp_instance.execute_tool(tool_name, params)
    #     print(json.dumps({"status": "success", "result": result}, indent=2))
    # except MCPToolNotFoundError:
    #     print(json.dumps({"status": "error", "message": f"Tool '{tool_name}' not found."}, indent=2), file=sys.stderr)
    #     sys.exit(1)
    # except Exception as e:
    #     print(json.dumps({"status": "error", "message": str(e)}, indent=2), file=sys.stderr)
    #     sys.exit(1)
    pass # Placeholder for actual implementation

def get_resource_cli(args):
    # global mcp_instance
    # if mcp_instance is None: mcp_instance, _, _ = initialize()
    # uri = args.uri
    # try:
    #     resource_content = mcp_instance.get_resource(uri)
    #     print(json.dumps({"status": "success", "resource": resource_content}, indent=2))
    # except MCPResourceNotFoundError:
    #     print(json.dumps({"status": "error", "message": f"Resource '{uri}' not found."}, indent=2), file=sys.stderr)
    #     sys.exit(1)
    # except Exception as e:
    #     print(json.dumps({"status": "error", "message": str(e)}, indent=2), file=sys.stderr)
    #     sys.exit(1)
    pass # Placeholder for actual implementation

def start_server_cli(args):
    # initialize() # Ensure MCP core is ready and modules discovered
    # if args.transport == "stdio":
    #     print("Starting MCP server on stdio...", file=sys.stderr)
    #     start_stdio_server()
    # elif args.transport == "http":
    #     print(f"Starting MCP server on HTTP http://{args.host}:{args.port}...", file=sys.stderr)
    #     start_http_server(host=args.host, port=args.port)
    #     # Keep alive logic might be needed here if start_http_server is non-blocking
    #     try:
    #         while True:
    #             time.sleep(1)
    #     except KeyboardInterrupt:
    #         print("Shutting down HTTP server...", file=sys.stderr)
    pass # Placeholder for actual implementation

def main():
    parser = argparse.ArgumentParser(description="GNN Model Context Protocol CLI")
    # Global options like --verbose
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # 'list' command
    list_parser = subparsers.add_parser("list", help="List available MCP tools and resources")
    list_parser.set_defaults(func=list_capabilities)

    # 'execute' command
    execute_parser = subparsers.add_parser("execute", help="Execute an MCP tool")
    execute_parser.add_argument("tool_name", help="Name of the tool to execute (e.g., module.tool_name)")
    execute_parser.add_argument("--params", help="JSON string of parameters for the tool", default="{}")
    execute_parser.set_defaults(func=execute_tool_cli)

    # 'resource' command
    resource_parser = subparsers.add_parser("resource", help="Get an MCP resource")
    resource_parser.add_argument("uri", help="URI of the resource to retrieve")
    resource_parser.set_defaults(func=get_resource_cli)

    # 'server' command
    server_parser = subparsers.add_parser("server", help="Start the MCP server")
    server_parser.add_argument("--transport", choices=["stdio", "http"], default="stdio", help="Server transport type")
    server_parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP server")
    server_parser.add_argument("--port", type=int, default=8080, help="Port for HTTP server")
    server_parser.set_defaults(func=start_server_cli)

    args = parser.parse_args()

    # Configure logging based on --verbose
    # logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, ...)

    # Call the appropriate function based on the subcommand
    # args.func(args)

if __name__ == "__main__":
    # main()
    pass # Placeholder for actual execution
```

### 4.1. Argument Parsing

-   The `main()` function sets up an `ArgumentParser` from the `argparse` module.
-   It defines global options (like `--verbose` for logging levels) and then subparsers for different commands (`list`, `execute`, `resource`, `server`).
-   Each subcommand has its own set of arguments (e.g., `tool_name` and `params` for `execute`).

### 4.2. Subcommands

#### 4.2.1. `list`
-   **Purpose**: To display all available tools and resources that the MCP server can provide.
-   **Handler**: `list_capabilities(args)`.
-   **Action**: This function calls `mcp_instance.get_capabilities()` on the initialized `MCP` object and prints the resulting JSON structure (containing lists of tools with their schemas and resources with their URI templates) to standard output, typically formatted with indentation for readability.

#### 4.2.2. `execute`
-   **Purpose**: To invoke a specific MCP tool by name.
-   **Handler**: `execute_tool_cli(args)`.
-   **Arguments**:
    -   `tool_name`: The unique name of the tool to be executed (e.g., `export.export_gnn_to_json_mcp`).
    -   `--params` (optional): A JSON string representing a dictionary of parameters to be passed to the tool. Defaults to an empty dictionary `{}` if not provided.
-   **Action**: Parses the `tool_name` and `params` (decoding the JSON string). It then calls `mcp_instance.execute_tool(tool_name, parsed_params)`. The result (success or error) is printed to standard output as a JSON object. Errors during tool execution or if the tool is not found are caught and reported as structured JSON error messages to `sys.stderr`, and the CLI exits with a non-zero status code.

#### 4.2.3. `resource`
-   **Purpose**: To retrieve the content of a specific MCP resource by its URI.
-   **Handler**: `get_resource_cli(args)`.
-   **Arguments**:
    -   `uri`: The URI of the resource to retrieve.
-   **Action**: Calls `mcp_instance.get_resource(uri)`. The retrieved resource content is printed to standard output as part of a JSON success object. If the resource is not found or an error occurs, a JSON error message is printed to `sys.stderr` and the CLI exits with a non-zero status.

#### 4.2.4. `server`
-   **Purpose**: To start the MCP server using a specified transport layer.
-   **Handler**: `start_server_cli(args)`.
-   **Arguments**:
    -   `--transport`: Specifies the transport layer, either `stdio` (default) or `http`.
    -   `--host` (optional): The hostname or IP address for the HTTP server to bind to (default: `127.0.0.1`).
    -   `--port` (optional): The port number for the HTTP server (default: `8080`).
-   **Action**: Based on the `--transport` argument:
    -   If `stdio`, it calls `start_stdio_server()` (from `src/mcp/server_stdio.py`).
    -   If `http`, it calls `start_http_server(host, port)` (from `src/mcp/server_http.py`).
    The function also ensures that the main `MCP` instance is initialized (triggering tool discovery) before starting the chosen server. For the HTTP server, it might include a loop to keep the main thread alive until interrupted (e.g., by `Ctrl+C`).

### 4.3. Initialization of the `MCP` Instance

-   The `main()` function in `cli.py` is responsible for parsing command-line arguments.
-   Before any command handler (`list_capabilities`, `execute_tool_cli`, etc.) that requires access to MCP functionalities is called, the `initialize()` function from `src/mcp/mcp.py` must be invoked. This creates the `MCP` instance and triggers the `discover_modules()` process, populating the tool and resource registries.
-   This initialization might happen once at the beginning of `main()` or within each subcommand handler before it accesses the `mcp_instance`.
-   Logging levels (e.g., DEBUG, INFO) are also typically configured in `main()` based on global arguments like `--verbose`.

The CLI thus provides a complete interface for managing and interacting with the GNN MCP server from the command line, serving as a crucial tool for development, testing, and direct invocation of MCP functionalities.

## 5. Tool Module Integration (`src/<module_name>/mcp.py`)

The GNN MCP server achieves its broad range of capabilities by discovering and integrating tools from various functional sub-modules within the `src/` directory (e.g., `src/export/`, `src/render/`, `src/gnn_type_checker/`). This integration is standardized through a convention: each module that wishes to expose MCP tools must contain an `mcp.py` file with a specific structure.

### 5.1. Standard Structure of an `mcp.py` File in a Functional Module

A typical `mcp.py` file within a GNN functional module (e.g., `src/export/mcp.py`) will generally include:

1.  **Imports**: Import necessary functions or classes from its own module (e.g., the core exporting logic from `src/export/export_json.py`) and any types required for schemas (e.g., `Dict`, `Any`, `List` from `typing`).
2.  **Wrapper Functions (Optional but Common)**: These are functions specifically designed to be MCP tools. They often act as an adapter layer between the raw MCP request (parameters as a dictionary) and the underlying core GNN Python functions. Their responsibilities might include:
    *   Extracting and validating parameters from the `params` dictionary provided by the MCP call.
    *   Calling the core GNN functions with these parameters.
    *   Formatting the results from the core functions into the dictionary structure expected by the MCP client (as defined in the tool's output schema, though not explicitly part of the MCP registration schema itself, it's good practice).
    *   Handling exceptions from the core functions and translating them into meaningful error responses or raising specific exceptions that the main MCP server can catch and format.
3.  **`register_tools(mcp_instance)` Function**: This is the mandatory function that the core `MCP` server (`src/mcp/mcp.py`) will call during its `discover_modules` phase.

**Example Snippet (Conceptual, e.g., from `src/export/mcp.py`):**
```python
# In src/export/mcp.py
from typing import Dict, Any, Callable
# Assuming core_export_logic is in src/export/core_exporter.py or similar
# from .core_exporter import export_model_to_json_format 

# Placeholder for the actual export function
def _actual_export_to_json(gnn_data: Dict[str, Any], output_path: str) -> None:
    # ... core logic to perform the export ...
    with open(output_path, 'w') as f:
        json.dump(gnn_data, f, indent=2)
    pass

# Wrapper function to be exposed as an MCP tool
def export_gnn_to_json_mcp(gnn_file_path: str, output_file_path: str) -> Dict[str, Any]:
    try:
        # In a real scenario, you'd parse gnn_file_path to get gnn_data first
        # For this example, let's assume gnn_file_path IS the data for simplicity of spec
        # Or, more realistically, the tool would call a GNN parsing function.
        parsed_gnn_data = {"message": f"Data from {gnn_file_path}"} # Placeholder for parsed GNN data
        
        _actual_export_to_json(parsed_gnn_data, output_file_path)
        return {
            "status": "success",
            "message": f"GNN data from {gnn_file_path} exported to {output_file_path} in JSON format.",
            "output_uri": f"file://{output_file_path}"
        }
    except FileNotFoundError as e:
        # Specific error handling can be done here or in the core MCP dispatcher
        raise ValueError(f"Input GNN file not found: {gnn_file_path}") from e
    except Exception as e:
        # Log the full error internally
        # Raise a new error or re-raise with a message suitable for the client
        raise RuntimeError(f"Failed to export GNN to JSON: {e}") from e

def register_tools(mcp_instance):
    # mcp_instance is the MCP object from src/mcp/mcp.py
    mcp_instance.register_tool(
        name="export.export_gnn_to_json_mcp",
        func=export_gnn_to_json_mcp,
        schema={
            "type": "object",
            "properties": {
                "gnn_file_path": {"type": "string", "description": "Path to the input GNN specification file."},
                "output_file_path": {"type": "string", "description": "Path where the JSON output file will be saved."}
            },
            "required": ["gnn_file_path", "output_file_path"]
        },
        description="Exports a GNN model specified by a file path to a JSON file."
    )
    
    # ... register other export tools (XML, GraphML, etc.) ...
```

### 5.2. The `register_tools(mcp_instance)` Function

This function is the designated entry point for each module to announce its MCP capabilities to the main server.

-   **Argument**: It receives one argument, `mcp_instance`, which is the instance of the main `MCP` class from `src/mcp/mcp.py`.
-   **Action**: Inside this function, the module developer calls `mcp_instance.register_tool()` for each Python function they want to expose as an MCP tool, and `mcp_instance.register_resource()` for any data resources.
-   **Tool Naming Convention**: Tool names should be unique across the entire MCP server. A common convention is to prefix the tool name with the module's name followed by a dot, e.g., `export.export_gnn_to_json_mcp`, `render.render_gnn_to_pymdp_mcp`. This helps in organization and avoids naming collisions.
-   **Function Reference**: The `func` parameter in `register_tool` should be a direct reference to the Python callable (often a wrapper function) that implements the tool's logic.
-   **Schema Definition**: A crucial part of tool registration is providing an accurate JSON Schema for the tool's parameters (`schema` argument). This schema dictates:
    -   The expected parameter names.
    -   Their data types (e.g., `string`, `integer`, `boolean`, `object`, `array`).
    -   Descriptions for each parameter, aiding client understanding.
    -   Which parameters are `required` and which are optional.
    -   Constraints like `enum` for string parameters with fixed allowed values (e.g., `target_format: Literal["pymdp", "rxinfer"]` would translate to an `enum` in JSON Schema).
    The main `MCP` server might use this schema to perform basic validation before even calling the tool function, and clients use it to construct valid requests.
-   **Description**: A human-readable `description` of what the tool does, its inputs, and its outputs. This is used by clients (e.g., via `mcp/discover`) to understand the tool's purpose.

### 5.3. Wrapper Functions

While not strictly mandatory (a core GNN function could be registered directly if its signature matches MCP needs), wrapper functions are highly recommended for MCP tools for several reasons:

-   **Decoupling**: They separate the MCP interface logic from the core GNN business logic. The core logic can remain unaware of MCP.
-   **Parameter Handling**: They can gracefully handle the dictionary of parameters (`params`) passed by the MCP system, extract values by name, provide defaults for optional parameters, and perform initial type checks or transformations before calling the core function.
-   **Return Value Formatting**: Core GNN functions might return complex Python objects. The wrapper can convert these into a dictionary or a simple JSON-serializable structure suitable for an MCP response.
-   **Error Handling**: Wrappers can catch exceptions from the core logic and convert them into standardized error responses or raise specific exceptions that the `MCP.execute_tool` method can better interpret (e.g., distinguishing between a `FileNotFoundError` and a `GNNParserError`).
-   **Side Effects Management**: If a tool has side effects (like writing a file), the wrapper can manage these and report success/failure and output locations in a structured way (e.g., returning a URI to the created file).

By adhering to this structure, new GNN functionalities can be easily exposed via MCP by simply adding or modifying the respective `mcp.py` file within the relevant module, without needing to alter the central MCP server code in `src/mcp/mcp.py`.

## 6. Meta Tools (`src/mcp/meta_mcp.py`)

The GNN MCP server includes a special module, `src/mcp/meta_mcp.py`, which provides tools that offer information *about* the MCP server itself. These "meta tools" are essential for clients to understand the server's status, capabilities, and operational parameters. This module is discovered and its tools registered in the same way as other functional modules, by having a `register_tools(mcp_instance)` function.

**Purpose of Meta Tools:**

-   **Server Introspection**: Allow clients to query the server about its own state and configuration.
-   **Capability Discovery**: While the core `MCP.get_capabilities()` method provides a primary way to list tools, meta-tools can offer more granular or alternative ways to query capabilities, or provide extended server information beyond the basic tool list.
-   **Health Checks**: Provide endpoints that clients can use to check if the server is running and responsive.
-   **Debugging and Diagnostics**: Offer tools that might return internal server statistics or diagnostic information helpful for developers.

**Example Meta Tools (Conceptual - actual implementation details may vary):**

```python
# In src/mcp/meta_mcp.py
from typing import Dict, Any
# from .mcp import MCP # To access the mcp_instance fields, if not directly passed

# (These functions would typically access properties of the mcp_instance)

def get_mcp_server_status_mcp(mcp_instance_ref) -> Dict[str, Any]:
    # In a real scenario, mcp_instance_ref might be the global MCP instance
    # or this function is registered in a way that it can access it.
    # For this spec, assume it gets the mcp_instance somehow.
    return {
        "status": "running",
        "uptime_seconds": 12345, # Example: calculate actual uptime
        "discovered_modules_count": len(mcp_instance_ref.discovered_modules),
        "registered_tools_count": len(mcp_instance_ref.tool_registry),
        "registered_resources_count": len(mcp_instance_ref.resource_registry),
        "server_version": "GNN-MCP-1.0.0", # Example version
        # Potentially add info about active transport (stdio/http) if available
    }

def get_mcp_full_capabilities_mcp(mcp_instance_ref) -> Dict[str, Any]:
    # This essentially calls the main get_capabilities method on the MCP instance
    return mcp_instance_ref.get_capabilities()

def get_mcp_module_info_mcp(mcp_instance_ref, module_name: str) -> Dict[str, Any]:
    if module_name in mcp_instance_ref.discovered_modules:
        # Potentially list tools registered by this specific module
        module_tools = [
            tool.name for tool_name, tool in mcp_instance_ref.tool_registry.items() 
            if tool_name.startswith(module_name.split('.')[-2] + '.') # Crude way to link tool to module by name prefix
        ]
        return {
            "module_name": module_name,
            "status": "loaded",
            "path": str(mcp_instance_ref.discovered_modules[module_name].__file__),
            "provided_tools_sample": module_tools[:5] # Sample of tools
        }
    else:
        raise ValueError(f"Module '{module_name}' not found or not loaded.")


def register_tools(mcp_instance):
    # The mcp_instance is the main MCP object from src/mcp/mcp.py

    mcp_instance.register_tool(
        name="meta.get_server_status",
        # Pass the mcp_instance itself, or make meta tools methods of MCP class
        func=lambda: get_mcp_server_status_mcp(mcp_instance), 
        schema={},
        description="Returns the current operational status and basic statistics of the MCP server."
    )

    mcp_instance.register_tool(
        name="meta.get_full_capabilities",
        func=lambda: get_mcp_full_capabilities_mcp(mcp_instance),
        schema={},
        description="Returns a comprehensive list of all registered tools and resources with their schemas (equivalent to primary mcp/discover)."
    )

    mcp_instance.register_tool(
        name="meta.get_module_info",
        func=lambda module_name: get_mcp_module_info_mcp(mcp_instance, module_name=module_name),
        schema={
            "type": "object",
            "properties": {
                "module_name": {"type": "string", "description": "The fully qualified name of the discovered module (e.g., src.export.mcp)."}
            },
            "required": ["module_name"]
        },
        description="Provides information about a specific discovered MCP module."
    )
    
    # Potentially add other meta-tools:
    # - meta.ping: A simple tool that returns a pong, for health checks.
    # - meta.get_server_logs (with care, might expose sensitive info):
    #   To retrieve recent server log entries for debugging.
```

**Key Considerations for Meta Tools:**

-   **Access to `MCP` Instance**: The functions implementing meta-tools often need access to the state of the main `MCP` instance (e.g., its `tool_registry`, `discovered_modules`). This can be achieved by:
    -   Passing the `mcp_instance` as an argument when registering the tool (e.g., using a lambda `func=lambda: my_meta_tool_func(mcp_instance, **params)`).
    -   Designing meta-tool functions to retrieve the global singleton `MCP` instance if that pattern is used.
-   **Schema Definition**: Even if meta-tools take no parameters (like `meta.get_server_status`), they should still be registered with an empty schema (e.g., `schema={}`). If they take parameters (like `meta.get_module_info` needing a `module_name`), a proper schema must be provided.
-   **Naming**: Meta tools should follow a clear naming convention, typically prefixed with `meta.` (e.g., `meta.get_server_status`).
-   **Security/Sensitivity**: Care must be taken not to expose overly sensitive internal server details or configurations through meta-tools unless intended and secured appropriately.

The `meta_mcp.py` module ensures that the MCP server is not a black box and provides standardized ways for clients to learn about and interact with its operational aspects.

## 7. Error Handling and Response Structure

Robust error handling and consistent response structures are critical for a usable MCP server. The GNN MCP server adheres to JSON-RPC 2.0 for its response formats, including errors.

### 7.1. JSON-RPC 2.0 Response Structure

**Successful Response:**
A successful JSON-RPC response object will always contain the following members:
-   `jsonrpc`: A String specifying the version of the JSON-RPC protocol, which MUST be "2.0".
-   `result`: This member is REQUIRED on success. Its value is determined by the method invoked on the server. If there are no results to return, this member MAY be omitted (or be `null`).
-   `id`: This member is REQUIRED. It MUST be the same as the value of the `id` member in the Request Object. If there was an error in detecting the `id` in the Request Object (e.g., Parse error/Invalid Request), it MUST be `null`.

Example successful response (from `src/mcp/README.md` tool execution example):
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

**Error Response:**
When a JSON-RPC call encounters an error, the response object MUST contain the `error` member with the following structure:
-   `jsonrpc`: A String specifying the version of the JSON-RPC protocol, which MUST be "2.0".
-   `error`: This member is REQUIRED on error. This member MUST be an Object with the following members:
    -   `code`: A Number that indicates the error type that occurred. JSON-RPC pre-defines specific codes (see Section 7.2).
    -   `message`: A String providing a short description of the error.
    -   `data` (optional): A Primitive or Structured value that contains additional information about the error. May be omitted.
-   `id`: This member is REQUIRED. It MUST be the same as the value of the `id` member in the Request Object. If there was an error in detecting the `id` in the Request Object (e.g., Parse error/Invalid Request), it MUST be `null`.

Example error response:
```json
{
    "jsonrpc": "2.0",
    "error": {
        "code": -32601,
        "message": "Method not found",
        "data": "Tool 'non_existent_tool' not found."
    },
    "id": "request-456"
}
```

### 7.2. Standard JSON-RPC 2.0 Error Codes

The GNN MCP server aims to use standard JSON-RPC 2.0 error codes where applicable:

-   **`-32700 Parse error`**: Invalid JSON was received by the server. An error occurred on the server while parsing the JSON text.
-   **`-32600 Invalid Request`**: The JSON sent is not a valid Request object.
-   **`-32601 Method not found`**: The method does not exist / is not available.
-   **`-32602 Invalid params`**: Invalid method parameter(s). This is typically raised if the parameters provided do not match the tool's schema (e.g., missing required parameters, incorrect types).
-   **`-32603 Internal error`**: Internal JSON-RPC error. This indicates an issue on the server-side that was not caught by more specific error handlers (e.g., an unhandled exception within a tool's execution).
-   **`-32000 to -32099 Server error`**: Reserved for implementation-defined server-errors. The GNN MCP server might use this range for specific GNN-related errors that don't fit the standard categories.

### 7.3. Custom Error Types in GNN MCP

Within `src/mcp/mcp.py` (or a dedicated errors module), custom Python exception classes may be defined to represent specific error conditions. These are then caught by the `MCP.execute_tool` or `MCP.get_resource` methods (or by the transport layer handlers) and translated into appropriate JSON-RPC error objects.

Example custom exceptions (conceptual):
```python
# In mcp.py or a new mcp_exceptions.py
class MCPError(Exception):
    """Base class for MCP related errors."""
    def __init__(self, message, code=-32000, data=None):
        super().__init__(message)
        self.code = code
        self.data = data

class MCPToolNotFoundError(MCPError):
    def __init__(self, tool_name):
        super().__init__(f"Tool '{tool_name}' not found.", code=-32601, data=f"Tool '{tool_name}' not found.")

class MCPResourceNotFoundError(MCPError):
    def __init__(self, uri):
        super().__init__(f"Resource '{uri}' not found.", code=-32601, data=f"Resource '{uri}' not found.") # Or a custom code

class MCPInvalidParamsError(MCPError):
    def __init__(self, message, details=None):
        super().__init__(message, code=-32602, data=details)

class MCPToolExecutionError(MCPError):
    def __init__(self, tool_name, original_exception):
        super().__init__(f"Error executing tool '{tool_name}': {original_exception}", code=-32000, data=str(original_exception))

class MCPSDKNotFoundError(MCPError):
    def __init__(self, message="MCP SDK not found or failed to initialize."):
        super().__init__(message, code=-32001) # Example custom server error code
```

### 7.4. Error Handling in `MCP` Core Methods

-   **`MCP.execute_tool(tool_name, params)`**:
    -   If `tool_name` is not in `tool_registry`, it should raise `MCPToolNotFoundError`.
    -   If `params` do not conform to the tool's schema (as validated by `MCP.execute_tool` or by the tool wrapper itself), it should raise `MCPInvalidParamsError` (or let `ValueError` propagate to be caught and converted).
    -   If the tool function itself raises an unhandled exception, `MCP.execute_tool` should catch this and raise an `MCPToolExecutionError` (or a generic internal error), logging the original traceback for server-side debugging.
-   **`MCP.get_resource(uri)`**:
    -   If no resource matches the `uri`, it should raise `MCPResourceNotFoundError`.
    -   If the resource's retriever function fails, it should raise an appropriate error (e.g., `MCPToolExecutionError` if retriever is considered like a tool, or a generic internal error).
-   **`MCP.discover_modules()`**: Logs errors for modules that fail to load or register tools but generally does not halt the entire server unless a critical module is missing.

### 7.5. Error Handling in Transport Layers (`server_stdio.py`, `server_http.py`)

-   **Parsing Errors**: If the incoming request (e.g., a line from stdin or an HTTP POST body) is not valid JSON, the transport layer is responsible for generating a JSON-RPC error response with code `-32700 (Parse error)` and `id: null`.
-   **Invalid Request Structure**: If the JSON is valid but doesn't conform to the JSON-RPC request structure (e.g., missing `jsonrpc` or `method`), the transport layer should generate an error response with code `-32600 (Invalid Request)`.
-   **Catching Exceptions from `MCP` Core**: The transport layer handlers (e.g., `StdioServer._process_message`, `MCPHTTPHandler._handle_jsonrpc`) will call methods like `mcp_instance.execute_tool`. They must have `try...except` blocks to catch exceptions raised by these core methods (like `MCPToolNotFoundError`, `ValueError` for invalid params, general `Exception` for internal errors) and convert them into the appropriate JSON-RPC error object format to send back to the client.
    -   For instance, an `MCPToolNotFoundError` from `mcp_instance.execute_tool` would be caught and translated into a JSON-RPC error object with `code: -32601`.
    -   A `ValueError` due to missing parameters might be translated to `code: -32602`.
    -   Any other unexpected exceptions caught from the core logic execution should be translated to `code: -32603 (Internal error)` or a custom server error code, and the original error message might be included in the `data` field or logged server-side for diagnostics.

### 7.6. Logging

Comprehensive logging is essential for debugging errors.
-   All significant operations (tool execution attempts, successes, failures, module discovery) are logged by the `MCP` class and transport layers.
-   Errors, especially unhandled exceptions within tools or the MCP framework, are logged with tracebacks to help identify the root cause.
-   The CLI (and server start-up scripts) should allow configuring log levels (e.g., DEBUG, INFO, WARNING, ERROR) to control verbosity.

By consistently applying these error handling strategies and response structures, the GNN MCP server provides a predictable and developer-friendly interface.

## 8. Security Considerations

<!-- Content for Security Considerations -->

## 9. Testing (`src/mcp/test_mcp.py`)

<!-- Content for Testing -->
