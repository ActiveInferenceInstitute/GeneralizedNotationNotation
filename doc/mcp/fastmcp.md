# FastMCP: Simplified Model Context Protocol Development

FastMCP is a high-level Python framework designed to simplify the development of Model Context Protocol (MCP) servers and clients. It enables standardized, secure, and efficient interactions between Large Language Models (LLMs) and external systems, tools, and resources. FastMCP 2.0 is the actively developed successor to FastMCP 1.0 (which was contributed to the official MCP Python SDK) and significantly expands on its predecessor with powerful client capabilities, server proxying, composition, OpenAPI/FastAPI integration, and more advanced features [2, 1].

This document provides a comprehensive technical overview of FastMCP's specifications, architecture, and usage patterns.

---

## What is MCP?

The Model Context Protocol (MCP) is a standard that allows you to build servers exposing data and functionality to LLM applications in a secure, standardized manner. It's often described as "the USB-C port for AI," providing a uniform way to connect LLMs to resources they can use. Think of it as an API specifically designed for LLM interactions. MCP servers can [2]:

* **Expose data** through `Resources` (akin to GET endpoints, used to load information into an LLM's context).
* **Provide functionality** through `Tools` (akin to POST endpoints, used to execute code or produce side effects).
* **Define interaction patterns** through `Prompts` (reusable templates for LLM interactions).

While a low-level Python SDK exists for direct protocol implementation, FastMCP offers a high-level, Pythonic interface to simplify this process [2].

---

## Why FastMCP?

FastMCP handles the complex protocol details and server management, allowing developers to focus on building valuable tools and resources. It aims to be [2]:

* 🚀 **Fast**: A high-level interface means less code and faster development cycles.
* 🍀 **Simple**: Build MCP servers with minimal boilerplate code; often, decorating a function is sufficient.
* 🐍 **Pythonic**: Designed to feel natural and intuitive for Python developers.
* 🔍 **Complete**: Aims to provide a full implementation of the core MCP specification.

FastMCP 2.0 is the recommended path for building modern, powerful MCP applications [2].

---

## Core Architecture

**MCP Protocol Implementation**: FastMCP robustly implements the Model Context Protocol, facilitating secure and structured LLM tooling interactions. It effectively creates specialized web APIs tailored for AI agents [1, 2].

**Server Types**:

* **Direct Servers**: Host tools and resources directly using the `FastMCP` class.
* **Proxy Servers**: Act as intermediaries for other MCP servers (local or remote) using `FastMCP.from_client()`. This is useful for bridging transports or adding logic to existing servers [1].
* **Composite Servers**: Build modular applications by mounting multiple `FastMCP` instances onto a parent server using `mcp.mount()` (live link) or `mcp.import_server()` (static copy) [1].

---

## Key Components

### 1. Tools

Tools are functions exposed by the MCP server that an LLM can call to perform actions.
* Decorate Python functions with `@mcp.tool()` to expose them as MCP tools.
* Supports both synchronous and asynchronous functions.
* Automatic schema generation from type hints and docstrings.
* Context injection is available via the `ctx: Context` parameter [1, 2].

```python
from fastmcp import FastMCP

mcp = FastMCP("Demo Server 🚀")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
async def calculate_entropy(data: list[float]) -> float:
    """Computes Shannon entropy of numerical data."""
    # ... implementation ...
    return sum(data) # Placeholder
```

### 2. Resources

Resources are data endpoints that an LLM can query to retrieve information.
* Define resources using the `@mcp.resource("schema://uri")` decorator.
* Supports static and templated URIs for dynamic resource access [1, 2].

```python
from fastmcp import FastMCP # Assuming mcp is an instance of FastMCP
# from db_setup import db, User # Placeholder for database setup

mcp = FastMCP("Resource Demo Server")

@mcp.resource("users://{id}/profile")
def get_user_profile(id: str) -> dict:
    """Retrieves a user's profile by their ID."""
    # return db.query(User).filter(User.id == id).first().__dict__ # Example
    return {"id": id, "name": "Jane Doe", "email": "jane@example.com"} # Placeholder
```

### 3. Prompts

Prompts are reusable message templates that can guide LLM interactions.
* Create prompts using the `@mcp.prompt()` decorator.
* Prompts can be dynamically generated using function parameters [1, 2].

```python
from fastmcp import FastMCP # Assuming mcp is an instance of FastMCP
import pandas as pd # Assuming pandas is used

mcp = FastMCP("Prompt Demo Server")

@mcp.prompt()
def generate_analysis_prompt(data: pd.DataFrame, focus_metric: str) -> str:
    """Generates a prompt to analyze trends in the provided DataFrame."""
    return f"Analyze trends in the following data, paying special attention to '{focus_metric}':\n{data.head().to_markdown()}"
```

### 4. Context System (`ctx`)

When a tool is called, FastMCP can inject a `Context` object (`ctx`) as a parameter. This object provides powerful capabilities for interacting with the client and server environment [1]:
* **Logging**: Send messages back to the client LLM (e.g., `await ctx.info("Processing started...")`).
* **LLM Sampling**: Request the client LLM to generate text (e.g., `summary = await ctx.sample("Summarize this text: ...")`).
* **Resource Access**: Read other resources exposed by the server (e.g., `data = await ctx.read_resource("users://all")`).
* **HTTP Requests**: Make external HTTP requests (e.g., `response = await ctx.http_request("GET", "https://api.example.com/data")`).

```python
from fastmcp import FastMCP, Context
import asyncio

mcp = FastMCP("Context Demo Server")

@mcp.tool()
async def process_data_with_context(uri: str, ctx: Context):
    """
    Processes data from a given URI using the context object
    for logging, resource access, and LLM sampling.
    """
    await ctx.info(f"Processing data from URI: {uri}...")

    # Example: Read another resource from the server
    # data_resource = await ctx.read_resource(uri)
    # content_to_summarize = data_resource.content[:500] # Assuming text content

    # For demonstration, using a placeholder string
    content_to_summarize = "This is some sample data that needs to be summarized by the LLM."
    
    await ctx.info(f"Content to summarize: {content_to_summarize}")

    # Ask client LLM to summarize the data
    summary_response = await ctx.sample(f"Please summarize the following text: {content_to_summarize}")
    
    await ctx.info(f"LLM summary: {summary_response.text}")

    return {"summary": summary_response.text, "original_uri": uri}

```

---

## Running Your Server

The primary way to run a FastMCP server is by calling the `run()` method on your server instance. FastMCP supports several transport protocols [1]:

**1. STDIO (Default)**:
Best for local tools and command-line scripts.

```python
# server.py
from fastmcp import FastMCP

mcp_stdio = FastMCP("My Local Server STDIO")

@mcp_stdio.tool()
def greet_stdio(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    # This condition is for making the example runnable.
    # In a real setup, you'd likely pick one way to run.
    print("Running STDIO server example. Call greet_stdio tool.")
    mcp_stdio.run() # Defaults to transport="stdio"
    # or explicitly: mcp_stdio.run(transport="stdio")
```

**2. Streamable HTTP**:
Recommended for web deployments. This typically involves an ASGI server like Uvicorn.

```python
# server_http.py
from fastmcp import FastMCP

mcp_http = FastMCP("My HTTP Server")

@mcp_http.tool()
def greet_http(name: str) -> str:
    return f"Hello from HTTP server, {name}!"

# Expose the ASGI app for Uvicorn or other ASGI servers
app = mcp_http.build_asgi_app()

# To run: uvicorn server_http:app --host 127.0.0.1 --port 8000
# if __name__ == "__main__":
#     # Direct run is less common for HTTP but possible for testing
#     print("To run with Uvicorn: uvicorn server_http:app --host 127.0.0.1 --port 8000")
#     print("Attempting direct run (for simple tests only)...")
#     try:
#        mcp_http.run(transport="streamable-http", host="127.0.0.1", port=8000, path="/mcp")
#     except Exception as e:
#        print(f"Direct run failed (likely due to asyncio loop): {e}")
#        print("Please use Uvicorn for HTTP.")
```

To serve via HTTP, you would typically define `app = mcp.build_asgi_app()` in your Python file (e.g., `server_http.py`) and run it with an ASGI server:
`uvicorn server_http:app --host 127.0.0.1 --port 8000`

**3. Server-Sent Events (SSE)**:
For compatibility with existing SSE clients.

```python
# server_sse.py
from fastmcp import FastMCP

mcp_sse = FastMCP("My SSE Server")

@mcp_sse.tool()
def greet_sse(name: str) -> str:
    return f"Hello from SSE server, {name}!"

if __name__ == "__main__":
    print("Running SSE server example on http://127.0.0.1:8000")
    mcp_sse.run(transport="sse", host="127.0.0.1", port=8000)
```

**Command-Line Interface (CLI)**:
FastMCP also provides a CLI for running servers (ensure your server file defines an `mcp` instance):

```bash
# Local execution using STDIO (assuming server.py has 'mcp = FastMCP(...)')
fastmcp run server.py --transport stdio

# Production-like deployment using Streamable HTTP
fastmcp run server.py --transport streamable-http --host 0.0.0.0 --port 8080 --path /mcp_api
```

---

## Client Integration

FastMCP includes a versatile `Client` for interacting with any MCP server programmatically. It supports various transports (Stdio, SSE, In-Memory) and often auto-detects the correct one [1].

### Basic Usage

```python
from fastmcp import Client, FastMCP
import asyncio

# A simple server for the client to connect to (for stdio example)
# This would typically be in a separate my_server.py file
# For this example, we define it here for self-contained execution.
client_test_mcp = FastMCP("ClientTestServer")
@client_test_mcp.tool()
def add_for_client(a: int, b: int) -> int:
    return a + b

async def interact_with_server():
    # Example 1: Connecting via stdio to a conceptual local script
    # For a real stdio client, 'my_server.py' would be a script that runs mcp.run()
    # Here, we'll simulate it by passing the mcp instance directly for an in-memory example later
    print("\n--- STDIO Client Example (Simulated In-Memory) ---")
    # This shows the pattern; for true stdio, use: Client("python my_server.py")
    async with Client(client_test_mcp) as local_client: # Simulating with in-memory
        tools_response = await local_client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools_response.tools]}")
        if "add_for_client" in [tool.name for tool in tools_response.tools]:
            result = await local_client.call_tool("add_for_client", {"a": 5, "b": 3})
            print(f"Result of add_for_client(5, 3): {result.text}")
        else:
            print("'add_for_client' tool not found.")

    # Example 2: Connecting via HTTP (ensure an HTTP server like server_http.py is running)
    # print("\n--- HTTP Client Example ---")
    # try:
    #     async with Client("http://localhost:8000/mcp") as http_client: # Adjust URL as needed
    #         http_tools = await http_client.list_tools()
    #         print(f"Available tools on HTTP server: {[tool.name for tool in http_tools.tools]}")
    #         # result = await http_client.call_tool("greet_http", {"name": "HTTP Client"})
    #         # print(f"Result from HTTP server: {result.text}")
    # except Exception as e:
    #     print(f"Could not connect to HTTP server at http://localhost:8000/mcp: {e}")
    #     print("Ensure the HTTP FastMCP server is running.")

if __name__ == "__main__":
    # To run the client example:
    # 1. Ensure the `client_test_mcp` server part can be accessed (it's in-memory here).
    # 2. For HTTP, run `server_http.py` (or similar) first, then uncomment that section.
    asyncio.run(interact_with_server())
```

### In-Memory Testing

The client allows for efficient in-memory testing of your servers by connecting directly to a `FastMCP` server instance via the `FastMCPTransport`. This eliminates the need for process management or network calls during tests [1].

```python
from fastmcp import FastMCP, Client
import asyncio
import pytest # For test structure

# Define a simple server for testing
test_mcp_server_instance = FastMCP("TestServerForClientInMemory")
@test_mcp_server_instance.tool()
def echo_tool(message: str) -> str:
    return f"Server echoes: {message}"

@pytest.mark.asyncio
async def test_in_memory_client_interaction():
    async with Client(test_mcp_server_instance) as client:  # In-memory transport
        response = await client.call_tool("echo_tool", {"message": "Hello, in-memory test!"})
        assert response.text == "Server echoes: Hello, in-memory test!"
        print("In-memory test passed successfully via pytest.")

# To run this test, you would typically use pytest:
# $ pytest your_test_file.py
#
# If you want to run it directly for demonstration (without pytest):
async def main_run_test():
    await test_in_memory_client_interaction()

if __name__ == "__main__":
    # This will run the test if the file is executed directly.
    # For proper testing, use pytest.
    # print("Running in-memory test directly (for demo):")
    # asyncio.run(main_run_test())
    pass # Pytest will discover and run the test
```

---

## Advanced Features

FastMCP introduces powerful ways to structure and deploy your MCP applications [1].

| Feature                        | Implementation                                    | Use Case                                     |
|--------------------------------|---------------------------------------------------|----------------------------------------------|
| **Proxy Servers**              | `FastMCP.from_client(existing_client)`            | Bridging transports, augmenting other servers|
| **Composing MCP Servers**      | `parent_mcp.mount(child_mcp, path_prefix="/child")` | Modular application architecture             |
|                                | `parent_mcp.import_server(child_mcp)`             | Static composition of tools/resources        |
| **OpenAPI Integration**        | `FastMCP.from_openapi("spec.yaml_or_url")`        | Expose existing OpenAPI specs as MCP tools   |
| **FastAPI Integration**        | `FastMCP.from_fastapi(fastapi_app)`               | Expose FastAPI app routes as MCP tools       |
| **Flexible Transports**        | STDIO, Streamable HTTP, SSE                       | Versatile deployment options                 |
| **Lifespan Management**        | `lifespan` argument in `FastMCP` constructor      | Manage resources like DB connections         |
| **Tagging & Metadata**         | `tags`, `instructions` in `FastMCP` constructor   | Richer server/tool description for LLMs    |

### Proxy Servers

Create a FastMCP server that acts as an intermediary for another local or remote MCP server using `FastMCP.from_client()`. This is useful for bridging transports (e.g., remote SSE to local Stdio) or adding a layer of logic to a server you don't control [1].

### Composing MCP Servers

Build modular applications by mounting multiple `FastMCP` instances onto a parent server using `mcp.mount()` (for live, prefix-based mounting) or `mcp.import_server()` (for a static copy of tools/resources) [1].

### OpenAPI & FastAPI Generation

Automatically generate FastMCP servers from existing OpenAPI specifications (`FastMCP.from_openapi()`) or FastAPI applications (`FastMCP.from_fastapi()`), instantly bringing your web APIs to the MCP ecosystem [1].

---

## Development Ecosystem

### Installation

```bash
# Recommended (using uv)
uv pip install fastmcp

# Alternative (using pip)
uv pip install fastmcp
```

Ensure you have Python 3.10+ [1].

### Contributing

Contributions are highly welcome! FastMCP uses `pre-commit` for code formatting, linting, and type-checking. The project has a comprehensive unit test suite [1].

**Prerequisites**:

* Python 3.10+
* `uv` (Recommended for environment management)

**Setup**:

1. Clone the repository: `git clone https://github.com/jlowin/fastmcp.git && cd fastmcp`
2. Create and sync the environment: `uv sync` (installs dependencies, including dev tools)
3. Activate the virtual environment (e.g., `source .venv/bin/activate`).

**Unit Tests**:
Run tests using `pytest`:

```bash
pytest
# For coverage report:
uv run pytest --cov=src --cov=examples --cov-report=html
```

All Pull Requests must introduce or update tests as appropriate and pass the full suite.

**Static Checks**:
Install pre-commit hooks locally:

```bash
uv run pre-commit install
```

Hooks will run automatically on `git commit`. Run them manually:

```bash
pre-commit run --all-files
# or via uv
uv run pre-commit run --all-files
```

All PRs must pass these checks.

**Pull Requests**:

1. Fork the repository on GitHub.
2. Create a feature branch from `main`.
3. Make changes, including tests and documentation updates.
4. Ensure tests and pre-commit hooks pass.
5. Commit and push to your fork.
6. Open a PR against the `main` branch of `jlowin/fastmcp`.

---

## Performance Characteristics

* **Protocol Overhead**: FastMCP aims for minimal overhead, benchmarked favorably against raw MCP implementations [1]. (Specific figures would need current validation if critical).
* **Concurrency**: Built with an async-first architecture, supporting concurrent requests efficiently.
* **Payload Handling**: Provides automatic conversion between common data formats like JSON.

---

## LLM-Friendly Documentation

The official FastMCP documentation is also available in `llms.txt` format, a simple markdown standard designed for easy consumption by LLMs [2].

* `llms.txt`: A sitemap listing all pages in the documentation.
* `llms-full.txt`: Contains the entire documentation (note: may exceed some LLM context windows).

---

This specification demonstrates FastMCP's position as a comprehensive framework for building production-grade MCP solutions, combining protocol compliance with developer ergonomics. Its active development continues to expand integration capabilities [1, 2].

## Citations

[1] FastMCP GitHub Repository. URL: <https://github.com/jlowin/fastmcp>
[2] Welcome to FastMCP 2.0! - FastMCP Documentation. URL: <https://gofastmcp.com/getting-started/welcome>
[3] FastMCP Quickstart - FastMCP Documentation. URL: <https://gofastmcp.com/getting-started/quickstart> (Note: Content may overlap significantly with [2])
