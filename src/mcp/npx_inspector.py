#!/usr/bin/env python3
"""
GNN MCP Inspector (Python Implementation)

This script provides a command-line interface to inspect and interact with
the GNN Model Context Protocol (MCP) server. It can launch the server
and send requests to it.

Inspired by the concept of npx @modelcontextprotocol/inspector.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
import queue
from pathlib import Path
import shlex

# --- Configuration ---
# Adjust these paths if your project structure is different
GNN_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MCP_CLI_PATH = GNN_PROJECT_ROOT / "src" / "mcp" / "cli.py"
PYTHON_EXECUTABLE = sys.executable # Use the same python interpreter

# --- Helper Functions ---

def print_json(data):
    """Prints JSON data with indentation."""
    print(json.dumps(data, indent=2, sort_keys=True))

def read_server_output(process, output_queue, error_queue):
    """Reads stdout from the server process and puts lines into a queue."""
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            output_queue.put(line)
        process.stdout.close()

def read_server_errors(process, error_queue):
    """Reads stderr from the server process and puts lines into a queue."""
    if process.stderr:
        for line in iter(process.stderr.readline, ''):
            error_queue.put(line)
        process.stderr.close()

class StdioMCPClient:
    """A simple client to interact with an MCP server over stdio."""
    def __init__(self, process):
        self.process = process
        self.request_id_counter = 1
        self.response_timeout = 10 # seconds
        self.server_stdout_queue = queue.Queue()
        self.server_stderr_queue = queue.Queue()

        self.stdout_thread = threading.Thread(
            target=read_server_output, 
            args=(self.process, self.server_stdout_queue)
        )
        self.stderr_thread = threading.Thread(
            target=read_server_errors,
            args=(self.process, self.server_stderr_queue)
        )
        self.stdout_thread.daemon = True
        self.stderr_thread.daemon = True
        self.stdout_thread.start()
        self.stderr_thread.start()

    def _send_request(self, method: str, params: dict = None) -> dict:
        if not self.process.stdin:
            raise IOError("Server stdin is not available.")

        request_id = f"inspector-{self.request_id_counter}"
        self.request_id_counter += 1
        
        rpc_request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id
        }
        if params is not None:
            rpc_request["params"] = params

        request_str = json.dumps(rpc_request)
        print(f"INSPECTOR -> SERVER: {request_str}", file=sys.stderr)
        self.process.stdin.write(request_str + '\n')
        self.process.stdin.flush()

        # Wait for response
        start_time = time.time()
        while True:
            if time.time() - start_time > self.response_timeout:
                raise TimeoutError(f"Timeout waiting for response to request ID {request_id}")
            
            try:
                # Check stderr first for server-side issues unrelated to this request
                while not self.server_stderr_queue.empty():
                    err_line = self.server_stderr_queue.get_nowait().strip()
                    if err_line: # Only print if it's not an empty line
                        print(f"SERVER (stderr): {err_line}", file=sys.stderr)

                line = self.server_stdout_queue.get(timeout=0.1) # Check queue with timeout
                print(f"SERVER -> INSPECTOR: {line.strip()}", file=sys.stderr)
                response = json.loads(line)
                if response.get("id") == request_id:
                    return response
                else:
                    # Might be a notification or unrelated message, log it and continue
                    print(f"INSPECTOR (info): Received unrelated message or notification: {response}", file=sys.stderr)
            except queue.Empty:
                if self.process.poll() is not None: # Server process terminated
                    raise ConnectionError("Server process terminated unexpectedly.")
                continue # Timeout, try again
            except json.JSONDecodeError as e:
                print(f"INSPECTOR (error): Could not decode JSON from server: {line.strip()} - {e}", file=sys.stderr)
                # If it's a fatal error, we might not get a response with our ID.
                # This could be part of a multi-line error dump from the server.
                # For now, just log and continue waiting for our specific response ID.
            except Exception as e:
                print(f"INSPECTOR (error): Unexpected error reading server response: {e}", file=sys.stderr)
                raise # Re-raise for now

    def get_capabilities(self) -> dict:
        # MCP standard often uses "mcp/discover" or similar, 
        # but GNN server's meta_mcp.py registers "get_mcp_server_capabilities"
        return self._send_request(method="get_mcp_server_capabilities")

    def execute_tool(self, tool_name: str, tool_params: dict) -> dict:
        # GNN MCP server expects tool name as method and params as params object
        return self._send_request(method=tool_name, params=tool_params)

    def get_resource(self, uri: str) -> dict:
        # This might require a specific method name if not using raw URI as method
        # For now, assuming a hypothetical "resource/get" method, or that resource URIs are tools.
        # The GNN `cli.py` 'resource' command implies resource URIs are distinct.
        # Let's assume a tool like `meta.get_resource_content` for now, or adjust if GNN MCP has a specific one.
        # Based on GNN MCP spec, there isn't a generic "get resource" tool.
        # Resources are typically outputs of other tools. This function might be less useful directly.
        # For now, let's make it try to call the URI as if it were a tool (unlikely to work).
        print(f"INSPECTOR (warning): Direct resource GET not well-defined in GNN MCP. Trying URI as method.", file=sys.stderr)
        return self._send_request(method=uri) # This is a guess

# --- CLI Subcommands ---

def handle_list_capabilities(client: StdioMCPClient, args):
    """Handles the 'list-capabilities' command."""
    print("Inspector: Requesting server capabilities...", file=sys.stderr)
    try:
        response = client.get_capabilities()
        print_json(response)
    except Exception as e:
        print(f"Error getting capabilities: {e}", file=sys.stderr)
        if hasattr(e, '__cause__') and e.__cause__:
             print(f"Cause: {e.__cause__}", file=sys.stderr)

def handle_execute_tool(client: StdioMCPClient, args):
    """Handles the 'execute-tool' command."""
    tool_name = args.tool_name
    try:
        tool_params = json.loads(args.params) if args.params else {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in --params: {e}", file=sys.stderr)
        return

    print(f"Inspector: Executing tool '{tool_name}' with params: {tool_params}", file=sys.stderr)
    try:
        response = client.execute_tool(tool_name, tool_params)
        print_json(response)
    except Exception as e:
        print(f"Error executing tool '{tool_name}': {e}", file=sys.stderr)

def handle_get_resource(client: StdioMCPClient, args):
    """Handles the 'get-resource' command."""
    uri = args.uri
    print(f"Inspector: Attempting to get resource '{uri}'...", file=sys.stderr)
    try:
        response = client.get_resource(uri) # This might not work as expected with GNN MCP
        print_json(response)
    except Exception as e:
        print(f"Error getting resource '{uri}': {e}", file=sys.stderr)


# --- Main ---
def main():
    # This is the primary parser for the inspector tool itself.
    parser = argparse.ArgumentParser(
        description="GNN MCP Inspector. Launches and interacts with a GNN MCP server.",
        epilog=f"Example: python {sys.argv[0]} --server-cmd \"python src/mcp/cli.py server --transport stdio\" list-capabilities"
    )
    parser.add_argument(
        "--server-cmd",
        help="Full command string to start the GNN MCP server. "
             "Example: 'python src/mcp/cli.py server --transport stdio'. "
             "If not provided, defaults to stdio server via configured MCP_CLI_PATH.",
        default=f"{PYTHON_EXECUTABLE} {MCP_CLI_PATH} server --transport stdio"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output from inspector and server.")

    subparsers = parser.add_subparsers(dest="inspector_command", title="Inspector Commands", required=True)

    # List Capabilities
    list_parser = subparsers.add_parser("list-capabilities", help="List all tools and resources from the server.")
    list_parser.set_defaults(func=handle_list_capabilities)

    # Execute Tool
    exec_parser = subparsers.add_parser("execute-tool", help="Execute a specific tool on the server.")
    exec_parser.add_argument("tool_name", help="The name of the tool to execute (e.g., meta.get_server_status).")
    exec_parser.add_argument("--params", help="JSON string of parameters for the tool (e.g., '{\"key\": \"value\"}').", default="{}")
    exec_parser.set_defaults(func=handle_execute_tool)
    
    # Get Resource (Experimental for GNN MCP) - Currently commented out as per previous structure
    # resource_parser = subparsers.add_parser("get-resource", help="Attempt to retrieve a resource by URI (experimental).")
    # resource_parser.add_argument("uri", help="The URI of the resource.")
    # resource_parser.set_defaults(func=handle_get_resource)

    args = parser.parse_args()

    server_cmd_str = args.server_cmd
    print(f"Inspector: Using server command: {server_cmd_str}", file=sys.stderr)
    
    # Prepare server command for subprocess
    # shlex.split is good for this if the command is a single string.
    # If it's already a list, use that.
    if isinstance(server_cmd_str, str):
        server_cmd_list = shlex.split(server_cmd_str)
    else: # Assuming it could be pre-split if not default
        server_cmd_list = server_cmd_str

    if not Path(server_cmd_list[1]).is_file() and server_cmd_list[0] == PYTHON_EXECUTABLE : # Check if script path exists
         print(f"Inspector Error: MCP CLI script not found at {server_cmd_list[1]}", file=sys.stderr)
         print(f"Please ensure GNN_PROJECT_ROOT is correct or provide full path in --server-cmd.", file=sys.stderr)
         sys.exit(1)
    
    if args.verbose:
        if "--verbose" not in server_cmd_list and "server" in server_cmd_list : # Add verbose to server if not present
            try:
                server_idx = server_cmd_list.index("server")
                server_cmd_list.insert(server_idx, "--verbose") # GNN MCP CLI uses -v or --verbose at main level
            except ValueError:
                 # 'server' command not found, maybe it's a direct script call.
                 # For simplicity, we assume the main CLI is used.
                 pass # Don't add verbose if we can't find where to put it.
        print(f"Inspector: Augmented server command for verbose: {' '.join(server_cmd_list)}", file=sys.stderr)


    server_process = None
    client = None
    try:
        print(f"Inspector: Starting GNN MCP server process...", file=sys.stderr)
        server_process = subprocess.Popen(
            server_cmd_list,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, # Capture server's stderr separately
            text=True, # Work with text streams
            cwd=GNN_PROJECT_ROOT # Run from project root
        )
        
        # Give server a moment to start, especially if it logs to stderr/stdout on startup
        time.sleep(1 if "stdio" in server_cmd_str else 3) # Longer for http potentially

        if server_process.poll() is not None:
            print(f"Inspector Error: Server process terminated prematurely (exit code {server_process.returncode}).", file=sys.stderr)
            print("--- Server stderr (if any) ---", file=sys.stderr)
            if server_process.stderr:
                 for line in server_process.stderr: print(line.strip(), file=sys.stderr)
            print("-----------------------------", file=sys.stderr)
            sys.exit(1)

        print("Inspector: Server process started. Initializing client...", file=sys.stderr)
        # For now, only StdioMCPClient is implemented as it's simpler to manage process I/O.
        # An HTTP client would use `requests` and connect to the server's host/port.
        if "stdio" in server_cmd_str:
            client = StdioMCPClient(server_process)
        elif "http" in server_cmd_str:
            # HTTP client would be different. For now, raise error if HTTP is specified.
            # TODO: Implement an HTTP client similar to StdioMCPClient if needed.
            print("Inspector Error: HTTP client mode for inspector is not yet fully implemented.", file=sys.stderr)
            print("Please use stdio transport for the server with this inspector version.", file=sys.stderr)
            sys.exit(1)
        else:
            print("Inspector Error: Could not determine server transport from command. Assuming stdio.", file=sys.stderr)
            client = StdioMCPClient(server_process)


        # Execute the inspector command
        if hasattr(args, 'func'):
            args.func(client, args)

    except FileNotFoundError:
        print(f"Inspector Error: Could not find server command '{server_cmd_list[0]}'. Is it in PATH or path correct?", file=sys.stderr)
    except ConnectionError as e:
        print(f"Inspector Error: Connection to server failed: {e}", file=sys.stderr)
    except TimeoutError as e:
        print(f"Inspector Error: Timeout communicating with server: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Inspector: An unexpected error occurred: {e}", file=sys.stderr)
        print(f"Details: {type(e).__name__}: {e.args}", file=sys.stderr)

    finally:
        if server_process:
            print("Inspector: Shutting down server process...", file=sys.stderr)
            if server_process.stdin:
                server_process.stdin.close() # Signal EOF to server if it's reading stdin
            
            # Give threads a chance to process remaining output
            if client and client.stdout_thread.is_alive(): client.stdout_thread.join(timeout=0.5)
            if client and client.stderr_thread.is_alive(): client.stderr_thread.join(timeout=0.5)

            if server_process.poll() is None: # If still running
                server_process.terminate()
                try:
                    server_process.wait(timeout=2) # Wait for termination
                except subprocess.TimeoutExpired:
                    print("Inspector: Server did not terminate gracefully, killing.", file=sys.stderr)
                    server_process.kill()
            print("Inspector: Server process shut down.", file=sys.stderr)
            
            # Drain any remaining output from queues (after threads might have exited)
            if client:
                print("--- Remaining Server Stdout ---", file=sys.stderr)
                while not client.server_stdout_queue.empty():
                    print(client.server_stdout_queue.get_nowait().strip(), file=sys.stderr)
                print("--- Remaining Server Stderr ---", file=sys.stderr)
                while not client.server_stderr_queue.empty():
                    print(client.server_stderr_queue.get_nowait().strip(), file=sys.stderr)
                print("-----------------------------", file=sys.stderr)


if __name__ == "__main__":
    main()
