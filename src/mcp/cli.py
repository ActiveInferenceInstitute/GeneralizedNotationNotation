#!/usr/bin/env python3
import argparse
import json
import sys
import logging
from pathlib import Path
import importlib.util

# Configure logging
# logging.basicConfig(level=logging.INFO, 
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp.cli")

def import_mcp():
    """Import the MCP module dynamically."""
    try:
        from . import mcp_instance, initialize
        return mcp_instance, initialize
    except ImportError:
        # Try to import from path
        mcp_path = Path(__file__).parent / "mcp.py"
        if not mcp_path.exists():
            raise ImportError("MCP module not found")
            
        spec = importlib.util.spec_from_file_location("mcp", mcp_path)
        mcp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mcp_module)
        
        return mcp_module.mcp_instance, mcp_module.initialize

def list_capabilities(args):
    """List all available MCP capabilities."""
    mcp_instance, initialize = import_mcp()
    initialize()
    
    capabilities = mcp_instance.get_capabilities()
    print(json.dumps(capabilities, indent=2))

def execute_tool(args):
    """Execute an MCP tool with the given parameters."""
    mcp_instance, initialize = import_mcp()
    initialize()
    
    params = {}
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError:
            logger.error("Invalid JSON parameters")
            sys.exit(1)
    
    try:
        result = mcp_instance.execute_tool(args.tool_name, params)
        print(json.dumps(result, indent=2))
    except Exception as e:
        logger.error(f"Error executing tool: {str(e)}")
        sys.exit(1)

def get_resource(args):
    """Retrieve an MCP resource."""
    mcp_instance, initialize = import_mcp()
    initialize()
    
    try:
        result = mcp_instance.get_resource(args.uri)
        print(json.dumps(result, indent=2))
    except Exception as e:
        logger.error(f"Error retrieving resource: {str(e)}")
        sys.exit(1)

def start_server(args):
    """Start the MCP server."""
    try:
        # Import specific server implementation based on transport type
        if args.transport == "stdio":
            from .server_stdio import start_stdio_server
            start_stdio_server()
        elif args.transport == "http":
            from .server_http import start_http_server
            start_http_server(args.host, args.port)
        else:
            logger.error(f"Unsupported transport: {args.transport}")
            sys.exit(1)
    except ImportError as e:
        logger.error(f"Failed to import server implementation: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Model Context Protocol CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List capabilities
    list_parser = subparsers.add_parser("list", help="List available capabilities")
    list_parser.set_defaults(func=list_capabilities)
    
    # Execute tool
    execute_parser = subparsers.add_parser("execute", help="Execute a tool")
    execute_parser.add_argument("tool_name", help="Name of the tool to execute")
    execute_parser.add_argument("--params", help="JSON parameters for the tool")
    execute_parser.set_defaults(func=execute_tool)
    
    # Get resource
    resource_parser = subparsers.add_parser("resource", help="Get a resource")
    resource_parser.add_argument("uri", help="URI of the resource to retrieve")
    resource_parser.set_defaults(func=get_resource)
    
    # Start server
    server_parser = subparsers.add_parser("server", help="Start MCP server")
    server_parser.add_argument("--transport", choices=["stdio", "http"], default="stdio",
                              help="Transport mechanism to use")
    server_parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP server")
    server_parser.add_argument("--port", type=int, default=8080, help="Port for HTTP server")
    server_parser.set_defaults(func=start_server)
    
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    
    args.func(args)

if __name__ == "__main__":
    # Basic configuration for running this script standalone
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    main() 