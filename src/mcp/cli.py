#!/usr/bin/env python3
"""
MCP Command Line Interface

This module provides a comprehensive command-line interface for the Model Context Protocol (MCP),
allowing users to interact with GNN tools and resources directly from the command line.

Key Features:
- List all available tools and resources
- Execute tools with JSON parameters
- Retrieve resources by URI
- Start MCP servers (stdio/HTTP)
- Get server status and tool information
- Comprehensive error handling and logging
"""
import argparse
import json
import sys
import logging
from pathlib import Path
import importlib.util
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger("mcp.cli")

def import_mcp():
    """Import the MCP module dynamically."""
    try:
        from . import mcp_instance, initialize, MCPError
        return mcp_instance, initialize, MCPError
    except ImportError:
        # Try to import from path
        mcp_path = Path(__file__).parent / "mcp.py"
        if not mcp_path.exists():
            raise ImportError("MCP module not found")
            
        spec = importlib.util.spec_from_file_location("mcp", mcp_path)
        mcp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mcp_module)
        
        return mcp_module.mcp_instance, mcp_module.initialize, getattr(mcp_module, 'MCPError', Exception)

def list_capabilities(args):
    """List all available MCP capabilities."""
    try:
        mcp_instance, initialize, MCPError = import_mcp()
        initialize()
        
        capabilities = mcp_instance.get_capabilities()
        
        if args.format == "json":
            print(json.dumps(capabilities, indent=2))
        else:
            # Human-readable format
            print("=== GNN MCP Server Capabilities ===\n")
            
            print(f"Server: {capabilities.get('name', 'Unknown')}")
            print(f"Version: {capabilities.get('version', 'Unknown')}")
            print(f"Description: {capabilities.get('description', 'No description')}")
            
            # Server info
            server_info = capabilities.get('server_info', {})
            if server_info:
                print(f"\nServer Status:")
                print(f"  Uptime: {server_info.get('uptime', 0):.1f} seconds")
                print(f"  Requests: {server_info.get('request_count', 0)}")
                print(f"  Errors: {server_info.get('error_count', 0)}")
            
            # Tools
            tools = capabilities.get('tools', {})
            if tools:
                print(f"\nAvailable Tools ({len(tools)}):")
                for name, tool_info in tools.items():
                    print(f"  {name}")
                    print(f"    Description: {tool_info.get('description', 'No description')}")
                    print(f"    Module: {tool_info.get('module', 'Unknown')}")
                    print(f"    Category: {tool_info.get('category', 'General')}")
                    print(f"    Version: {tool_info.get('version', '1.0.0')}")
                    print()
            
            # Resources
            resources = capabilities.get('resources', {})
            if resources:
                print(f"Available Resources ({len(resources)}):")
                for uri, resource_info in resources.items():
                    print(f"  {uri}")
                    print(f"    Description: {resource_info.get('description', 'No description')}")
                    print(f"    Module: {resource_info.get('module', 'Unknown')}")
                    print()
            
            # Modules
            modules = capabilities.get('modules', {})
            if modules:
                print(f"Loaded Modules ({len(modules)}):")
                for name, module_info in modules.items():
                    status = module_info.get('status', 'unknown')
                    tools_count = module_info.get('tools_count', 0)
                    resources_count = module_info.get('resources_count', 0)
                    print(f"  {name}: {status} ({tools_count} tools, {resources_count} resources)")
                    
    except Exception as e:
        logger.error(f"Error listing capabilities: {e}")
        sys.exit(1)

def execute_tool(args):
    """Execute an MCP tool with the given parameters."""
    try:
        mcp_instance, initialize, MCPError = import_mcp()
        initialize()
        
        params = {}
        if args.params:
            try:
                params = json.loads(args.params)
            except json.JSONDecodeError:
                logger.error("Invalid JSON parameters")
                sys.exit(1)
        
        if not isinstance(params, dict):
            logger.error("Parameters must be a JSON object")
            sys.exit(1)
        
        result = mcp_instance.execute_tool(args.tool_name, params)
        
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            # Human-readable format
            print(f"Tool '{args.tool_name}' executed successfully:")
            print(json.dumps(result, indent=2))
            
    except MCPError as e:
        logger.error(f"MCP Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error executing tool: {e}")
        sys.exit(1)

def get_resource(args):
    """Retrieve an MCP resource."""
    try:
        mcp_instance, initialize, MCPError = import_mcp()
        initialize()
        
        result = mcp_instance.get_resource(args.uri)
        
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            # Human-readable format
            print(f"Resource '{args.uri}' retrieved successfully:")
            print(json.dumps(result, indent=2))
            
    except MCPError as e:
        logger.error(f"MCP Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error retrieving resource: {e}")
        sys.exit(1)

def get_server_status(args):
    """Get detailed server status information."""
    try:
        mcp_instance, initialize, MCPError = import_mcp()
        initialize()
        
        status = mcp_instance.get_server_status()
        
        if args.format == "json":
            print(json.dumps(status, indent=2))
        else:
            # Human-readable format
            print("=== GNN MCP Server Status ===\n")
            print(f"Status: {status.get('status', 'Unknown')}")
            print(f"Uptime: {status.get('uptime_formatted', 'Unknown')}")
            print(f"Request Count: {status.get('request_count', 0)}")
            print(f"Error Count: {status.get('error_count', 0)}")
            print(f"Error Rate: {status.get('error_rate', 0):.2%}")
            print(f"Tools: {status.get('tools_count', 0)}")
            print(f"Resources: {status.get('resources_count', 0)}")
            print(f"Modules: {status.get('modules_count', 0)} loaded, {status.get('modules_failed', 0)} failed")
            
            # Average execution times
            avg_times = status.get('avg_execution_times', {})
            if avg_times:
                print(f"\nAverage Execution Times:")
                for tool, time_avg in avg_times.items():
                    print(f"  {tool}: {time_avg:.3f}s")
                    
    except Exception as e:
        logger.error(f"Error getting server status: {e}")
        sys.exit(1)

def get_tool_info(args):
    """Get detailed information about a specific tool."""
    try:
        mcp_instance, initialize, MCPError = import_mcp()
        initialize()
        
        tool_info = mcp_instance.tools.get(args.tool_name)
        if not tool_info:
            logger.error(f"Tool '{args.tool_name}' not found")
            sys.exit(1)
        
        info = {
            "name": tool_info.name,
            "description": tool_info.description,
            "schema": tool_info.schema,
            "module": tool_info.module,
            "category": tool_info.category,
            "version": tool_info.version
        }
        
        if args.format == "json":
            print(json.dumps(info, indent=2))
        else:
            # Human-readable format
            print(f"=== Tool Information: {info['name']} ===\n")
            print(f"Description: {info['description']}")
            print(f"Module: {info['module']}")
            print(f"Category: {info['category']}")
            print(f"Version: {info['version']}")
            print(f"\nSchema:")
            print(json.dumps(info['schema'], indent=2))
            
    except Exception as e:
        logger.error(f"Error getting tool info: {e}")
        sys.exit(1)

def start_server(args):
    """Start the MCP server."""
    try:
        # Import specific server implementation based on transport type
        if args.transport == "stdio":
            from .server_stdio import start_stdio_server
            logger.info("Starting MCP stdio server...")
            start_stdio_server()
        elif args.transport == "http":
            from .server_http import start_http_server
            logger.info(f"Starting MCP HTTP server on {args.host}:{args.port}...")
            start_http_server(args.host, args.port)
        else:
            logger.error(f"Unsupported transport: {args.transport}")
            sys.exit(1)
    except ImportError as e:
        logger.error(f"Failed to import server implementation: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Model Context Protocol CLI for GNN",
        epilog="Example: python -m src.mcp.cli list --format human"
    )
    
    # Global options
    parser.add_argument("--format", choices=["json", "human"], default="human",
                       help="Output format (default: human)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
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
    
    # Get server status
    status_parser = subparsers.add_parser("status", help="Get server status")
    status_parser.set_defaults(func=get_server_status)
    
    # Get tool info
    info_parser = subparsers.add_parser("info", help="Get tool information")
    info_parser.add_argument("tool_name", help="Name of the tool")
    info_parser.set_defaults(func=get_tool_info)
    
    # Start server
    server_parser = subparsers.add_parser("server", help="Start MCP server")
    server_parser.add_argument("--transport", choices=["stdio", "http"], default="stdio",
                              help="Transport mechanism to use (default: stdio)")
    server_parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP server")
    server_parser.add_argument("--port", type=int, default=8080, help="Port for HTTP server")
    server_parser.set_defaults(func=start_server)
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
                           format='%(asctime)s - %(levelname)s - %(message)s')
    
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    
    args.func(args)

if __name__ == "__main__":
    main() 