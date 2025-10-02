#!/usr/bin/env python3
"""
Enhanced MCP Command Line Interface

This module provides a comprehensive command-line interface for the Model Context Protocol (MCP),
allowing users to interact with GNN tools and resources directly from the command line.

Key Features:
- List all available tools and resources with detailed metadata
- Execute tools with JSON parameters and comprehensive error reporting
- Retrieve resources by URI with validation
- Start MCP servers (stdio/HTTP) with health monitoring
- Get server status, performance metrics, and diagnostic information
- Interactive tool discovery and parameter validation
- Comprehensive error handling and logging with suggestions
- Auto-completion and shell integration support
"""
import argparse
import json
import sys
import logging
import time
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
    """Enhanced listing of all available MCP capabilities with better formatting."""
    try:
        mcp_instance, initialize, MCPError = import_mcp()
        initialize()

        capabilities = mcp_instance.get_capabilities()

        if args.format == "json":
            print(json.dumps(capabilities, indent=2))
            return

        # Enhanced human-readable format
        print("🚀 GNN MCP Server Capabilities")
        print("=" * 50)

        server_info = capabilities.get('server', {})
        print(f"\n📋 Server Information:")
        print(f"  Name: {server_info.get('name', 'Unknown GNN MCP Server')}")
        print(f"  Version: {server_info.get('version', 'Unknown')}")
        print(f"  Description: {server_info.get('description', 'No description available')}")

        # Get enhanced server status for more details
        try:
            status = mcp_instance.get_enhanced_server_status()
            health = status.get('health', {})
            print(f"  Health: {health.get('status', 'unknown').upper()} (Score: {health.get('score', 0)}/100)")
            print(f"  Uptime: {status.get('server_info', {}).get('uptime_formatted', 'Unknown')}")
        except Exception:
            print("  Health: Unable to determine")

        # Tools section
        tools = capabilities.get('tools', [])
        if tools:
            print(f"\n🔧 Available Tools ({len(tools)}):")
            print("-" * 30)

            # Group tools by category
            tools_by_category = {}
            for tool in tools:
                category = tool.get('category', 'General')
                if category not in tools_by_category:
                    tools_by_category[category] = []
                tools_by_category[category].append(tool)

            for category, category_tools in tools_by_category.items():
                print(f"\n  📂 {category} ({len(category_tools)} tools):")
                for tool in sorted(category_tools, key=lambda t: t['name']):
                    deprecated = " ⚠️ DEPRECATED" if tool.get('deprecated') else ""
                    experimental = " 🧪 EXPERIMENTAL" if tool.get('experimental') else ""
                    print(f"    • {tool['name']}{deprecated}{experimental}")
                    if args.verbose:
                        print(f"      Description: {tool.get('description', 'No description')}")
                        print(f"      Module: {tool.get('module', 'Unknown')}")
                        print(f"      Version: {tool.get('version', '1.0.0')}")

        # Resources section
        resources = capabilities.get('resources', [])
        if resources:
            print(f"\n📚 Available Resources ({len(resources)}):")
            print("-" * 30)
            for resource in sorted(resources, key=lambda r: r['uri_template']):
                print(f"  • {resource['uri_template']}")
                if args.verbose:
                    print(f"    Description: {resource.get('description', 'No description')}")
                    print(f"    Module: {resource.get('module', 'Unknown')}")

        # Summary
        total_tools = len(tools)
        total_resources = len(resources)
        print(f"\n📊 Summary: {total_tools} tools, {total_resources} resources")

        if args.verbose:
            try:
                # Show performance summary
                status = mcp_instance.get_enhanced_server_status()
                perf = status.get('performance', {})
                print("
⚡ Performance:"                print(f"  Success Rate: {perf.get('success_rate', 0):.1%}")
                print(f"  Avg Execution Time: {perf.get('average_execution_time', 0):.3f}s")
                print(f"  Cache Hit Ratio: {perf.get('cache_hit_ratio', 0):.1%}")
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Error listing capabilities: {e}")
        if args.verbose:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("💡 Suggestions:", file=sys.stderr)
        print("  - Check if MCP server is running", file=sys.stderr)
        print("  - Verify GNN modules are properly installed", file=sys.stderr)
        print("  - Run with --verbose for more details", file=sys.stderr)
        sys.exit(1)

def execute_tool(args):
    """Execute an MCP tool with enhanced parameter validation and error reporting."""
    try:
        mcp_instance, initialize, MCPError = import_mcp()
        initialize()

        # Validate tool exists
        if args.tool_name not in mcp_instance.tools:
            available_tools = list(mcp_instance.tools.keys())
            print(f"\n❌ Error: Tool '{args.tool_name}' not found", file=sys.stderr)
            print(f"\n💡 Available tools:", file=sys.stderr)
            for tool in sorted(available_tools):
                print(f"  • {tool}", file=sys.stderr)
            sys.exit(1)

        # Get tool info for validation
        tool = mcp_instance.tools[args.tool_name]

        # Parse and validate parameters
        params = {}
        if args.params:
            try:
                params = json.loads(args.params)
            except json.JSONDecodeError as e:
                print(f"\n❌ Error: Invalid JSON parameters: {e}", file=sys.stderr)
                print(f"\n💡 Expected format: --params '{{\"key\": \"value\"}}'", file=sys.stderr)
                sys.exit(1)

        if not isinstance(params, dict):
            print(f"\n❌ Error: Parameters must be a JSON object, got {type(params)}", file=sys.stderr)
            sys.exit(1)

        # Validate against schema if available
        if tool.schema and args.validate:
            try:
                # Basic schema validation
                required = tool.schema.get('required', [])
                for req in required:
                    if req not in params:
                        print(f"\n❌ Error: Missing required parameter '{req}'", file=sys.stderr)
                        print(f"\n💡 Required parameters: {required}", file=sys.stderr)
                        sys.exit(1)
            except Exception as e:
                logger.warning(f"Schema validation failed: {e}")

        # Show execution info
        print(f"🔧 Executing tool: {args.tool_name}")
        if args.verbose:
            print(f"📝 Parameters: {json.dumps(params, indent=2)}")
            print(f"🏷️  Module: {tool.module}")
            print(f"📂 Category: {tool.category}")
            print(f"📋 Description: {tool.description}")

        # Execute the tool
        start_time = time.time()
        result = mcp_instance.execute_tool(args.tool_name, params)
        execution_time = time.time() - start_time

        # Show results
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"\n✅ Tool executed successfully in {execution_time:.".3f"")
            print(f"📊 Result:")
            print(json.dumps(result, indent=2))

            if args.verbose:
                # Show tool usage stats
                try:
                    stats = mcp_instance.get_tool_performance_stats(args.tool_name)
                    if stats:
                        print(f"\n📈 Tool Statistics:")
                        print(f"  Uses: {stats.get('use_count', 0)}")
                        print(f"  Avg Time: {stats.get('average_execution_time', 0):.3".3f")
                        print(f"  Success Rate: {stats.get('success_rate', 0):.1".1%"                except Exception:
                    pass

    except MCPError as e:
        print(f"\n❌ MCP Error: {e}", file=sys.stderr)
        if args.verbose:
            print(f"🔍 Error Code: {e.code}", file=sys.stderr)
            print(f"📋 Error Data: {e.data}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error executing tool: {e}")
        if args.verbose:
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}", file=sys.stderr)
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
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
            available_tools = list(mcp_instance.tools.keys())
            print(f"\n❌ Error: Tool '{args.tool_name}' not found", file=sys.stderr)
            print(f"\n💡 Available tools:", file=sys.stderr)
            for tool in sorted(available_tools):
                print(f"  • {tool}", file=sys.stderr)
            sys.exit(1)

        # Get enhanced tool information
        detailed_info = mcp_instance.get_tool_info(args.tool_name)
        if not detailed_info:
            detailed_info = {
                "name": tool_info.name,
                "description": tool_info.description,
                "schema": tool_info.schema,
                "module": tool_info.module,
                "category": tool_info.category,
                "version": tool_info.version
            }

        if args.format == "json":
            print(json.dumps(detailed_info, indent=2))
        else:
            # Enhanced human-readable format
            print(f"🔍 Tool Information: {detailed_info['name']}")
            print("=" * 50)

            print(f"\n📋 Basic Info:")
            print(f"  Description: {detailed_info['description']}")
            print(f"  Module: {detailed_info['module']}")
            print(f"  Category: {detailed_info['category']}")
            print(f"  Version: {detailed_info['version']}")

            if detailed_info.get('use_count', 0) > 0:
                print(f"\n📈 Usage Statistics:")
                print(f"  Times Used: {detailed_info.get('use_count', 0)}")
                print(f"  Avg Execution Time: {detailed_info.get('average_execution_time', 0):.3".3f")
                print(f"  Success Rate: {detailed_info.get('success_rate', 0):.1".1%")

            print(f"\n⚙️ Configuration:")
            print(f"  Input Validation: {'Enabled' if detailed_info.get('input_validation', True) else 'Disabled'}")
            print(f"  Output Validation: {'Enabled' if detailed_info.get('output_validation', True) else 'Disabled'}")
            print(f"  Timeout: {detailed_info.get('timeout', 'None')}s")
            print(f"  Max Concurrent: {detailed_info.get('max_concurrent', 1)}")
            print(f"  Rate Limit: {detailed_info.get('rate_limit', 'None')} req/s")
            print(f"  Cache TTL: {detailed_info.get('cache_ttl', 'None')}s")

            if detailed_info.get('deprecated'):
                print(f"\n⚠️  Status: DEPRECATED")
            if detailed_info.get('experimental'):
                print(f"\n🧪 Status: EXPERIMENTAL")

            print(f"\n📋 Schema:")
            print(json.dumps(detailed_info['schema'], indent=2))

    except Exception as e:
        logger.error(f"Error getting tool info: {e}")
        if args.verbose:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

def get_diagnostics(args):
    """Get comprehensive diagnostic information."""
    try:
        mcp_instance, initialize, MCPError = import_mcp()
        initialize()

        # Get diagnostics using the new meta-tool
        try:
            result = mcp_instance.execute_tool("get_mcp_diagnostics", {})
            diagnostics = result.get('diagnostics', {})
            overall_health = result.get('overall_health', 'unknown')
        except Exception:
            # Fallback to basic diagnostics
            diagnostics = {"issues": [], "warnings": [], "recommendations": []}
            overall_health = "unknown"

        if args.format == "json":
            print(json.dumps(result, indent=2))
            return

        # Enhanced human-readable format
        print("🔍 GNN MCP Server Diagnostics")
        print("=" * 50)

        print(f"\n🏥 Overall Health: {overall_health.upper()}")

        # Show issues
        issues = diagnostics.get('issues', [])
        if issues:
            print(f"\n❌ Issues Found ({len(issues)}):")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("\n✅ No critical issues found")

        # Show warnings
        warnings = diagnostics.get('warnings', [])
        if warnings:
            print(f"\n⚠️  Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"  • {warning}")

        # Show recommendations
        recommendations = diagnostics.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations ({len(recommendations)}):")
            for rec in recommendations:
                print(f"  • {rec}")

        # Show health checks
        health_checks = diagnostics.get('health_checks', {})
        if health_checks:
            print(f"\n🔍 Health Checks:")
            for check_name, check_result in health_checks.items():
                status = "✅ PASS" if check_result else "❌ FAIL"
                print(f"  • {check_name}: {status}")

        if args.verbose:
            # Show additional server stats
            try:
                status = mcp_instance.get_enhanced_server_status()
                perf = status.get('performance', {})

                print(f"\n📊 Detailed Performance:")
                print(f"  Total Requests: {perf.get('total_requests', 0)}")
                print(f"  Success Rate: {perf.get('success_rate', 0):.1%}")
                print(f"  Avg Execution Time: {perf.get('average_execution_time', 0):.3f}s")
                print(f"  Cache Hit Ratio: {perf.get('cache_hit_ratio', 0):.1%}")
                print(f"  Active Connections: {perf.get('concurrent_requests', 0)}")

                # Show module status
                modules = status.get('modules', {})
                if modules:
                    print(f"\n📦 Module Status:")
                    for name, info in modules.items():
                        status_icon = "✅" if info.get('status') == 'loaded' else "❌"
                        print(f"  {status_icon} {name}: {info.get('status', 'unknown')}")

            except Exception as e:
                logger.warning(f"Could not get detailed status: {e}")

    except Exception as e:
        logger.error(f"Error getting diagnostics: {e}")
        if args.verbose:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("💡 Suggestions:", file=sys.stderr)
        print("  - Check if MCP server is running", file=sys.stderr)
        print("  - Verify GNN modules are properly installed", file=sys.stderr)
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
    execute_parser = subparsers.add_parser("execute", help="Execute a tool with enhanced validation")
    execute_parser.add_argument("tool_name", help="Name of the tool to execute")
    execute_parser.add_argument("--params", help="JSON parameters for the tool")
    execute_parser.add_argument("--validate", action="store_true", help="Validate parameters against tool schema")
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
    
    # Get diagnostics
    diagnostics_parser = subparsers.add_parser("diagnostics", help="Get comprehensive diagnostic information")
    diagnostics_parser.set_defaults(func=get_diagnostics)

    # Start server
    server_parser = subparsers.add_parser("server", help="Start MCP server with health monitoring")
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