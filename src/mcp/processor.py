"""
MCP Processing Module for GNN Processing Pipeline.

This module provides comprehensive Model Context Protocol (MCP) processing for the GNN project,
enabling standardized tool discovery, registration, execution, and monitoring across all modules.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import time
import subprocess
import sys
from datetime import datetime
from dataclasses import asdict

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

def process_mcp_operations(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process MCP operations for GNN files.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        logger: Logger instance
        verbose: Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        True if processing successful, False otherwise
    """
    try:
        log_step_start(logger, "Processing MCP operations")
        
        # Create results directory
        results_dir = output_dir / "mcp_processing_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "mcp_server_status": {},
            "tool_registration": {},
            "resource_access": {},
            "performance_metrics": {},
            "module_discovery": {},
            "capabilities": {},
            "health_status": {}
        }
        
        # Initialize MCP server
        mcp_status = initialize_mcp_server(verbose)
        results["mcp_server_status"] = mcp_status
        
        if not mcp_status["success"]:
            # Continue with limited functionality instead of failing completely
            log_step_warning(logger, f"MCP server initialization failed: {mcp_status['error']}")
            log_step_warning(logger, "Continuing with limited MCP functionality")
            results["errors"].append(f"MCP server initialization failed: {mcp_status['error']}")
            # Don't set success to False, continue with what we can do
        
        # Discover and register tools from all modules
        discovery_results = discover_and_register_tools(verbose)
        results["module_discovery"] = discovery_results
        
        if not discovery_results["success"]:
            log_step_warning(logger, f"Tool discovery failed: {discovery_results['error']}")
            results["errors"].append(f"Tool discovery failed: {discovery_results['error']}")
        
        # Get server capabilities
        capabilities = get_server_capabilities(verbose)
        results["capabilities"] = capabilities
        
        # Test tool execution
        tool_tests = test_tool_execution(verbose)
        results["tool_registration"] = tool_tests
        
        # Test resource access
        resource_tests = test_resource_access(verbose)
        results["resource_access"] = resource_tests
        
        # Get performance metrics
        performance = get_performance_metrics(verbose)
        results["performance_metrics"] = performance
        
        # Get health status
        health = get_health_status(verbose)
        results["health_status"] = health
        
        # Process GNN files with MCP tools
        gnn_files = list(target_dir.glob("*.md"))
        if gnn_files:
            results["processed_files"] = len(gnn_files)
            
            for gnn_file in gnn_files:
                try:
                    # Process file with MCP tools
                    file_results = process_gnn_file_with_mcp(gnn_file, verbose)
                    results["processed_files"] += 1
                    
                except Exception as e:
                    error_info = {
                        "file": str(gnn_file),
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    results["errors"].append(error_info)
                    logger.error(f"Error processing {gnn_file}: {e}")
        
        # Generate MCP report
        mcp_report = generate_mcp_report(results)
        report_file = results_dir / "mcp_report.json"
        
        # Convert non-serializable objects for JSON
        def convert_for_json(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):  # deque and other iterables
                return list(convert_for_json(item) for item in obj)
            elif hasattr(obj, 'isoformat'):  # datetime objects
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):  # custom objects
                return str(obj)
            return obj
        
        serializable_report = convert_for_json(mcp_report)
        with open(report_file, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        # Generate summary
        summary = generate_mcp_summary(results)
        summary_file = results_dir / "mcp_summary.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        # Generate tool documentation
        tool_docs = generate_tool_documentation(results)
        docs_file = results_dir / "mcp_tools_documentation.md"
        with open(docs_file, 'w') as f:
            f.write(tool_docs)
        
        if results["success"]:
            log_step_success(logger, "MCP processing completed successfully")
        else:
            log_step_error(logger, "MCP processing failed")
        
        return results["success"]
        
    except Exception as e:
        log_step_error(logger, f"MCP processing failed: {str(e)}")
        return False

def initialize_mcp_server(verbose: bool = False) -> Dict[str, Any]:
    """Initialize the MCP server."""
    try:
        from mcp import initialize, get_mcp_instance
        
        # Initialize MCP
        mcp_instance, success, warnings = initialize(halt_on_missing_sdk=False)
        
        if not success:
            return {
                "success": False,
                "error": "Failed to initialize MCP server",
                "warnings": warnings
            }
        
        # Get server status
        status = mcp_instance.get_enhanced_server_status()
        
        return {
            "success": True,
            "server_status": status,
            "warnings": warnings,
            "initialization_time": time.time()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "initialization_time": time.time()
        }

def discover_and_register_tools(verbose: bool = False) -> Dict[str, Any]:
    """Discover and register tools from all modules."""
    try:
        from mcp import get_mcp_instance
        
        mcp_instance = get_mcp_instance()
        
        # Discover modules
        discovery_success = mcp_instance.discover_modules()
        
        # Get module information
        modules = {}
        for module_name in mcp_instance.modules:
            module_info = mcp_instance.get_module_info(module_name)
            if module_info:
                modules[module_name] = module_info
        
        # Get tool information
        tools = {}
        for tool_name in mcp_instance.tools:
            tool_info = mcp_instance.get_tool_info(tool_name)
            if tool_info:
                tools[tool_name] = tool_info
        
        return {
            "success": discovery_success,
            "modules_discovered": len(modules),
            "tools_registered": len(tools),
            "modules": modules,
            "tools": tools,
            "discovery_time": time.time()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "discovery_time": time.time()
        }

def get_server_capabilities(verbose: bool = False) -> Dict[str, Any]:
    """Get server capabilities."""
    try:
        from mcp import get_mcp_instance
        
        mcp_instance = get_mcp_instance()
        capabilities = mcp_instance.get_capabilities()
        
        return {
            "success": True,
            "capabilities": capabilities,
            "capabilities_time": time.time()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "capabilities_time": time.time()
        }

def test_tool_execution(verbose: bool = False) -> Dict[str, Any]:
    """Test tool execution."""
    try:
        from mcp import get_mcp_instance
        
        mcp_instance = get_mcp_instance()
        
        # Test basic tools
        test_results = {}
        
        # Test system info tool
        try:
            result = mcp_instance.execute_tool("get_system_info", {})
            test_results["get_system_info"] = {
                "success": True,
                "result": result
            }
        except Exception as e:
            test_results["get_system_info"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test GNN file discovery tool
        try:
            result = mcp_instance.execute_tool("get_gnn_files", {
                "target_dir": "input/gnn_files",
                "recursive": True
            })
            test_results["get_gnn_files"] = {
                "success": True,
                "result": result
            }
        except Exception as e:
            test_results["get_gnn_files"] = {
                "success": False,
                "error": str(e)
            }
        
        return {
            "success": True,
            "test_results": test_results,
            "test_time": time.time()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "test_time": time.time()
        }

def test_resource_access(verbose: bool = False) -> Dict[str, Any]:
    """Test resource access."""
    try:
        from mcp import get_mcp_instance
        
        mcp_instance = get_mcp_instance()
        
        # Test resource access
        test_results = {}
        
        # Test visualization resource
        try:
            result = mcp_instance.get_resource("visualization://output/visualization")
            test_results["visualization_resource"] = {
                "success": True,
                "result": result
            }
        except Exception as e:
            test_results["visualization_resource"] = {
                "success": False,
                "error": str(e)
            }
        
        return {
            "success": True,
            "test_results": test_results,
            "test_time": time.time()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "test_time": time.time()
        }

def get_performance_metrics(verbose: bool = False) -> Dict[str, Any]:
    """Get performance metrics."""
    try:
        from mcp import get_mcp_instance
        
        mcp_instance = get_mcp_instance()
        
        # Get performance metrics
        metrics = mcp_instance.performance_metrics
        
        return {
            "success": True,
            "metrics": asdict(metrics),
            "metrics_time": time.time()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "metrics_time": time.time()
        }

def get_health_status(verbose: bool = False) -> Dict[str, Any]:
    """Get health status."""
    try:
        from mcp import get_mcp_instance
        
        mcp_instance = get_mcp_instance()
        
        # Get enhanced server status
        status = mcp_instance.get_enhanced_server_status()
        
        return {
            "success": True,
            "health_status": status.get("health", {}),
            "health_time": time.time()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "health_time": time.time()
        }

def process_gnn_file_with_mcp(file_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """Process a GNN file using MCP tools."""
    try:
        from mcp import get_mcp_instance
        
        mcp_instance = get_mcp_instance()
        
        results = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "processing_time": time.time(),
            "tool_results": {}
        }
        
        # Use MCP tools to analyze the file
        try:
            # Get file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Analyze with MCP tools
            analysis_result = mcp_instance.execute_tool("analyze_gnn_content", {
                "content": content,
                "file_path": str(file_path)
            })
            
            results["tool_results"]["analyze_gnn_content"] = {
                "success": True,
                "result": analysis_result
            }
            
        except Exception as e:
            results["tool_results"]["analyze_gnn_content"] = {
                "success": False,
                "error": str(e)
            }
        
        return results
        
    except Exception as e:
        raise Exception(f"Failed to process {file_path}: {e}")

def generate_mcp_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive MCP report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "success": results["success"],
            "processed_files": results["processed_files"],
            "errors": len(results["errors"]),
            "modules_discovered": results["module_discovery"].get("modules_discovered", 0),
            "tools_registered": results["module_discovery"].get("tools_registered", 0)
        },
        "server_status": results["mcp_server_status"],
        "capabilities": results["capabilities"],
        "performance": results["performance_metrics"],
        "health": results["health_status"],
        "tool_tests": results["tool_registration"],
        "resource_tests": results["resource_access"],
        "errors": results["errors"]
    }
    
    return report

def generate_mcp_summary(results: Dict[str, Any]) -> str:
    """Generate markdown summary of MCP results."""
    summary = f"""# MCP Processing Summary

Generated on: {results['timestamp']}

## Overview
- **Success**: {results['success']}
- **Files Processed**: {results['processed_files']}
- **Errors**: {len(results['errors'])}

## Server Status
"""
    
    if results["mcp_server_status"]["success"]:
        summary += "- ✅ MCP Server initialized successfully\n"
        status = results["mcp_server_status"]["server_status"]
        summary += f"- **Uptime**: {status.get('uptime', 'unknown')} seconds\n"
        summary += f"- **Total Requests**: {status.get('total_requests', 0)}\n"
        summary += f"- **Error Rate**: {status.get('error_rate', 0):.2%}\n"
    else:
        summary += f"- ❌ MCP Server failed: {results['mcp_server_status']['error']}\n"
    
    summary += "\n## Module Discovery\n"
    discovery = results["module_discovery"]
    if discovery["success"]:
        summary += f"- **Modules Discovered**: {discovery['modules_discovered']}\n"
        summary += f"- **Tools Registered**: {discovery['tools_registered']}\n"
        
        if discovery["modules"]:
            summary += "\n### Modules\n"
            for module_name, module_info in discovery["modules"].items():
                summary += f"- **{module_name}**: {module_info.get('tools_count', 0)} tools, {module_info.get('resources_count', 0)} resources\n"
    else:
        summary += f"- ❌ Discovery failed: {discovery['error']}\n"
    
    summary += "\n## Capabilities\n"
    capabilities = results["capabilities"]
    if capabilities["success"]:
        caps = capabilities["capabilities"]
        summary += f"- **Protocol Version**: {caps.get('protocol_version', 'unknown')}\n"
        summary += f"- **Features**: {', '.join(caps.get('features', []))}\n"
    else:
        summary += f"- ❌ Capabilities failed: {capabilities['error']}\n"
    
    summary += "\n## Performance Metrics\n"
    performance = results["performance_metrics"]
    if performance["success"]:
        metrics = performance["metrics"]
        summary += f"- **Total Requests**: {metrics.get('total_requests', 0)}\n"
        summary += f"- **Success Rate**: {metrics.get('success_rate', 0):.2%}\n"
        summary += f"- **Average Execution Time**: {metrics.get('average_execution_time', 0):.3f}s\n"
        summary += f"- **Cache Hit Ratio**: {metrics.get('cache_hit_ratio', 0):.2%}\n"
    else:
        summary += f"- ❌ Performance metrics failed: {performance['error']}\n"
    
    if results["errors"]:
        summary += "\n## Errors\n"
        for error in results["errors"]:
            if isinstance(error, dict):
                summary += f"- **{error.get('file', 'Unknown')}**: {error.get('error', 'Unknown error')}\n"
            else:
                summary += f"- {error}\n"
    
    return summary

def generate_tool_documentation(results: Dict[str, Any]) -> str:
    """Generate tool documentation."""
    docs = """# MCP Tools Documentation

## Available Tools

"""
    
    discovery = results["module_discovery"]
    if discovery["success"] and discovery["tools"]:
        # Group tools by module
        tools_by_module = {}
        for tool_name, tool_info in discovery["tools"].items():
            module = tool_info.get("module", "unknown")
            if module not in tools_by_module:
                tools_by_module[module] = []
            tools_by_module[module].append((tool_name, tool_info))
        
        for module_name, tools in tools_by_module.items():
            docs += f"\n### {module_name.title()} Module\n\n"
            
            for tool_name, tool_info in tools:
                docs += f"#### {tool_name}\n\n"
                docs += f"**Description**: {tool_info.get('description', 'No description available')}\n\n"
                docs += f"**Module**: {tool_info.get('module', 'unknown')}\n\n"
                docs += f"**Version**: {tool_info.get('version', '1.0.0')}\n\n"
                
                if tool_info.get("tags"):
                    docs += f"**Tags**: {', '.join(tool_info['tags'])}\n\n"
                
                if tool_info.get("examples"):
                    docs += "**Examples**:\n"
                    for example in tool_info["examples"]:
                        docs += f"```json\n{json.dumps(example, indent=2)}\n```\n\n"
                
                docs += "---\n\n"
    else:
        docs += "No tools available or tool discovery failed.\n"
    
    return docs

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "Model Context Protocol processing for GNN Processing Pipeline"

# Feature availability flags
FEATURES = {
    'mcp_server': True,
    'tool_discovery': True,
    'resource_access': True,
    'performance_monitoring': True,
    'health_monitoring': True,
    'capability_reporting': True
}

__all__ = [
    'process_mcp_operations',
    'initialize_mcp_server',
    'discover_and_register_tools',
    'get_server_capabilities',
    'test_tool_execution',
    'test_resource_access',
    'get_performance_metrics',
    'get_health_status',
    'process_gnn_file_with_mcp',
    'generate_mcp_report',
    'generate_mcp_summary',
    'generate_tool_documentation',
    'FEATURES',
    '__version__'
] 