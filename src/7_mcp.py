#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 7: MCP (Model Context Protocol)

This script handles MCP operations and tool registrations, providing comprehensive
status reporting and validation of the MCP implementation.

Usage:
    python 7_mcp.py [options]
    (Typically called by main.py)
"""

import sys
from pathlib import Path
import argparse
import logging
import json
import time
from typing import Dict, Any, List, Optional

# Import centralized utilities and configuration
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    UTILS_AVAILABLE
)

from pipeline import (
    STEP_METADATA,
    get_output_dir_for_script
)

# Initialize logger for this step
logger = setup_step_logging("7_mcp", verbose=False)

# Attempt to import the main MCP instance and initializer
try:
    from mcp import mcp_instance, initialize as initialize_mcp_system, MCPError
    logger.info("Successfully imported project's internal MCP components")
    MCP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Failed to import MCP components: {e}")
    mcp_instance = None
    initialize_mcp_system = None
    MCPError = Exception
    MCP_AVAILABLE = False

# Define all expected module directories that should have MCP integration
EXPECTED_MCP_MODULE_DIRS = [
    "export",
    "gnn", 
    "type_checker",
    "ontology",
    "setup",
    "tests",
    "visualization",
    "llm",
    "render",
    "execute",
    "site",
    "sapf",
    "pipeline",
    "utils",
    "src"
]

def get_mcp_capabilities() -> Dict[str, Any]:
    """Get comprehensive MCP capabilities information."""
    if not mcp_instance:
        return {"error": "MCP instance not available"}
    
    try:
        capabilities = mcp_instance.get_capabilities()
        return capabilities
    except Exception as e:
        return {"error": f"Failed to get capabilities: {e}"}

def get_mcp_server_status() -> Dict[str, Any]:
    """Get MCP server status information."""
    if not mcp_instance:
        return {"error": "MCP instance not available"}
    
    try:
        status = mcp_instance.get_server_status()
        return status
    except Exception as e:
        return {"error": f"Failed to get server status: {e}"}

def analyze_mcp_tools() -> Dict[str, Any]:
    """Analyze registered MCP tools comprehensively."""
    if not mcp_instance:
        return {"error": "MCP instance not available"}
    
    try:
        tools = getattr(mcp_instance, 'tools', {})
        modules = getattr(mcp_instance, 'modules', {})
        
        # Analyze tools by category
        tools_by_category = {}
        tools_by_module = {}
        
        for tool_name, tool_obj in tools.items():
            # Get tool information safely
            try:
                description = getattr(tool_obj, 'description', 'No description')
                module = getattr(tool_obj, 'module', 'Unknown')
                category = getattr(tool_obj, 'category', 'General')
                version = getattr(tool_obj, 'version', '1.0.0')
                
                # Group by category
                if category not in tools_by_category:
                    tools_by_category[category] = []
                tools_by_category[category].append({
                    "name": tool_name,
                    "description": description,
                    "module": module,
                    "version": version
                })
                
                # Group by module
                if module not in tools_by_module:
                    tools_by_module[module] = []
                tools_by_module[module].append({
                    "name": tool_name,
                    "description": description,
                    "category": category,
                    "version": version
                })
                
            except Exception as e:
                logger.warning(f"Error analyzing tool {tool_name}: {e}")
        
        return {
            "total_tools": len(tools),
            "tools_by_category": tools_by_category,
            "tools_by_module": tools_by_module,
            "categories_count": len(tools_by_category),
            "modules_with_tools": len(tools_by_module)
        }
    except Exception as e:
        return {"error": f"Failed to analyze tools: {e}"}

def check_module_mcp_integration() -> Dict[str, Any]:
    """Check MCP integration status for all modules."""
    src_dir = Path(__file__).parent
    module_status = {}
    
    for module_dir in EXPECTED_MCP_MODULE_DIRS:
        module_path = src_dir / module_dir
        mcp_file_path = module_path / "mcp.py"
        
        status = {
            "exists": module_path.exists(),
            "has_mcp_file": mcp_file_path.exists(),
            "has_init": (module_path / "__init__.py").exists(),
            "path": str(module_path)
        }
        
        # Check if module has register_tools function
        if mcp_file_path.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(f"mcp_{module_dir}", mcp_file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                status["has_register_tools"] = hasattr(module, 'register_tools')
                status["register_tools_type"] = type(getattr(module, 'register_tools', None)).__name__
                
            except Exception as e:
                status["import_error"] = str(e)
                status["has_register_tools"] = False
        
        module_status[module_dir] = status
    
    return module_status

def test_mcp_functionality() -> Dict[str, Any]:
    """Test basic MCP functionality."""
    if not mcp_instance:
        return {"error": "MCP instance not available"}
    
    test_results = {}
    
    # Test 1: Get capabilities
    try:
        capabilities = mcp_instance.get_capabilities()
        test_results["capabilities"] = {
            "success": True,
            "tools_count": len(capabilities.get("tools", {})),
            "resources_count": len(capabilities.get("resources", {}))
        }
    except Exception as e:
        test_results["capabilities"] = {"success": False, "error": str(e)}
    
    # Test 2: Get server status
    try:
        status = mcp_instance.get_server_status()
        test_results["server_status"] = {
            "success": True,
            "status": status.get("status", "unknown"),
            "uptime": status.get("uptime_seconds", 0)
        }
    except Exception as e:
        test_results["server_status"] = {"success": False, "error": str(e)}
    
    # Test 3: Try to execute a simple tool (if available)
    try:
        tools = getattr(mcp_instance, 'tools', {})
        if tools:
            # Try to execute the first available tool
            first_tool_name = list(tools.keys())[0]
            first_tool = tools[first_tool_name]
            
            # Check if it's a callable or has a func attribute
            if hasattr(first_tool, 'func') and callable(first_tool.func):
                # Try to call with empty params
                result = first_tool.func()
                test_results["tool_execution"] = {
                    "success": True,
                    "tool_tested": first_tool_name,
                    "result_type": type(result).__name__
                }
            else:
                test_results["tool_execution"] = {
                    "success": False,
                    "error": f"Tool {first_tool_name} is not callable"
                }
        else:
            test_results["tool_execution"] = {
                "success": False,
                "error": "No tools available for testing"
            }
    except Exception as e:
        test_results["tool_execution"] = {"success": False, "error": str(e)}
    
    return test_results

def run_mcp_operations(target_dir: Path, output_dir: Path, verbose: bool = False):
    """Run comprehensive MCP operations and generate detailed report."""
    log_step_start(logger, "Running comprehensive MCP operations")
    
    # Use centralized output directory configuration
    mcp_output_dir = get_output_dir_for_script("7_mcp.py", output_dir)
    mcp_output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Initialize MCP system if available
        if initialize_mcp_system and MCP_AVAILABLE:
            logger.info("Initializing MCP system...")
            initialize_mcp_system()
            log_step_success(logger, "MCP system initialized successfully")
        else:
            logger.warning("MCP system not available for initialization")
        
        # Collect comprehensive MCP information
        logger.info("Collecting MCP capabilities...")
        capabilities = get_mcp_capabilities()
        
        logger.info("Collecting MCP server status...")
        server_status = get_mcp_server_status()
        
        logger.info("Analyzing MCP tools...")
        tools_analysis = analyze_mcp_tools()
        
        logger.info("Checking module MCP integration...")
        module_integration = check_module_mcp_integration()
        
        logger.info("Testing MCP functionality...")
        functionality_tests = test_mcp_functionality()
        
        # Generate comprehensive report
        report_content = []
        report_content.append("# MCP (Model Context Protocol) Operations Report\n")
        report_content.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System Status
        report_content.append("## MCP System Status\n")
        if MCP_AVAILABLE and mcp_instance:
            report_content.append("- MCP Instance: ✅ Available\n")
            report_content.append("- Initialization: ✅ Successful\n")
            report_content.append("- Import Status: ✅ All components imported\n\n")
        else:
            report_content.append("- MCP Instance: ❌ Not Available\n")
            report_content.append("- Initialization: ❌ Failed\n")
            report_content.append("- Import Status: ❌ Import errors\n\n")
        
        # Capabilities Summary
        if "error" not in capabilities:
            report_content.append("## MCP Capabilities Summary\n")
            tools_count = len(capabilities.get("tools", {}))
            resources_count = len(capabilities.get("resources", {}))
            modules_count = len(capabilities.get("modules", {}))
            
            report_content.append(f"- **Total Tools**: {tools_count}\n")
            report_content.append(f"- **Total Resources**: {resources_count}\n")
            report_content.append(f"- **Loaded Modules**: {modules_count}\n")
            report_content.append(f"- **Server Name**: {capabilities.get('name', 'Unknown')}\n")
            report_content.append(f"- **Server Version**: {capabilities.get('version', 'Unknown')}\n\n")
        else:
            report_content.append("## MCP Capabilities Summary\n")
            report_content.append(f"- **Error**: {capabilities['error']}\n\n")
        
        # Tools Analysis
        if "error" not in tools_analysis:
            report_content.append("## Tools Analysis\n")
            report_content.append(f"- **Total Tools**: {tools_analysis['total_tools']}\n")
            report_content.append(f"- **Categories**: {tools_analysis['categories_count']}\n")
            report_content.append(f"- **Modules with Tools**: {tools_analysis['modules_with_tools']}\n\n")
            
            # Tools by category
            report_content.append("### Tools by Category\n")
            for category, tools in tools_analysis.get("tools_by_category", {}).items():
                report_content.append(f"- **{category}** ({len(tools)} tools):\n")
                for tool in tools[:5]:  # Show first 5 tools per category
                    report_content.append(f"  - {tool['name']}: {tool['description']}\n")
                if len(tools) > 5:
                    report_content.append(f"  - ... and {len(tools) - 5} more\n")
                report_content.append("\n")
        else:
            report_content.append("## Tools Analysis\n")
            report_content.append(f"- **Error**: {tools_analysis['error']}\n\n")
        
        # Module Integration Status
        report_content.append("## Module Integration Status\n")
        successful_modules = 0
        failed_modules = 0
        
        for module_name, status in module_integration.items():
            if status.get("has_mcp_file") and status.get("has_register_tools"):
                report_content.append(f"- **{module_name}**: ✅ Complete MCP integration\n")
                successful_modules += 1
            elif status.get("has_mcp_file"):
                report_content.append(f"- **{module_name}**: ⚠️ MCP file exists but missing register_tools\n")
                failed_modules += 1
            else:
                report_content.append(f"- **{module_name}**: ❌ No MCP integration\n")
                failed_modules += 1
        
        report_content.append(f"\n**Summary**: {successful_modules} complete, {failed_modules} incomplete\n\n")
        
        # Functionality Tests
        report_content.append("## Functionality Tests\n")
        for test_name, result in functionality_tests.items():
            if isinstance(result, dict) and result.get("success"):
                report_content.append(f"- **{test_name}**: ✅ Passed\n")
            else:
                error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
                report_content.append(f"- **{test_name}**: ❌ Failed - {error_msg}\n")
        report_content.append("\n")
        
        # Server Status
        if "error" not in server_status:
            report_content.append("## Server Status\n")
            report_content.append(f"- **Status**: {server_status.get('status', 'Unknown')}\n")
            report_content.append(f"- **Uptime**: {server_status.get('uptime_formatted', 'Unknown')}\n")
            report_content.append(f"- **Request Count**: {server_status.get('request_count', 0)}\n")
            report_content.append(f"- **Error Count**: {server_status.get('error_count', 0)}\n")
            report_content.append(f"- **Error Rate**: {server_status.get('error_rate', 0):.2%}\n\n")
        
        # Write report
        report_file = mcp_output_dir / "mcp_operations_report.md"
        with open(report_file, 'w') as f:
            f.writelines(report_content)
        
        # Write detailed JSON report
        json_report = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "mcp_available": MCP_AVAILABLE,
            "capabilities": capabilities,
            "server_status": server_status,
            "tools_analysis": tools_analysis,
            "module_integration": module_integration,
            "functionality_tests": functionality_tests,
            "execution_time": time.time() - start_time
        }
        
        json_file = mcp_output_dir / "mcp_operations_report.json"
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        log_step_success(logger, f"MCP operations completed successfully")
        logger.info(f"📄 Markdown report: {report_file}")
        logger.info(f"📊 JSON report: {json_file}")
        
        # Log summary statistics
        if "error" not in tools_analysis:
            logger.info(f"🔧 Total MCP tools available: {tools_analysis['total_tools']}")
            logger.info(f"📦 Modules with MCP integration: {tools_analysis['modules_with_tools']}")
        
        return True
        
    except Exception as e:
        log_step_error(logger, f"MCP operations failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def main(parsed_args: argparse.Namespace):
    """Main function for MCP operations."""
    
    # --- Robust Path Handling ---
    # Determine project root (parent of src/)
    project_root = Path(__file__).resolve().parent.parent
    cwd = Path.cwd()
    
    # Defensive conversion and resolution of paths
    target_dir = parsed_args.target_dir
    output_dir = parsed_args.output_dir
    
    # If not absolute, resolve relative to project root
    if not isinstance(target_dir, Path):
        target_dir = Path(target_dir)
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if not target_dir.is_absolute():
        target_dir = (project_root / target_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log all argument values and their types
    logger.info("--- MCP Step Argument Debugging ---")
    logger.info(f"Working directory: {cwd}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Resolved target_dir: {target_dir} (Type: {type(target_dir).__name__})")
    logger.info(f"Resolved output_dir: {output_dir} (Type: {type(output_dir).__name__})")
    logger.info(f"Verbose: {parsed_args.verbose}")
    logger.info("-------------------------------")
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("7_mcp.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Model Context Protocol operations')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run MCP operations
    success = run_mcp_operations(
        target_dir=target_dir,
        output_dir=output_dir,
        verbose=parsed_args.verbose
    )
    
    if success:
        log_step_success(logger, "MCP operations completed successfully")
        return 0
    else:
        log_step_error(logger, "MCP operations failed")
        return 1

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("7_mcp")
    else:
        # Fallback argument parsing
        parser = argparse.ArgumentParser(description="Model Context Protocol operations")
        parser.add_argument("--target-dir", type=Path, required=True,
                          help="Target directory containing GNN files")
        parser.add_argument("--output-dir", type=Path, required=True,
                          help="Output directory for generated artifacts")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose output")
        parsed_args = parser.parse_args()
    
    exit_code = main(parsed_args)
    sys.exit(exit_code) 