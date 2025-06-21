#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 7: MCP (Model Context Protocol)

This script handles MCP operations and tool registrations.

Usage:
    python 7_mcp.py [options]
    (Typically called by main.py)
"""

import sys
from pathlib import Path
import argparse
import logging

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
    from mcp import mcp_instance, initialize as initialize_mcp_system
    logger.info("Successfully imported project's internal MCP components. This is the primary MCP implementation for this project.")
except ImportError as e:
    log_step_error(logger, f"Failed to import MCP components: {e}. Report may be incomplete for registered tool descriptions.")
    mcp_instance = None
    initialize_mcp_system = None

# Define the expected functional module directories that should have MCP integration
EXPECTED_MCP_MODULE_DIRS = [
    "export",
    "gnn",
    "gnn_type_checker",
    "ontology",
    "setup",
    "tests",
    "visualization",
    "llm"
]

def run_mcp_operations(target_dir: Path, output_dir: Path):
    """Run MCP operations and generate report."""
    log_step_start(logger, "Running MCP operations")
    
    # Use centralized output directory configuration
    mcp_output_dir = get_output_dir_for_script("7_mcp.py", output_dir)
    mcp_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize MCP system if available
        if initialize_mcp_system:
            initialize_mcp_system()
            log_step_success(logger, "MCP system initialized successfully")
        
        # Generate MCP report
        report_content = []
        report_content.append("# MCP (Model Context Protocol) Operations Report\n")
        
        # Check MCP instance availability
        if mcp_instance:
            report_content.append("## MCP System Status\n")
            report_content.append("- MCP Instance: ✅ Available\n")
            report_content.append("- Initialization: ✅ Successful\n\n")
            
            # Check for registered tools
            try:
                tools = getattr(mcp_instance, 'tools', {})
                report_content.append(f"## Registered Tools ({len(tools)})\n")
                for tool_name, tool_info in tools.items():
                    report_content.append(f"- **{tool_name}**: {tool_info.get('description', 'No description')}\n")
                report_content.append("\n")
            except Exception as e:
                report_content.append(f"## Tool Registration Error\n- {e}\n\n")
        else:
            report_content.append("## MCP System Status\n")
            report_content.append("- MCP Instance: ❌ Not Available\n")
            report_content.append("- Error: MCP components could not be imported\n\n")
        
        # Check expected MCP module directories
        report_content.append("## MCP Module Integration Status\n")
        for module_dir in EXPECTED_MCP_MODULE_DIRS:
            module_path = Path(__file__).parent / module_dir / "mcp.py"
            if module_path.exists():
                report_content.append(f"- **{module_dir}**: ✅ MCP integration found\n")
            else:
                report_content.append(f"- **{module_dir}**: ❌ No MCP integration\n")
        
        # Write report
        report_file = mcp_output_dir / "mcp_operations_report.md"
        with open(report_file, 'w') as f:
            f.writelines(report_content)
        
        log_step_success(logger, f"MCP operations report generated: {report_file}")
        return True
        
    except Exception as e:
        log_step_error(logger, f"MCP operations failed: {e}")
        return False

def main(parsed_args: argparse.Namespace):
    """Main function for MCP operations."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("7_mcp.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Model Context Protocol operations')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run MCP operations
    success = run_mcp_operations(
        target_dir=Path(parsed_args.target_dir),
        output_dir=Path(parsed_args.output_dir)
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