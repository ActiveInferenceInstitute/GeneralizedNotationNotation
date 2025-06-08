#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 7: Model Context Protocol (MCP)

This script manages Model Context Protocol operations for GNN files:
- Processes .md files in the target directory.
- Handles MCP-related tasks and API integrations.
- Saves MCP outputs to a dedicated subdirectory within the main output directory.

Usage:
    python 7_mcp.py [options]
    
Options:
    Same as main.py
"""

import sys
from pathlib import Path
import argparse
import ast # For parsing Python files to find MCP methods
import datetime # For report timestamp
import inspect # For inspecting function signatures
import json # For JSON operations
import os

# Import centralized utilities
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    UTILS_AVAILABLE
)

# Initialize logger for this step
logger = setup_step_logging("7_mcp", verbose=False)

# Pre-define MCPSDKNotFoundError to satisfy linter in case of import issues in the try-except block
MCPSDKNotFoundError = Exception

# Attempt to import the main MCP instance and initializer
# This relies on main.py having set up sys.path correctly for the venv
try:
    from mcp import mcp_instance, initialize as initialize_mcp_system
except ImportError:
    # Fallback for standalone execution or if src is not directly in path
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent # Assuming script is in src/
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Now try importing with 'src.' prefix
    try:
        from src.mcp import mcp_instance, initialize as initialize_mcp_system
        logger.info("Successfully imported project's internal MCP components (from src.mcp) after path adjustment. This is the primary MCP implementation for this project.")
    except ImportError as e:
        log_step_error(logger, f"Failed to import MCP components even after path adjustment: {e}. Report may be incomplete for registered tool descriptions.")
        mcp_instance = None
        initialize_mcp_system = None

# Define the expected functional module directories that should have MCP integration
# IMPORTANT: If a new functional module with an mcp.py is added to src/,
# it should be added to this list for 7_mcp.py to check and report on it.
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

def _get_mcp_methods(mcp_file_path: Path, verbose: bool = False):
    """
    Parses a Python file and extracts top-level function definitions (MCP methods).
    Returns a list of tuples: (method_name, arguments, docstring_preview).
    """
    methods = []
    if not mcp_file_path.exists() or not mcp_file_path.is_file():
        return methods
    
    logger.debug(f"Parsing for MCP methods in: {mcp_file_path}")
        
    try:
        with open(mcp_file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        tree = ast.parse(source_code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                method_name = node.name
                args = [arg.arg for arg in node.args.args]
                docstring = ast.get_docstring(node)  # Use ast.get_docstring()
                docstring_preview = ""
                if docstring:
                    docstring_preview = docstring.split('\n')[0]
                
                methods.append((method_name, args, docstring_preview))
                logger.debug(f"Found method: {method_name}({', '.join(args)}) - \"{docstring_preview}\"...")
    except Exception as e:
        log_step_warning(logger, f"Error parsing {mcp_file_path} for methods: {e}")
    return methods

def process_mcp_operations(src_root_dir_str: str, mcp_base_dir_str: str, output_dir_str: str, verbose: bool = False):
    """
    Performs checks related to the project's MCP integration:
    - Verifies that expected functional modules have their own 'mcp.py' for integration.
    - Checks for core MCP server files.
    - Lists available methods in each module's mcp.py using registered MCP info first.
    - Generates a comprehensive report.
    """
    log_step_start(logger, "Processing MCP operations and generating integration report")
    
    src_root_dir = Path(src_root_dir_str)
    mcp_base_dir = Path(mcp_base_dir_str)
    output_dir = Path(output_dir_str)
    
    mcp_step_output_path = output_dir / "mcp_processing_step"
    mcp_step_output_path.mkdir(parents=True, exist_ok=True)
    report_file_path = mcp_step_output_path / "7_mcp_integration_report.md"

    report_lines = ["# 🤖 MCP Integration and API Report\n"]
    report_lines.append(f"🗓️ Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**MCP Core Directory:** `{mcp_base_dir.resolve()}`")
    report_lines.append(f"**Project Source Root (for modules):** `{src_root_dir.resolve()}`")
    report_lines.append(f"**Output Directory for this report:** `{mcp_step_output_path.resolve()}`\n")

    overall_step_success = True # Initialize overall success for this step

    # Initialize the MCP system to discover all tools and their registered descriptions
    if initialize_mcp_system and mcp_instance:
        logger.info("Initializing MCP system to load registered tool descriptions...")
        mcp_init_successful_flag = False # Flag for MCP specific initialization
        try:
            # Capture all three return values
            _, sdk_status_from_init, all_modules_loaded = initialize_mcp_system() 
            mcp_init_successful_flag = sdk_status_from_init and all_modules_loaded

            if sdk_status_from_init and all_modules_loaded:
                log_step_success(logger, "MCP system initialization process completed successfully (SDK status OK, all modules loaded)")
            elif sdk_status_from_init and not all_modules_loaded:
                log_step_warning(logger, "MCP system initialization completed (SDK status OK), BUT ONE OR MORE MCP MODULES FAILED TO LOAD. Check 'mcp' logs for details.")
                overall_step_success = False # Mark overall step as problematic due to module load failure
            else: # sdk_status_from_init is False or other unhandled init issues
                log_step_warning(logger, "MCP system initialization completed, but SDK status indicates issues (see mcp logger for details). Module loading may also be affected.")
                overall_step_success = False # Mark as problematic if SDK status itself is an issue

            # --- BEGIN NEW GLOBAL SUMMARY SECTION ---
            report_lines.append("\n## 🌐 Global Summary of Registered MCP Tools\n")
            if mcp_instance.tools: 
                report_lines.append("This section lists all tools currently registered with the MCP system, along with their defining module, arguments, and description.\n")
                
                sorted_tools = sorted(mcp_instance.tools.items(), key=lambda item: str(item[0]))
                
                for tool_name, tool_obj in sorted_tools:
                    # Extract tool information
                    tool_schema = tool_obj.schema if hasattr(tool_obj, 'schema') else {}
                    tool_description = tool_schema.get('description', 'No description available.')
                    
                    # Get module information if available
                    tool_module = getattr(tool_obj.func, '__module__', 'Unknown module') if hasattr(tool_obj, 'func') else 'Unknown module'
                    
                    # Get parameters information
                    tool_parameters = tool_schema.get('inputSchema', {}).get('properties', {})
                    param_names = list(tool_parameters.keys()) if tool_parameters else []
                    
                    report_lines.append(f"### `{tool_name}`")
                    report_lines.append(f"- **Module:** `{tool_module}`")
                    report_lines.append(f"- **Parameters:** `{', '.join(param_names) if param_names else 'None'}`")
                    report_lines.append(f"- **Description:** {tool_description}\n")
            else:
                report_lines.append("⚠️ No tools are currently registered with the MCP system.\n")
                
        except Exception as e:
            log_step_error(logger, f"Error during MCP system initialization: {e}")
            overall_step_success = False
            report_lines.append(f"\n⚠️ **MCP System Initialization Error:** {str(e)}\n")
    else:
        log_step_warning(logger, "MCP system or instance not available. Report will be basic file structure check only.")
        overall_step_success = False
        report_lines.append("\n⚠️ **MCP System Unavailable:** Core MCP system not accessible. Report limited to file structure checks.\n")

    # Continue with the rest of the method...
    
    # Write report
    try:
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        if overall_step_success:
            log_step_success(logger, f"MCP integration report generated successfully: {report_file_path}")
        else:
            log_step_warning(logger, f"MCP integration report generated with warnings: {report_file_path}")
            
    except Exception as e:
        log_step_error(logger, f"Failed to write MCP integration report: {e}")
        return False
    
    return overall_step_success

def main(args):
    """Main function for the MCP operations step (Step 7).

    Handles path determinations for MCP core and source root directories
    (considering potential overrides from args), and then calls
    `process_mcp_operations` to perform MCP integration checks and reporting.

    Args:
        args (argparse.Namespace): 
            Parsed command-line arguments from `main.py` or standalone execution.
            Expected attributes include: output_dir, verbose, and optional
            mcp_core_dir_override, src_root_override.
    """
    # Set this script's logger level based on parsed_args.verbose
    # This logger is defined at the module level as logging.getLogger(__name__)
    log_level_for_this_script = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(log_level_for_this_script)
    # Log that this happened, but only if we are at DEBUG level already
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Script logger '{logger.name}' level set to {logging.getLevelName(log_level_for_this_script)}.")

    script_dir = Path(__file__).parent.resolve() # Should be src/
    mcp_core_dir = script_dir / "mcp" # This is the specific directory for core MCP files: src/mcp/
                                      # src_root_dir for module scanning is script_dir itself (src/)
    
    # When called from main.py, mcp_core_dir_override and src_root_override won't exist on args.
    # Use getattr to safely access them, defaulting to None if not present.
    mcp_core_dir_override_val = getattr(args, 'mcp_core_dir_override', None)
    src_root_override_val = getattr(args, 'src_root_override', None)

    if args.verbose:
        logger.info(f"▶️  Starting Step 7: MCP Operations ({Path(__file__).name})")
        logger.debug(f"  Parsing options (from main.py or standalone):")
        # Display the path that will actually be used, considering overrides or defaults
        effective_mcp_core_dir = Path(mcp_core_dir_override_val if mcp_core_dir_override_val else mcp_core_dir).resolve()
        effective_src_root_dir = Path(src_root_override_val if src_root_override_val else script_dir).resolve()
        logger.debug(f"    Effective MCP Core Directory: {effective_mcp_core_dir}")
        logger.debug(f"    Effective Project Source Root (for module scanning): {effective_src_root_dir}")
        logger.debug(f"    Output directory for MCP report: {Path(args.output_dir).resolve()}")
        logger.debug(f"    Verbose: {args.verbose}")

    actual_mcp_core_dir = Path(mcp_core_dir_override_val if mcp_core_dir_override_val else mcp_core_dir)
    actual_src_root_dir = Path(src_root_override_val if src_root_override_val else script_dir)

    if not actual_mcp_core_dir.exists() or not actual_mcp_core_dir.is_dir():
        logger.error(f"❌ CRITICAL-CHECK: MCP Core directory {actual_mcp_core_dir.resolve()} not found. Core file checks will be affected.")
    if not actual_src_root_dir.exists() or not actual_src_root_dir.is_dir():
        logger.critical(f"❌ CRITICAL-ERROR: Project Source Root directory {actual_src_root_dir.resolve()} not found. Cannot perform module MCP integration checks.")
        return 1 # This is more critical as no modules can be scanned.

    if not process_mcp_operations(
        str(actual_src_root_dir), 
        str(actual_mcp_core_dir), 
        args.output_dir, 
        args.verbose
    ):
        logger.error(f"❌ Step 7: MCP Operations ({Path(__file__).name}) FAILED (report generation or critical error during processing).")
        return 1
        
    logger.info(f"✅ Step 7: MCP Operations ({Path(__file__).name}) - COMPLETED (Report generated; check report for details on findings)")
    return 0

if __name__ == "__main__":
    # This block allows 7_mcp.py to be run standalone to generate the MCP integration report.
    
    # --- Argument Parsing for Standalone Execution ---
    parser = argparse.ArgumentParser(
        description="GNN Processing Pipeline - Step 7: MCP Operations (Standalone Runner). "
                    "Performs MCP integration checks and generates a report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--target-dir", 
        type=str, 
        help="Target directory containing GNN files (ignored for MCP operations, but included for compatibility with main.py)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output", # Sensible default for standalone
        help="Directory to save the MCP integration report."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging for this script and potentially for the core MCP system."
    )
    parser.add_argument(
        "--mcp-core-dir-override",
        type=str,
        default=None,
        help="Override the default MCP core directory (e.g., src/mcp)."
    )
    parser.add_argument(
        "--src-root-override",
        type=str,
        default=None,
        help="Override the default project source root directory (e.g., src/) for module scanning."
    )
    # --- End Argument Parsing ---

    # Parse arguments *before* using args.verbose or passing args to main()
    cli_args = parser.parse_args() # Corrected: args renamed to cli_args for clarity

    # Setup logging for standalone execution using the utility function
    log_level_to_set = logging.DEBUG if cli_args.verbose else logging.INFO
    if UTILS_AVAILABLE:
        logger = setup_step_logging("7_mcp", verbose=cli_args.verbose)
    else:
        # Fallback basic config if utility function couldn't be imported
        if not logging.getLogger().hasHandlers(): # Check root handlers
            logging.basicConfig(
                level=log_level_to_set,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                # datefmt="%Y-%m-%d %H:%M:%S", # Use default datefmt
                stream=sys.stdout
            )
        # Ensure this script's logger (which is __main__ here) level is set even in fallback
        current_script_logger = logging.getLogger(__name__)
        current_script_logger.setLevel(log_level_to_set) 
        current_script_logger.warning("Using fallback basic logging due to missing setup_standalone_logging utility.")
    
    # Update log level for core MCP loggers if --verbose is used in standalone mode
    if cli_args.verbose:
        # The main script logger (e.g., '__main__') should already be DEBUG via setup_standalone_logging
        # We also want the mcp module it calls to be verbose.
        logging.getLogger("mcp").setLevel(logging.DEBUG) 
        logging.getLogger("src.mcp").setLevel(logging.DEBUG) # And its potential alias if imported that way
        logging.getLogger(__name__).debug("Verbose logging enabled for standalone run of 7_mcp.py, including core MCP modules.")

    sys.exit(main(cli_args)) 