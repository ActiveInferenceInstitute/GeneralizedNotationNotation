#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 7: MCP Operations

This script handles Model Context Protocol (MCP) specific operations, such as:
- MCP validation (checking for core files and module integrations).
- Reporting on available MCP methods within functional modules.

Usage:
    python 7_mcp.py [options]
    
Options:
    Same as main.py (though many may not be relevant for MCP)
"""

import os
import sys
from pathlib import Path
import argparse
import ast # For parsing Python files to find MCP methods
import datetime # For report timestamp
import logging # Import logging
import inspect # For inspecting function signatures
import json # For JSON operations

# Pre-define MCPSDKNotFoundError to satisfy linter in case of import issues in the try-except block
MCPSDKNotFoundError = Exception

# Attempt to import the main MCP instance and initializer
# This relies on main.py having set up sys.path correctly for the venv
try:
    from mcp import mcp_instance, initialize as initialize_mcp_system
except ImportError:
    # Fallback for standalone execution or if src is not directly in path
    # Add parent of current script's parent to sys.path (i.e., project root)
    # to allow from mcp import ...
    # This is a common pattern if 7_mcp.py is in src/ and mcp.py is in src/mcp/
    # and we want to treat 'src' as a package root for 'from mcp import ...'
    # For 'from src.mcp import ...' the project root (parent of src) must be in sys.path
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent # Assuming script is in src/
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Now try importing with 'src.' prefix
    try:
        from src.mcp import mcp_instance, initialize as initialize_mcp_system
        logging.getLogger(__name__).info("Successfully imported project's internal MCP components (from src.mcp) after path adjustment. This is the primary MCP implementation for this project.")
    except ImportError as e:
        logging.getLogger(__name__).error(f"Failed to import MCP components even after path adjustment: {e}. Report may be incomplete for registered tool descriptions.")
        mcp_instance = None
        initialize_mcp_system = None

# Logger for this module
logger = logging.getLogger(__name__)

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

def _get_mcp_methods(mcp_file_path: Path, verbose: bool = False):
    """
    Parses a Python file and extracts top-level function definitions (MCP methods).
    Returns a list of tuples: (method_name, arguments, docstring_preview).
    """
    methods = []
    if not mcp_file_path.exists() or not mcp_file_path.is_file():
        return methods
    
    # This function's verbose prints become logger.debug messages.
    # They will only show if this module's logger is set to DEBUG.
    logger.debug(f"      🐍 Parsing for MCP methods in: {mcp_file_path}")
        
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
                logger.debug(f"        Found method: {method_name}({', '.join(args)}) - \"{docstring_preview}\"...")
    except Exception as e:
        logger.warning(f"      ⚠️ Error parsing {mcp_file_path} for methods: {e}") # Was print if verbose
    return methods

def process_mcp_operations(src_root_dir_str: str, mcp_base_dir_str: str, output_dir_str: str, verbose: bool = False):
    """
    Performs checks related to the project's MCP integration:
    - Verifies that expected functional modules have their own 'mcp.py' for integration.
    - Checks for core MCP server files.
    - Lists available methods in each module's mcp.py using registered MCP info first.
    - Generates a comprehensive report.
    """
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
    
    # Control verbosity of the core 'mcp' module (from src/mcp or mcp package)
    core_mcp_logger = logging.getLogger("mcp") # Assuming the core mcp logger is named "mcp"
    # If it's imported as 'from src.mcp ...', its logger might be "src.mcp"
    # It might be safer to try to get both if unsure, or rely on the one that was successfully imported.
    # For now, let's assume "mcp" or that initialize_mcp_system handles its own verbosity based on root.
    # A more robust way is if initialize_mcp_system took a verbosity/logger level argument.
    # Given the current setup, we adjust it here:
    if verbose: # verbose is from the pipeline's args.verbose
        core_mcp_logger.setLevel(logging.INFO) # Allow core mcp to show its INFO messages
    else:
        core_mcp_logger.setLevel(logging.WARNING) # Suppress core mcp INFO messages by default

    overall_step_success = True # Initialize overall success for this step

    # Initialize the MCP system to discover all tools and their registered descriptions
    if initialize_mcp_system and mcp_instance:
        logger.info("    🚀 Initializing MCP system to load registered tool descriptions...")
        mcp_init_successful_flag = False # Flag for MCP specific initialization
        try:
            # Capture all three return values
            _, sdk_status_from_init, all_modules_loaded = initialize_mcp_system() 
            mcp_init_successful_flag = sdk_status_from_init and all_modules_loaded

            if sdk_status_from_init and all_modules_loaded:
                logger.info("    ✅ MCP system initialization process completed successfully (SDK status OK, all modules loaded).")
            elif sdk_status_from_init and not all_modules_loaded:
                logger.warning("    ⚠️ MCP system initialization completed (SDK status OK), BUT ONE OR MORE MCP MODULES FAILED TO LOAD. Check 'mcp' logs for details.")
                overall_step_success = False # Mark overall step as problematic due to module load failure
            else: # sdk_status_from_init is False or other unhandled init issues
                logger.warning("    ⚠️ MCP system initialization completed, but SDK status indicates issues (see mcp logger for details). Module loading may also be affected.")
                overall_step_success = False # Mark as problematic if SDK status itself is an issue

            # --- BEGIN NEW GLOBAL SUMMARY SECTION ---
            report_lines.append("\n## 🌐 Global Summary of Registered MCP Tools\n")
            if mcp_instance.tools: 
                report_lines.append("This section lists all tools currently registered with the MCP system, along with their defining module, arguments, and description.\n")
                
                sorted_tools = sorted(mcp_instance.tools.items(), key=lambda item: str(item[0]))
                
                for tool_name, tool_obj in sorted_tools:
                    tool_module = "N/A"
                    if hasattr(tool_obj.func, '__module__'):
                        tool_module = tool_obj.func.__module__
                    
                    args_str = "..."
                    try:
                        sig = inspect.signature(tool_obj.func)
                        args_list = [param.name for param in sig.parameters.values()]
                        args_str = f"({', '.join(args_list)})"
                    except (ValueError, TypeError):
                        logger.debug(f"Could not retrieve signature for tool {tool_name}")
                    
                    description = tool_obj.description if tool_obj.description else "No description provided."
                    schema_str = json.dumps(tool_obj.schema, indent=4) if hasattr(tool_obj, 'schema') and tool_obj.schema else "No schema provided."
                    
                    report_lines.append(f"- **Tool:** `{tool_name}`")
                    report_lines.append(f"  - **Defined in Module:** `{tool_module}`")
                    report_lines.append(f"  - **Arguments (from signature):** `{args_str}`")
                    report_lines.append(f"  - **Description:** \"{description}\"")
                    report_lines.append(f"  - **Schema:**")
                    report_lines.append(f"    ```json")
                    report_lines.append(f"    {schema_str.replace('\n', '\n    ')}") # Indent schema for markdown
                    report_lines.append(f"    ```")
                report_lines.append("\n") 
            else:
                report_lines.append("No MCP tools found registered in `mcp_instance.tools` after initialization.\n")
            # --- END NEW GLOBAL SUMMARY SECTION ---

        except MCPSDKNotFoundError as e: # Specific exception from mcp.initialize
            logger.error(f"    ❌ Error initializing MCP system (MCPSDKNotFoundError): {e}. Registered descriptions might be unavailable.", exc_info=True)
            overall_step_success = False
            # Add a placeholder line for the global summary in case of error
            report_lines.append("\n## 🌐 Global Summary of Registered MCP Tools\n")
            report_lines.append(f"MCP SDK Not Found Error during initialization. Tool summary unavailable. Details: {e}\n")
        except Exception as e:
            logger.error(f"    ❌ Error initializing MCP system: {e}. Registered descriptions might be unavailable.", exc_info=True)
            overall_step_success = False
            # Add a placeholder line for the global summary in case of error
            report_lines.append("\n## 🌐 Global Summary of Registered MCP Tools\n")
            report_lines.append("Failed to initialize MCP system. Tool summary unavailable.\n")
    else:
        logger.warning("    ⚠️ MCP instance or initializer not available. Registered tool descriptions will be missing. Falling back to AST parsing for docstrings.")
        # Add a placeholder line for the global summary if MCP system is not available
        # report_lines.append("\n## 🌐 Global Summary of Registered MCP Tools\n") # Already added if initialized at top
        # report_lines.append("MCP system not available. Tool summary unavailable.\n") # Already added if initialized at top
        # Ensure the section is present even if MCP init fails or not available.
        if not any("🌐 Global Summary of Registered MCP Tools" in line for line in report_lines):
            report_lines.append("\n## 🌐 Global Summary of Registered MCP Tools\n")
            report_lines.append("MCP system not available or initialization failed. Tool summary unavailable.\n")

    # verbose parameter here controls whether debug messages from this function are emitted.
    # The logger's level should already be set by main() based on args.verbose.
    
    logger.info(f"  🔎 Processing MCP integration checks and method discovery...") # General info
    logger.debug(f"    📖 MCP Core Directory: {Path(mcp_base_dir).resolve()}")
    logger.debug(f"    🏗️ Project Source Root for modules: {Path(src_root_dir).resolve()}")
        
    # 1. Check for core MCP server files in src/mcp/
    logger.debug("    Checking for core MCP files...")
    report_lines.append("\n## 🔬 Core MCP File Check\n") # Added heading for this section
    report_lines.append(f"This section verifies the presence of essential MCP files in the core directory: `{mcp_base_dir.resolve()}`\n")
    core_mcp_files = ["mcp.py", "meta_mcp.py", "cli.py", "server_stdio.py", "server_http.py"]
    all_core_found = True
    found_core_files_count = 0
    for fname in core_mcp_files:
        fpath = mcp_base_dir / fname
        if fpath.exists() and fpath.is_file():
            report_lines.append(f"- ✅ `{fname}`: Found ({fpath.stat().st_size} bytes)")
            logger.debug(f"      📖 Core MCP file found: {fpath} ({fpath.stat().st_size} bytes)")
            found_core_files_count += 1
        else:
            report_lines.append(f"- ❌ `{fname}`: **NOT FOUND**")
            all_core_found = False
            logger.warning(f"      ⚠️ Core MCP file NOT FOUND: {fpath}")
    report_lines.append(f"\n**Status:** {found_core_files_count}/{len(core_mcp_files)} core MCP files found. {'All core files seem present.' if all_core_found else 'One or more core MCP files are missing.'}\n")

    # 2. Verify MCP module integrations (mcp.py in each functional dir) and list methods
    report_lines.append("## 🧩 Functional Module MCP Integration & API Check\n")
    report_lines.append(f"Checking for `mcp.py` in these subdirectories of `{src_root_dir}`: {EXPECTED_MCP_MODULE_DIRS}\n")
    
    all_integrations_found = True
    missing_integrations = []
    found_integrations_count = 0

    # Prepare a set to keep track of methods already documented from mcp_instance
    # to avoid duplicates when AST parsing later (if we decide to mix)
    # Key: (module_name, method_name)
    registered_methods_added_for_this_module = set()

    for module_name in EXPECTED_MCP_MODULE_DIRS:
        # Initialize the set here for each module
        registered_methods_added_for_this_module = set()
        
        module_dir = src_root_dir / module_name
        mcp_integration_file = module_dir / "mcp.py"
        logger.debug(f"      Processing module: {module_name}")

        module_report_methods = [] # Store (name, args_str, description_str) tuples

        if not module_dir.is_dir():
            report_lines.append(f"### Module: `{module_name}` (at `{module_dir}`)")
            report_lines.append(f"- ❓ **Directory Status:** Directory does not exist. Cannot check for `mcp.py`.")
            logger.warning(f"        ❓ Module directory not found for '{module_name}': {module_dir}")
            report_lines.append("\n---\n")
            continue

        report_lines.append(f"### Module: `{module_name}` (at `{src_root_dir / module_name}`)")

        if mcp_integration_file.exists() and mcp_integration_file.is_file():
            report_lines.append(f"- ✅ **`mcp.py` Status:** Found ({mcp_integration_file.stat().st_size} bytes)")
            logger.debug(f"        ✅ MCP integration file found: {mcp_integration_file} ({mcp_integration_file.stat().st_size} bytes)")
            found_integrations_count += 1
            
            # 1. List methods from mcp_instance.tools for this module
            for tool_obj_name, tool_obj in mcp_instance.tools.items():
                if hasattr(tool_obj.func, '__module__') and tool_obj.func.__module__ == f"src.{module_name}.mcp":
                    try:
                        sig = inspect.signature(tool_obj.func)
                        args = [param.name for param in sig.parameters.values()]
                        args_str_sig = ', '.join(args)
                    except (ValueError, TypeError):
                        args_str_sig = "..." # Fallback if signature can't be read
                    
                    # Use the registered description and schema
                    desc_str = f" - *Description: \"{tool_obj.description}\"" if tool_obj.description else ""
                    schema_info = ""
                    if hasattr(tool_obj, 'schema') and tool_obj.schema:
                        schema_dump = json.dumps(tool_obj.schema, indent=2).replace('\n', '\n    ') # Indent for display
                        schema_info = f"\n    - Schema:\n      ```json\n      {schema_dump}\n      ```"
                    
                    # Combine method name, signature-derived args, description, and schema for the report line
                    # The original `module_report_methods` stored (name, args_str, description_str)
                    # We need to adjust this or how it's used.
                    # For now, let's make the description string richer for these tools.
                    full_tool_info_str = f"`def {tool_obj.name}({args_str_sig})`{desc_str}{schema_info}"
                    module_report_methods.append(full_tool_info_str) # Storing the full string
                    registered_methods_added_for_this_module.add(tool_obj.name)
                    logger.debug(f"          Found registered MCP tool: {tool_obj.name}({args_str_sig}) - Description: {tool_obj.description}")
            
            # 2. List other functions (like register_tools) using AST, avoiding duplicates
            ast_methods = _get_mcp_methods(mcp_integration_file, verbose=verbose)
            if ast_methods:
                for name, args, doc_preview in ast_methods:
                    if name not in registered_methods_added_for_this_module:
                        args_str = ', '.join(args)
                        # Use AST-parsed docstring for these non-MCP-tool functions
                        desc_str = f" - *\"{doc_preview}\"" if doc_preview else ""
                        # Format as a string consistent with MCP tools for sorting
                        ast_method_info_str = f"`def {name}({args_str})` (AST parsed){desc_str}"
                        module_report_methods.append(ast_method_info_str) 
                        logger.debug(f"          Found AST method (not a direct MCP tool or already listed): {name}({args_str}) - Docstring: {doc_preview}")

            if module_report_methods:
                report_lines.append("- **Exposed Methods & Tools:**")
                for method_info_str in sorted(module_report_methods): # Sort for consistent output
                    report_lines.append(f"  - {method_info_str}") # Directly use the formatted string
            else:
                report_lines.append("- **Exposed Methods & Tools:** No methods found or file could not be parsed effectively.")
        else:
            report_lines.append(f"- ❌ **`mcp.py` Status:** **NOT FOUND**")
            all_integrations_found = False
            missing_integrations.append(module_name)
            logger.warning(f"        ❌ MCP integration file NOT FOUND: {mcp_integration_file}")
        report_lines.append("\n---\n")
    
    report_lines.append("\n## 📊 Overall Module Integration Summary\n")
    report_lines.append(f"- **Modules Checked:** {len(EXPECTED_MCP_MODULE_DIRS)}")
    report_lines.append(f"- **`mcp.py` Integrations Found:** {found_integrations_count}/{len(EXPECTED_MCP_MODULE_DIRS)}")
    if all_integrations_found:
        report_lines.append("- **Status:** All expected functional modules appear to have an `mcp.py` integration file.")
    else:
        report_lines.append(f"- **Status:** Missing `mcp.py` integration files in: {', '.join(missing_integrations)}.")
    report_lines.append("  Please ensure each functional module that should be exposed via MCP has its own `mcp.py` following the project's MCP architecture.\n")

    # Write report
    try:
        with open(report_file_path, "w", encoding="utf-8") as f_report:
            f_report.write("\n".join(report_lines))
        report_size = report_file_path.stat().st_size
        logger.debug(f"  ✅ MCP integration and API report saved: {report_file_path.resolve()} ({report_size} bytes)")
    except Exception as e:
        logger.error(f"❌ Failed to write MCP integration report to {report_file_path}: {e}", exc_info=True)
        overall_step_success = False # Report writing failure also means overall failure
            
    return overall_step_success # Return the overall success status

def main(args):
    """Main function for the MCP operations step."""
    # Set this script's logger level based on pipeline's args.verbose
    # This is typically handled by main.py for child modules.
    # The process_mcp_operations function also sets levels for core_mcp_logger.
    # if args.verbose:
    #     logger.setLevel(logging.DEBUG)
    # else:
    #     logger.setLevel(logging.INFO)
    
    # When this script's logger is set to DEBUG (due to args.verbose),
    # its own logger.debug messages will show.
    # The verbosity of the *core* 'mcp' module (from src/mcp or package) is set 
    # inside process_mcp_operations based on args.verbose passed to it.

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
    # Basic configuration for running this script standalone
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Argument parsing for standalone execution
    parser = argparse.ArgumentParser(description="GNN Processing Pipeline - Step 7: Model Context Protocol (MCP) Integration Report Generator")
    parser.add_argument("--output-dir", default="../output",
                        help="Base directory to save the MCP report (default: ../output)")
    parser.add_argument("--mcp-core-dir-override", default=None,
                        help="Override the default path for the MCP core directory (e.g., src/mcp)")
    parser.add_argument("--src-root-override", default=None,
                        help="Override the default path for the project source root (e.g., src/)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output for this MCP step.")
    
    parsed_args = parser.parse_args()

    # Update log level if --verbose is used in standalone mode
    if parsed_args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        # Also set the mcp module's logger to DEBUG if verbose
        logging.getLogger("mcp").setLevel(logging.DEBUG) 
        logger.debug("Verbose logging enabled for standalone run of 7_mcp.py.")

    sys.exit(main(parsed_args)) 