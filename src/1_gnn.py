#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 1: GNN File Discovery and Basic Parsing

This script handles initial GNN-specific operations, such as:
- Discovering .md GNN files in the target directory.
- Performing basic parsing for key GNN sections (ModelName, StateSpaceBlock, Connections).
- Generating a summary report of findings.

Usage:
    python 1_gnn.py [options]
    (Typically run as part of main.py pipeline)
    
Options:
    Same as main.py (verbose, target-dir, output-dir, recursive)
"""

import os
import sys
from pathlib import Path
import re # For parsing sections
import logging
import argparse # Added

# Attempt to import the new logging utility
try:
    from utils.logging_utils import setup_standalone_logging
except ImportError:
    # Fallback if utils is not in path (e.g. very direct execution or testing)
    # This assumes utils/ is a sibling to the script's directory if script is in src/
    # or that src/ is in PYTHONPATH.
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    try:
        from utils.logging_utils import setup_standalone_logging
    except ImportError:
        setup_standalone_logging = None # Define it as None if import fails
        # Log a warning using a temporary basic config if util is missing
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
        logging.getLogger(__name__).warning(
            "Could not import setup_standalone_logging from utils.logging_utils. Standalone logging might be basic."
        )

# --- Logger Setup ---
# Configure logger for this specific script.
# The main.py script will also configure a root logger, but this allows
# for script-specific naming if desired.
logger = logging.getLogger(__name__) # Use module name for logger
# Logger level will be set in main based on args.verbose
# --- End Logger Setup ---

# --- Global variable to store project_root if determined by main() or process_gnn_folder() ---
_project_root_path_1_gnn = None
# --- End Global ---

def _get_relative_path_if_possible(absolute_path_obj: Path, project_root: Path | None) -> str:
    """Returns a path string relative to project_root if provided and applicable, otherwise absolute."""
    if project_root:
        try:
            return str(absolute_path_obj.relative_to(project_root))
        except ValueError:
            return str(absolute_path_obj) # Not under project root
    return str(absolute_path_obj)

def process_gnn_folder(target_dir: Path, output_dir: Path, project_root: Path | None, recursive: bool = False, verbose: bool = False):
    """
    Process the GNN folder:
    - Discover .md files.
    - Perform basic parsing for key GNN sections.
    - Log findings and simple statistics to a report file.
    """
    logger.info(f"Starting GNN file processing for directory: '{_get_relative_path_if_possible(target_dir.resolve(), project_root)}'")
    if recursive:
        logger.info("Recursive mode enabled: searching in subdirectories.")
    else:
        logger.info("Recursive mode disabled: searching in top-level directory only.")

    # Determine project root, assuming this script is in 'src/' subdirectory of project root
    # This is now passed as an argument, but can be determined as a fallback if not provided.
    # if not project_root:
    #     try:
    #         script_file_path = Path(__file__).resolve()
    #         project_root_determined = script_file_path.parent.parent
    #         logger.debug(f"Determined project root internally: {project_root_determined}")
    #         # No longer setting global _project_root_path_1_gnn
    #     except Exception as e:
    #         logger.warning(f"Could not automatically determine project root. File paths in report might be absolute or less standardized: {e}")
    # else:
    #     logger.debug(f"Using provided project root: {project_root}")

    gnn_target_path_abs = target_dir.resolve()

    if not target_dir.is_dir():
        logger.warning(f"GNN target directory '{_get_relative_path_if_possible(gnn_target_path_abs, project_root)}' not found or not a directory. Skipping GNN processing for this target.")
        return 2 # Return 2 for warning/non-fatal issue

    # Ensure output directory for this step exists
    step_output_dir = output_dir / "gnn_processing_step"
    step_output_dir_abs = step_output_dir.resolve()
    try:
        step_output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {step_output_dir_abs}")
    except OSError as e:
        logger.error(f"Failed to create output directory {step_output_dir_abs}: {e}")
        return False # Cannot proceed without output directory

    report_file_path = step_output_dir / "1_gnn_discovery_report.md"
    report_file_path_abs = report_file_path.resolve()

    processed_files_summary = []
    file_pattern = "**/*.md" if recursive else "*.md"
    
    # --- Counters for summary ---
    found_model_name_count = 0
    found_statespace_count = 0
    found_connections_count = 0
    files_with_errors_count = 0
    # --- End Counters ---

    logger.debug(f"Searching for GNN files matching pattern '{file_pattern}' in '{gnn_target_path_abs}'")
    gnn_files = list(target_dir.glob(file_pattern))

    if not gnn_files:
        logger.info(f"No .md files found in '{_get_relative_path_if_possible(gnn_target_path_abs, project_root)}' with pattern '{file_pattern}'.")
        try:
            with open(report_file_path, "w", encoding="utf-8") as f_report:
                f_report.write("# GNN File Discovery Report\n\n")
                f_report.write(f"No .md files found in `{_get_relative_path_if_possible(gnn_target_path_abs, project_root)}` using pattern `{file_pattern}`.\n")
            logger.info(f"Empty report saved to: {_get_relative_path_if_possible(report_file_path_abs, project_root)}")
        except IOError as e:
            logger.error(f"Failed to write empty report to {report_file_path_abs}: {e}")
            # Decide if this is fatal for the step; for now, non-fatal to allow pipeline progress
        return 0 # No files found is not an error for the script itself, just an outcome.

    logger.info(f"Found {len(gnn_files)} .md file(s) to process in '{_get_relative_path_if_possible(gnn_target_path_abs, project_root)}'.")

    for gnn_file_path_obj in gnn_files:
        resolved_gnn_file_path = gnn_file_path_obj.resolve() # gnn_file_path_obj is already a Path
        path_for_report_str = _get_relative_path_if_possible(resolved_gnn_file_path, project_root)
        
        logger.debug(f"Processing file: {path_for_report_str}")
        
        file_summary = {
            "file_name": resolved_gnn_file_path.name, # Use resolved path's name
            "path": path_for_report_str,
            "model_name": "Not found", # Added for specific storage
            "sections_found": [],
            "model_parameters": {}, # Added to store parsed ModelParameters
            "errors": []
        }
        try:
            with open(resolved_gnn_file_path, "r", encoding="utf-8") as f: # Use resolved path to open
                content = f.read()
            logger.debug(f"Successfully read content from {path_for_report_str}.")
            
            # Basic section parsing
            # ModelName parsing
            model_name_section_header_text = "ModelName"
            model_name_str = "Not found" # Default status message
            parsed_model_name = "Not found" # Actual parsed name

            # Regex to find the "## ModelName" header, case-insensitive for "ModelName"
            # It captures the header itself to know its end position.
            model_name_header_pattern = re.compile(rf"^##\s*{model_name_section_header_text}\s*$\r?", re.IGNORECASE | re.MULTILINE)
            model_name_header_match = model_name_header_pattern.search(content)

            if model_name_header_match:
                logger.debug(f"  Found '## {model_name_section_header_text}' header in {path_for_report_str}")
                found_model_name_count += 1 # Count if header is present
                
                # Determine the content region for the model name:
                # from end of its header to start of the next section or end of file.
                content_after_header = content[model_name_header_match.end():]
                next_section_header_match = re.search(r"^##\s+\\w+", content_after_header, re.MULTILINE)
                
                if next_section_header_match:
                    name_region_content = content_after_header[:next_section_header_match.start()]
                else:
                    name_region_content = content_after_header
                
                # Find the first suitable line in this region
                extracted_name_candidate = ""
                for line in name_region_content.splitlines():
                    stripped_line = line.strip()
                    if stripped_line and not stripped_line.startswith("#"):
                        extracted_name_candidate = stripped_line
                        break # Found the first potential name
                
                if extracted_name_candidate:
                    parsed_model_name = extracted_name_candidate
                    model_name_str = f"Found: {parsed_model_name}" # Status message for report
                    logger.debug(f"    Extracted {model_name_section_header_text}: '{parsed_model_name}' from {path_for_report_str}")
                else:
                    # Header was found, but no suitable non-comment/non-empty line followed in its section
                    parsed_model_name = "(Header found, but name line empty or only comments)"
                    model_name_str = "Found (header only, name line empty/commented)" # Status message
                    logger.debug(f"    '## {model_name_section_header_text}' header found, but no suitable name line in {path_for_report_str}")
            else:
                # '## ModelName' header itself was not found
                parsed_model_name = "Not found"
                model_name_str = "Not found" # Status message (already default)
                logger.debug(f"  '## {model_name_section_header_text}' section header not found in {path_for_report_str}")

            file_summary["model_name"] = parsed_model_name # Store the parsed name or status
            # The sections_found list in the report will use model_name_str for its verbose status.
            # We need to update how sections_found is populated for ModelName.
            # First, remove any old ModelName entry from sections_found if it exists from a previous iteration logic.
            file_summary["sections_found"] = [s for s in file_summary["sections_found"] if not s.startswith("ModelName:")]
            file_summary["sections_found"].insert(0, f"ModelName: {model_name_str}") # Insert at beginning

            # StateSpaceBlock parsing
            statespace_section_header_text = "StateSpaceBlock"
            # Match if line starts with "## StateSpaceBlock" (case insensitive for "StateSpaceBlock"), allowing only optional comment after.
            statespace_search_pattern = rf"^##\\s*{re.escape(statespace_section_header_text)}\\s*(?:#.*)?$" 
            statespace_match = re.search(statespace_search_pattern, content, re.MULTILINE | re.IGNORECASE)
            if statespace_match:
                file_summary["sections_found"].append("StateSpaceBlock: Found")
                logger.debug(f"  Found {statespace_section_header_text} section in {path_for_report_str}")
                found_statespace_count += 1
            else:
                file_summary["sections_found"].append("StateSpaceBlock: Not found")
                logger.debug(f"  {statespace_section_header_text} section not found in {path_for_report_str}")
                if verbose:
                    logger.debug(f"    Content snippet for {path_for_report_str} (up to 500 chars) where {statespace_section_header_text} was not found:\n{content[:500]}")

            # Connections parsing
            connections_section_header_text = "Connections"
            # Match if line starts with "## Connections" (case insensitive for "Connections"), allowing only optional comment after.
            connections_search_pattern = rf"^##\\s*{re.escape(connections_section_header_text)}\\s*(?:#.*)?$"
            connections_match = re.search(connections_search_pattern, content, re.MULTILINE | re.IGNORECASE)
            if connections_match:
                file_summary["sections_found"].append("Connections: Found")
                logger.debug(f"  Found {connections_section_header_text} section in {path_for_report_str}")
                found_connections_count += 1
            else:
                file_summary["sections_found"].append("Connections: Not found")
                logger.debug(f"  {connections_section_header_text} section not found in {path_for_report_str}")
                if verbose:
                    logger.debug(f"    Content snippet for {path_for_report_str} (up to 500 chars) where {connections_section_header_text} was not found:\n{content[:500]}")

            # ModelParameters parsing
            model_params_section_header_text = "ModelParameters"
            model_params_search_pattern = rf"^##\s*{re.escape(model_params_section_header_text)}\s*(?:#.*)?$"
            model_params_match = re.search(model_params_search_pattern, content, re.MULTILINE | re.IGNORECASE)
            if model_params_match:
                logger.debug(f"  Found {model_params_section_header_text} section in {path_for_report_str}")
                section_content_start = model_params_match.end()
                # Find the next section header or end of file
                next_section_match = re.search(r"^##\s*\w+", content[section_content_start:], re.MULTILINE)
                section_content_end = next_section_match.start() + section_content_start if next_section_match else len(content)
                
                params_content = content[section_content_start:section_content_end]
                parsed_params_count = 0
                for line in params_content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"): # Skip empty lines and comments
                        continue
                    
                    match = re.match(r"([\w_]+):\s*(\[.*?\])\s*(?:#.*)?", line) # Capture key and list-like value
                    if match:
                        key = match.group(1).strip()
                        value_str = match.group(2).strip()
                        try:
                            # Attempt to evaluate the string as a Python literal (e.g., "[1, 2]" -> [1, 2])
                            # This is safer than full eval() but still needs caution if input isn't controlled.
                            # For GNN, we assume parameters are simple lists of numbers.
                            import ast
                            value = ast.literal_eval(value_str)
                            if isinstance(value, list): # Ensure it's a list
                                file_summary["model_parameters"][key] = value
                                logger.debug(f"    Parsed ModelParameter: {key} = {value}")
                                parsed_params_count += 1
                            else:
                                logger.warning(f"    ModelParameter '{key}' value '{value_str}' did not evaluate to a list in {path_for_report_str}. Storing as string.")
                                file_summary["model_parameters"][key] = value_str # Store as string if not a list
                        except (ValueError, SyntaxError) as e:
                            logger.warning(f"    Could not parse ModelParameter value for '{key}' ('{value_str}') as list in {path_for_report_str}: {e}. Storing as string.")
                            file_summary["model_parameters"][key] = value_str # Store as string on error
                    elif ':' in line: # Fallback for lines that might not be list format but are key:value
                        key_part, value_part = line.split(":", 1)
                        key = key_part.strip()
                        value = value_part.split("#", 1)[0].strip() # Remove comments
                        file_summary["model_parameters"][key] = value # Store as string
                        logger.debug(f"    Parsed ModelParameter (as string): {key} = {value}")
                        parsed_params_count +=1

                if parsed_params_count > 0:
                    file_summary["sections_found"].append(f"ModelParameters: Found ({parsed_params_count} parameters parsed)")
                else:
                    file_summary["sections_found"].append("ModelParameters: Found (section present, but no parameters parsed)")
            else:
                file_summary["sections_found"].append("ModelParameters: Not found")
                logger.debug(f"  {model_params_section_header_text} section not found in {path_for_report_str}")

        except Exception as e:
            logger.error(f"Error processing file {path_for_report_str}: {e}", exc_info=verbose) # Show traceback if verbose
            file_summary["errors"].append(str(e))
            files_with_errors_count += 1
        
        processed_files_summary.append(file_summary)

    # Generate the report
    try:
        with open(report_file_path, "w", encoding="utf-8") as f_report:
            f_report.write("# GNN File Discovery Report\n\n") 
            f_report.write(f"Processed {len(gnn_files)} GNN file(s) from directory: `{_get_relative_path_if_possible(gnn_target_path_abs, project_root)}`\n")
            f_report.write(f"Search pattern used: `{file_pattern}`\n\n")

            # --- Add Overall Summary ---
            f_report.write("## Overall Summary\n\n")
            f_report.write(f"- GNN files processed: {len(gnn_files)}\n")
            f_report.write(f"- Files with ModelName found: {found_model_name_count}\n")
            f_report.write(f"- Files with StateSpaceBlock found: {found_statespace_count}\n")
            f_report.write(f"- Files with Connections section found: {found_connections_count}\n")
            f_report.write(f"- Files with processing errors: {files_with_errors_count}\n\n")
            f_report.write("---\n") 
            # --- End Overall Summary ---
            
            f_report.write("## Detailed File Analysis\n\n")

            for summary in processed_files_summary:
                f_report.write(f"### File: `{summary['path']}`\n\n") 
                f_report.write("#### Found Sections:\n") 
                if summary["sections_found"]:
                    for section_info in summary["sections_found"]:
                        f_report.write(f"- {section_info}\n")
                else:
                    f_report.write("- (No specific sections parsed or found)\n")
                
                if summary["errors"]:
                    f_report.write("\n#### Errors During Processing:\n") # Changed from H3 to H4 for consistency
                    for error in summary["errors"]:
                        f_report.write(f"- {error}\n")
                f_report.write("\n---\n")
        logger.info(f"GNN discovery report saved to: {_get_relative_path_if_possible(report_file_path_abs, project_root)}")
    except IOError as e:
        logger.error(f"Failed to write GNN discovery report to {report_file_path_abs}: {e}")
        return 1 # Failure to write report is an error for this step

    if files_with_errors_count > 0:
        logger.warning(f"{files_with_errors_count} file(s) encountered errors during processing.")
        return 2 # Success with warnings
    
    return 0 # Pure success

def main(parsed_args: argparse.Namespace):
    """Main function to handle argument parsing and call processing logic.

    Args:
        parsed_args (argparse.Namespace): Pre-parsed command-line arguments.
            Expected attributes include: target_dir, output_dir, recursive, verbose.
    """
    # Setup logging level for this script's logger based on verbosity.
    # If run standalone, setup_standalone_logging (called in __main__ block) would have already
    # configured the root logger and potentially this logger's level.
    # If run as part of a pipeline (main.py), the pipeline's logger is configured,
    # and this script's logger will inherit. We explicitly set this script's logger level
    # here to ensure it respects its own verbose flag if the pipeline's verbosity differs
    # or if setup_standalone_logging didn't target this specific logger name with setLevel.
    log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
    logger.setLevel(log_level)
    # Ensure handlers also respect this level if they were configured higher by a root setup.
    # This is particularly for the case where main.py sets a global INFO, but this script is --verbose.
    for handler in logging.getLogger().handlers: # Check root handlers
        if handler.level > log_level: # only if root handler is less verbose
            # This is tricky. We don't want to override main.py's console handler level usually.
            # setup_standalone_logging handles its own logger.setLevel(logger_name) call.
            # For now, let's assume main.py's verbosity setting for handlers is king if already set.
            # The script's own logger.setLevel(log_level) above is the main control for this script's messages.
            pass 

    logger.debug(f"Script logger '{logger.name}' level set to {logging.getLevelName(log_level)}.")

    # Determine project_root for use in process_gnn_folder
    # This assumes the script itself (1_gnn.py) is in a 'src' directory, 
    # and 'src' is a direct child of the project root.
    try:
        script_file_path = Path(__file__).resolve()
        current_project_root = script_file_path.parent.parent 
        logger.debug(f"Project root determined for path relativization: {current_project_root}")
    except Exception:
        current_project_root = None
        logger.warning("Could not determine project root from script location. Paths in report may be absolute.")

    # Convert string paths from argparse to Path objects if they aren't already
    # (argparse with type=Path should handle this, but defensive)
    target_dir_path = Path(parsed_args.target_dir)
    output_dir_path = Path(parsed_args.output_dir)

    # Ensure paths are absolute before passing to processing function
    # This is good practice, though process_gnn_folder also resolves them.
    # Argparse with type=Path and default values based on Path(__file__) should result in absolute paths
    # if defaults are used, or if user provides absolute paths.
    # If user provides relative paths, they are relative to CWD. We should resolve them.
    target_dir_abs = target_dir_path.resolve()
    output_dir_abs = output_dir_path.resolve()

    logger.info(f"GNN Step 1: Target directory: {target_dir_abs}")
    logger.info(f"GNN Step 1: Output directory: {output_dir_abs}")
    logger.info(f"GNN Step 1: Recursive: {parsed_args.recursive}")
    logger.info(f"GNN Step 1: Verbose: {parsed_args.verbose}")

    result_code = process_gnn_folder(
        target_dir=target_dir_abs, 
        output_dir=output_dir_abs, 
        project_root=current_project_root,
        recursive=parsed_args.recursive, 
        verbose=parsed_args.verbose
    )
    
    if result_code == 0:
        logger.info("Step 1_gnn completed successfully.")
    elif result_code == 2:
        logger.warning("Step 1_gnn completed with warnings.")
    else:
        logger.error("Step 1_gnn failed.")
        sys.exit(result_code)

if __name__ == "__main__":
    # When run as a script, create a simple parser for standalone execution
    parser = argparse.ArgumentParser(description="GNN Processing Pipeline - Step 1: GNN File Discovery and Basic Parsing (Standalone)")
    
    # Determine project root for defaults, assuming script is in src/
    script_file_path_for_defaults = Path(__file__).resolve()
    project_root_for_defaults = script_file_path_for_defaults.parent.parent

    default_target_dir = project_root_for_defaults / "src" / "gnn" / "examples"
    default_output_dir = project_root_for_defaults / "output"

    parser.add_argument("--target-dir", type=Path, default=default_target_dir, help="Target directory for GNN files.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Directory to save outputs.")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="Recursively search target directory.")
    # Changed default for --verbose to False for standalone, as main.py controls its default True for pipeline context
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose (DEBUG level) logging.")
    
    cli_args = parser.parse_args()

    # Setup logging for standalone execution using the utility function
    log_level_to_set = logging.DEBUG if cli_args.verbose else logging.INFO
    if setup_standalone_logging:
        # Pass __name__ so the utility can also set this script's specific logger level
        setup_standalone_logging(level=log_level_to_set, logger_name=__name__)
    else:
        # Fallback basic config if utility function couldn't be imported
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=log_level_to_set,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                stream=sys.stdout
            )
        # Ensure this script's logger level is set even in fallback
        logging.getLogger(__name__).setLevel(log_level_to_set)
        logging.getLogger(__name__).warning("Using fallback basic logging due to missing setup_standalone_logging utility.")

    # Call the script's main function, passing the parsed args.
    main(cli_args)

# --- Helper function to simulate main.py args for testing (if needed) ---
# This is how main.py (subprocess version) would effectively call this script:
# python 1_gnn.py --target-dir path/to/target --output-dir path/to/output --recursive --verbose

# Example for direct call testing (if you were to import and run main() from another script):
# class DummyArgs:
#     def __init__(self):
#         self.target_dir = Path("../src/gnn/examples").resolve() # Adjust path as needed for testing
#         self.output_dir = Path("../output").resolve()
#         self.recursive = True
#         self.verbose = True # Set to False to test non-verbose
# test_args = DummyArgs()
# # If testing standalone, the __main__ block handles logging setup.
# # If testing main() directly after importing, ensure logging is configured before calling main(test_args).
# # Example for direct main() call with manual logging setup (mimicking standalone for test):
# # if setup_standalone_logging:
# #    setup_standalone_logging(level=logging.DEBUG if test_args.verbose else logging.INFO, logger_name=__name__)
# # else:
# #    logging.basicConfig(level=logging.DEBUG if test_args.verbose else logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
# #    logging.getLogger(__name__).setLevel(logging.DEBUG if test_args.verbose else logging.INFO)
# main(test_args) 