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

def _get_relative_path_if_possible(absolute_path_obj: Path) -> str:
    """Returns a path string relative to project_root if set, otherwise absolute."""
    global _project_root_path_1_gnn
    if _project_root_path_1_gnn:
        try:
            return str(absolute_path_obj.relative_to(_project_root_path_1_gnn))
        except ValueError:
            return str(absolute_path_obj) # Not under project root
    return str(absolute_path_obj)

def process_gnn_folder(target_dir: str, output_dir: str, recursive: bool = False, verbose: bool = False):
    """
    Process the GNN folder:
    - Discover .md files.
    - Perform basic parsing for key GNN sections.
    - Log findings and simple statistics to a report file.
    """
    logger.info(f"Starting GNN file processing for directory: '{_get_relative_path_if_possible(Path(target_dir).resolve())}'")
    if recursive:
        logger.info("Recursive mode enabled: searching in subdirectories.")
    else:
        logger.info("Recursive mode disabled: searching in top-level directory only.")

    # Determine project root, assuming this script is in 'src/' subdirectory of project root
    try:
        script_file_path = Path(__file__).resolve()
        # project_root is the parent of the 'src' directory
        project_root = script_file_path.parent.parent
        logger.debug(f"Determined project root: {project_root}")
        global _project_root_path_1_gnn
        _project_root_path_1_gnn = project_root
    except Exception as e:
        logger.warning(f"Could not automatically determine project root. File paths in report might be absolute or less standardized: {e}")
        project_root = None # Fallback

    gnn_target_path = Path(target_dir)
    gnn_target_path_abs = gnn_target_path.resolve()

    if not gnn_target_path.is_dir():
        logger.warning(f"GNN target directory '{_get_relative_path_if_possible(gnn_target_path_abs)}' not found or not a directory. Skipping GNN processing for this target.")
        return True # Not a fatal error for this step, allows pipeline to continue

    # Ensure output directory for this step exists
    step_output_dir = Path(output_dir) / "gnn_processing_step"
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
    gnn_files = list(gnn_target_path.glob(file_pattern))

    if not gnn_files:
        logger.info(f"No .md files found in '{_get_relative_path_if_possible(gnn_target_path_abs)}' with pattern '{file_pattern}'.")
        try:
            with open(report_file_path, "w", encoding="utf-8") as f_report:
                f_report.write("# GNN File Discovery Report\n\n")
                f_report.write(f"No .md files found in `{_get_relative_path_if_possible(gnn_target_path_abs)}` using pattern `{file_pattern}`.\n")
            logger.info(f"Empty report saved to: {_get_relative_path_if_possible(report_file_path_abs)}")
        except IOError as e:
            logger.error(f"Failed to write empty report to {report_file_path_abs}: {e}")
            # Decide if this is fatal for the step; for now, non-fatal to allow pipeline progress
        return True

    logger.info(f"Found {len(gnn_files)} .md file(s) to process in '{_get_relative_path_if_possible(gnn_target_path_abs)}'.")

    for gnn_file_path_obj in gnn_files: # Renamed to avoid confusion with Path object
        # Determine path for reporting
        path_for_report_str = ""
        resolved_gnn_file_path = Path(gnn_file_path_obj).resolve()
        if project_root:
            try:
                path_for_report_str = str(resolved_gnn_file_path.relative_to(project_root))
            except ValueError: # If gnn_file is not under project_root for some reason
                path_for_report_str = str(resolved_gnn_file_path)
        else: # Fallback if project_root couldn't be determined
            path_for_report_str = str(resolved_gnn_file_path)
        
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
            # Regex for the "## ModelName" header line itself
            model_name_search_pattern = rf"^##\s*{re.escape(model_name_section_header_text)}\s*$" 
            model_name_str = "Not found"
            model_name_header_match = re.search(model_name_search_pattern, content, re.MULTILINE | re.IGNORECASE)

            if model_name_header_match:
                header_end_pos = model_name_header_match.end()
                # Regex to find the first non-empty, non-header line after the ## ModelName header.
                # It captures content that doesn't start with '#' and is not just whitespace.
                actual_model_name_match = re.search(r"^\s*([^\s#].*?)\s*$", content[header_end_pos:], re.MULTILINE)
                if actual_model_name_match:
                    extracted_name = actual_model_name_match.group(1).strip()
                    if extracted_name: # Ensure it's not an empty capture
                        model_name_str = f"Found: {extracted_name}"
                        file_summary["model_name"] = extracted_name # Store parsed name
                        found_model_name_count += 1
                        logger.debug(f"  Found {model_name_section_header_text}: '{extracted_name}' in {path_for_report_str}")
                    else: # Header found, but name line was effectively empty after stripping
                        model_name_str = "Found (header only, name line empty)"
                        file_summary["model_name"] = "(Header only, name empty)" # Store status
                        found_model_name_count +=1 # Count if header is present
                        logger.debug(f"  Found ## {model_name_section_header_text} header, but name line was empty in {path_for_report_str}")
                else: # Header found, but no suitable line for name found after it
                    model_name_str = "Found (header only, name line not found or suitable)"
                    file_summary["model_name"] = "(Header only, name not found)" # Store status
                    found_model_name_count += 1 # Count if header is present
                    logger.debug(f"  Found ## {model_name_section_header_text} header, but no subsequent content line found for name in {path_for_report_str}")
            else: # ## ModelName header itself not found
                logger.debug(f"  ## {model_name_section_header_text} section not found in {path_for_report_str}")
            file_summary["sections_found"].append(f"ModelName: {model_name_str}")


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
            f_report.write(f"Processed {len(gnn_files)} GNN file(s) from directory: `{_get_relative_path_if_possible(gnn_target_path_abs)}`\n")
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
        logger.info(f"GNN discovery report saved to: {_get_relative_path_if_possible(report_file_path_abs)}")
    except IOError as e:
        logger.error(f"Failed to write GNN discovery report to {report_file_path_abs}: {e}")
        return False # Report generation is a key output of this step.

    return True

def main(args):
    """Main function for the GNN file discovery and basic parsing step."""
    # The logger level for __name__ should be set by main.py based on args.verbose.
    # If this script is run standalone, the default logging level (WARNING) will apply
    # unless explicitly configured here or via environment variables.
    # For pipeline integration, main.py's logger setup for the root logger and its propagation
    # settings, along with this module's logger.setLevel in main.py's loop, will handle verbosity.
    global _project_root_path_1_gnn
    try:
        script_file_path = Path(__file__).resolve()
        _project_root_path_1_gnn = script_file_path.parent.parent
        logger.debug(f"[{Path(__file__).name}] Determined project root for relative path logging: {_project_root_path_1_gnn}")
    except Exception:
        _project_root_path_1_gnn = None # Ensure it's None if determination fails
        logger.debug(f"[{Path(__file__).name}] Could not determine project root for 1_gnn.py logging, paths may be absolute.")

    script_name = Path(__file__).name
    logger.info(f"Executing GNN processing step: {script_name}")
    
    # Resolve target_dir and output_dir to absolute paths for consistency.
    # main.py already resolves paths before passing them in `args`, 
    # but this makes 1_gnn.py's main() more robust if called directly with relative paths.
    resolved_target_dir = str(Path(args.target_dir).resolve())
    resolved_output_dir = str(Path(args.output_dir).resolve())
    
    logger.debug(f"Resolved target directory for GNN processing: {resolved_target_dir}")
    logger.debug(f"Resolved output directory for GNN processing: {resolved_output_dir}")

    success = process_gnn_folder(
        target_dir=resolved_target_dir, 
        output_dir=resolved_output_dir, 
        recursive=args.recursive if hasattr(args, 'recursive') else False, 
        verbose=args.verbose 
    )
        
    if not success:
        logger.error(f"{script_name} failed during GNN file processing.")
        return 1 # Indicate failure
        
    logger.info(f"{script_name} completed successfully.")
    return 0 # Indicate success

if __name__ == "__main__":
    # Basic configuration for running this script standalone
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Create a dummy args object or use a minimal one for standalone execution
    class DummyArgs:
        def __init__(self):
            self.target_dir = "gnn/examples"  # Default GNN source directory
            self.output_dir = "../output"     # Default main output directory
            self.recursive = True
            self.verbose = (log_level == logging.DEBUG)
            # Add any other attributes main() might expect from the pipeline's args object
            # e.g., self.skip_steps = "", self.only_steps = "", etc.

    dummy_args = DummyArgs()
    sys.exit(main(dummy_args)) 