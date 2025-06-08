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
import argparse # Added
from typing import TypedDict, List, Dict, Any # Added for FileSummaryType

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
logger = setup_step_logging("1_gnn", verbose=False)  # Will be updated based on args

# --- Global variable to store project_root if determined by main() or process_gnn_folder() ---
_project_root_path_1_gnn = None
# --- End Global ---

# --- Define TypedDict for file_summary structure ---
class FileSummaryType(TypedDict):
    file_name: str
    path: str
    model_name: str
    sections_found: List[str]
    model_parameters: Dict[str, Any] # Or more specific type if parameters are uniform
    errors: List[str]
# --- End TypedDict ---

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

    gnn_target_path_abs = target_dir.resolve()

    if not target_dir.is_dir():
        log_step_warning(logger, f"GNN target directory '{_get_relative_path_if_possible(gnn_target_path_abs, project_root)}' not found or not a directory. Skipping GNN processing for this target.")
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
        return 0 

    logger.info(f"Found {len(gnn_files)} .md file(s) to process in '{_get_relative_path_if_possible(gnn_target_path_abs, project_root)}'.")

    for gnn_file_path_obj in gnn_files:
        resolved_gnn_file_path = gnn_file_path_obj.resolve() 
        path_for_report_str = _get_relative_path_if_possible(resolved_gnn_file_path, project_root)
        
        logger.debug(f"Processing file: {path_for_report_str}")
        
        file_summary: FileSummaryType = {
            "file_name": resolved_gnn_file_path.name,
            "path": path_for_report_str,
            "model_name": "Not found",
            "sections_found": list(),
            "model_parameters": dict(),
            "errors": list()
        }
        try:
            with open(resolved_gnn_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"Successfully read content from {path_for_report_str}.")
            
            # ModelName parsing
            model_name_section_header_text = "ModelName"
            model_name_str = "Not found" 
            parsed_model_name = "Not found" 

            _model_name_regex_string: str = rf"^##\s*{re.escape(model_name_section_header_text)}\s*$\r?"
            model_name_header_pattern: re.Pattern[str] = re.compile(_model_name_regex_string, re.IGNORECASE | re.MULTILINE)
            model_name_header_match = model_name_header_pattern.search(content)

            if model_name_header_match:
                logger.debug(f"  Found '## {model_name_section_header_text}' header in {path_for_report_str}")
                found_model_name_count += 1
                
                content_after_header = content[model_name_header_match.end():]
                next_section_header_match = re.search(r"^##\s+\w+", content_after_header, re.MULTILINE)
                
                if next_section_header_match:
                    name_region_content = content_after_header[:next_section_header_match.start()]
                else:
                    name_region_content = content_after_header
                
                extracted_name_candidate = ""
                for line in name_region_content.splitlines():
                    stripped_line = line.strip()
                    if stripped_line and not stripped_line.startswith("#"):
                        extracted_name_candidate = stripped_line
                        break
                
                if extracted_name_candidate:
                    parsed_model_name = extracted_name_candidate
                    model_name_str = f"Found: {parsed_model_name}"
                    logger.debug(f"    Extracted {model_name_section_header_text}: '{parsed_model_name}' from {path_for_report_str}")
                else:
                    parsed_model_name = "(Header found, but name line empty or only comments)"
                    model_name_str = "Found (header only, name line empty/commented)"
                    logger.debug(f"    '## {model_name_section_header_text}' header found, but no suitable name line in {path_for_report_str}")
            else:
                parsed_model_name = "Not found"
                model_name_str = "Not found"
                logger.debug(f"  '## {model_name_section_header_text}' section header not found in {path_for_report_str}")

            file_summary["model_name"] = parsed_model_name
            file_summary["sections_found"] = [s for s in file_summary["sections_found"] if not s.startswith("ModelName:")]
            file_summary["sections_found"].insert(0, f"ModelName: {model_name_str}")

            # StateSpaceBlock parsing
            statespace_section_header_text = "StateSpaceBlock"
            statespace_search_pattern = rf"^##\s*{re.escape(statespace_section_header_text)}\s*(?:#.*)?$"
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
            connections_search_pattern = rf"^##\s*{re.escape(connections_section_header_text)}\s*(?:#.*)?$"
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
            parameters_section_header_text = "ModelParameters"
            parameters_search_pattern = rf"^##\s*{re.escape(parameters_section_header_text)}\s*(?:#.*)?$"
            parameters_match = re.search(parameters_search_pattern, content, re.MULTILINE | re.IGNORECASE)
            if parameters_match:
                logger.debug(f"  Found {parameters_section_header_text} section in {path_for_report_str}")
                
                content_after_header = content[parameters_match.end():]
                next_section_match = re.search(r"^##\s+\w+", content_after_header, re.MULTILINE)
                
                param_region = content_after_header
                if next_section_match:
                    param_region = content_after_header[:next_section_match.start()]
                
                parsed_params = {}
                # Regex to capture key: value pairs, ignoring comments
                param_line_pattern = re.compile(r"^\s*([\w_]+)\s*:\s*(.*?)\s*(?:#.*)?$")
                for line in param_region.splitlines():
                    match = param_line_pattern.match(line)
                    if match:
                        key, value = match.groups()
                        parsed_params[key] = value.strip()
                        logger.debug(f"    Parsed ModelParameter: {key} = {value.strip()}")
                file_summary["model_parameters"] = parsed_params
            
        except Exception as e:
            logger.error(f"Error processing file '{path_for_report_str}': {e}", exc_info=verbose)
            file_summary["errors"].append(str(e))
            files_with_errors_count += 1
        
        processed_files_summary.append(file_summary)

    # Now write the report
    try:
        with open(report_file_path, "w", encoding="utf-8") as f_report:
            f_report.write("# GNN File Discovery Report\n\n")
            f_report.write("## Summary\n\n")
            f_report.write(f"- Total .md files found: {len(gnn_files)}\n")
            f_report.write(f"- Files with 'ModelName' found: {found_model_name_count}\n")
            f_report.write(f"- Files with 'StateSpaceBlock' found: {found_statespace_count}\n")
            f_report.write(f"- Files with 'Connections' found: {found_connections_count}\n")
            f_report.write(f"- Files with processing errors: {files_with_errors_count}\n\n")
            f_report.write("## Detailed File Analysis\n\n")

            for summary in processed_files_summary:
                f_report.write(f"### `{summary['path']}`\n\n")
                f_report.write(f"- **ModelName**: {summary['model_name']}\n")
                
                # Reformat sections_found list for cleaner report output
                sections_output = []
                for section in summary['sections_found']:
                    if section.startswith("ModelName:"): continue # Skip, as it's already displayed
                    parts = section.split(":", 1)
                    if len(parts) == 2:
                        sections_output.append(f"- **{parts[0].strip()}**: {parts[1].strip()}")
                    else:
                        sections_output.append(f"- {section}")
                
                f_report.write("\n".join(sections_output))
                f_report.write("\n")
                
                if summary["model_parameters"]:
                    f_report.write("- **ModelParameters**:\n")
                    for key, val in summary["model_parameters"].items():
                        f_report.write(f"  - `{key}`: `{val}`\n")
                
                if summary["errors"]:
                    f_report.write("- **Errors**:\n")
                    for error in summary["errors"]:
                        f_report.write(f"  - `{error}`\n")
                f_report.write("\n---\n\n")
        
        logger.info(f"GNN discovery report saved to: {_get_relative_path_if_possible(report_file_path_abs, project_root)}")
    except IOError as e:
        logger.error(f"Failed to write GNN discovery report to {report_file_path_abs}: {e}")
        return False
    
    logger.info("Step 1_gnn completed successfully.")
    return True

def main(parsed_args: argparse.Namespace):
    """Main function to handle argument parsing and call processing logic.

    Args:
        parsed_args (argparse.Namespace): Pre-parsed command-line arguments.
            Expected attributes include: target_dir, output_dir, recursive, verbose.
    """
    # Update logger verbosity based on args
    if UTILS_AVAILABLE and hasattr(parsed_args, 'verbose') and parsed_args.verbose:
        from utils import PipelineLogger
        PipelineLogger.set_verbosity(True)
    
    log_step_start(logger, "Starting GNN file discovery and basic parsing")

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
    
    if result_code:
        log_step_success(logger, "GNN file discovery and basic parsing completed successfully")
    else:
        log_step_error(logger, "GNN file discovery and basic parsing failed")
        sys.exit(1)

if __name__ == "__main__":
    # Use enhanced argument parsing for standalone execution
    try:
        if UTILS_AVAILABLE:
            from utils import EnhancedArgumentParser
            cli_args = EnhancedArgumentParser.parse_step_arguments("1_gnn")
        else:
            # Fallback to basic parser
            parser = argparse.ArgumentParser(description="GNN Processing Pipeline - Step 1: GNN File Discovery and Basic Parsing (Standalone)")
            
            # Determine project root for defaults, assuming script is in src/
            script_file_path_for_defaults = Path(__file__).resolve()
            project_root_for_defaults = script_file_path_for_defaults.parent.parent

            default_target_dir = project_root_for_defaults / "src" / "gnn" / "examples"
            default_output_dir = project_root_for_defaults / "output"

            parser.add_argument("--target-dir", type=Path, default=default_target_dir, help="Target directory for GNN files.")
            parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Directory to save outputs.")
            parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="Recursively search target directory.")
            parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose (DEBUG level) logging.")
            
            cli_args = parser.parse_args()
    except Exception as e:
        logger.error(f"Failed to parse arguments: {e}")
        sys.exit(1)

    # Call the script's main function, passing the parsed args.
    main(cli_args) 