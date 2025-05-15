#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 5: Export

This script exports processed GNN data and generates comprehensive reports:
- Parses GNN files from the target directory.
- Exports GNN data to specified formats (e.g., JSON, XML, GEXF).
- Generates a summary report of all processed GNN files and export activities.

Usage:
    python 5_export.py [options]
    
Options:
    --target-dir DIR        Target directory for GNN files.
    --output-dir DIR        Directory to save outputs.
    --recursive / --no-recursive Recursively process directories.
    --formats FORMATS       Comma-separated list of formats to export (e.g., json,xml,dsl). Default: all.
    --verbose               Enable verbose output.
"""

import os
import sys
import datetime
import shutil
from pathlib import Path
import logging
import re
import argparse

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# --- End Logger Setup ---

# Attempt to import format exporters. Assumes 'src' is in PYTHONPATH or script is run correctly by main.py
try:
    from export.format_exporters import (
        _gnn_model_to_dict, export_to_json_gnn, export_to_xml_gnn,
        export_to_plaintext_summary, export_to_plaintext_dsl,
        export_to_gexf, export_to_graphml,
        export_to_json_adjacency_list, export_to_python_pickle
    )
    FORMAT_EXPORTERS_LOADED = True
except ImportError as e:
    logger.critical(f"CRITICAL: Failed to import format_exporters. Individual GNN export formats will be unavailable. Error: {e}")
    FORMAT_EXPORTERS_LOADED = False
    # Define dummy functions so the script doesn't crash if these are called later
    def _gnn_model_to_dict(*args, **kwargs): raise NotImplementedError("format_exporters not loaded")
    def export_to_json_gnn(*args, **kwargs): logger.error("export_to_json_gnn not available due to import error"); return False
    def export_to_xml_gnn(*args, **kwargs): logger.error("export_to_xml_gnn not available due to import error"); return False
    def export_to_plaintext_summary(*args, **kwargs): logger.error("export_to_plaintext_summary not available due to import error"); return False
    def export_to_plaintext_dsl(*args, **kwargs): logger.error("export_to_plaintext_dsl not available due to import error"); return False
    def export_to_gexf(*args, **kwargs): logger.error("export_to_gexf not available due to import error"); return False
    def export_to_graphml(*args, **kwargs): logger.error("export_to_graphml not available due to import error"); return False
    def export_to_json_adjacency_list(*args, **kwargs): logger.error("export_to_json_adjacency_list not available due to import error"); return False
    def export_to_python_pickle(*args, **kwargs): logger.error("export_to_python_pickle not available due to import error"); return False

# --- Available Export Formats Mapping ---
# This mapping should align with the functions available in format_exporters.py
AVAILABLE_EXPORT_FUNCTIONS = {
    "json": export_to_json_gnn,
    "xml": export_to_xml_gnn,
    "txt_summary": export_to_plaintext_summary,
    "dsl": export_to_plaintext_dsl,
    "gexf": export_to_gexf,
    "graphml": export_to_graphml,
    "json_adj": export_to_json_adjacency_list,
    "pkl": export_to_python_pickle
} if FORMAT_EXPORTERS_LOADED else {}

DEFAULT_EXPORT_FORMATS = ["json", "xml", "txt_summary", "dsl"] # A sensible default set

# --- Helper Functions (generate_summary_report, export_gnn_to_all_formats) ---
# These will be modified to use the formats argument and improve logging/structure
# (Content of these functions from the previous version will be integrated and refactored below)


def _get_relative_path_if_possible(absolute_path_obj: Path, project_root: Path | None) -> str:
    if project_root:
        try:
            return str(absolute_path_obj.relative_to(project_root))
        except ValueError:
            return str(absolute_path_obj)
    return str(absolute_path_obj)

def generate_export_summary_report(target_dir: str, output_dir: str, num_files_processed: int, num_files_exported: int, num_export_failures: int, recursive: bool = False, verbose: bool = False):
    """Generates a summary report specifically for the export step."""
    logger.info("📄 Generating GNN Export Step Summary Report...")
    output_path = Path(output_dir)
    summary_file = output_path / "gnn_exports" / "5_export_step_report.md"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    project_root = Path(__file__).resolve().parent.parent
    reported_target_dir = _get_relative_path_if_possible(Path(target_dir).resolve(), project_root)
    reported_output_dir = _get_relative_path_if_possible(output_path.resolve(), project_root)

    with open(summary_file, "w") as f:
        f.write("# 📤 GNN Export Step Summary\n\n")
        f.write(f"🗓️ Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## ⚙️ Configuration\n")
        f.write(f"- **Source Directory for GNN files:** `{reported_target_dir}`\n")
        f.write(f"- **Base Output for Exports:** `{Path(reported_output_dir) / 'gnn_exports'}`\n")
        f.write(f"- **Recursive Search:** {'✅ Enabled' if recursive else '❌ Disabled'}\n\n")
        f.write("## 📊 Export Statistics\n")
        f.write(f"- **GNN Files Found/Attempted:** {num_files_processed}\n")
        f.write(f"- **GNN Files with Successful Exports (all selected formats):** {num_files_exported - num_export_failures}\n")
        f.write(f"- **GNN Files with At Least One Export Failure:** {num_export_failures}\n")
    logger.info(f"📄 Export step summary report generated: {summary_file}")

def export_gnn_file_to_selected_formats(gnn_file_path: Path, base_output_dir: Path, formats_to_export: list[str], project_root_for_reporting: Path, verbose: bool = False):
    """
    Parses a single GNN file and exports it to selected formats.
    Outputs are saved in a subdirectory of base_output_dir/gnn_exports/, named after the GNN file.
    Returns True if all selected exports for this file were successful, False otherwise.
    """
    if not FORMAT_EXPORTERS_LOADED:
        logger.error(f"Skipping export for {_get_relative_path_if_possible(gnn_file_path, project_root_for_reporting)} as format exporters module is not loaded.")
        return False

    # Use a relative path for logging if possible
    gnn_file_display_name = _get_relative_path_if_possible(gnn_file_path, project_root_for_reporting)
    logger.info(f"  Processing exports for GNN file: {gnn_file_display_name}")

    try:
        # _gnn_model_to_dict expects a string path
        gnn_model_dict = _gnn_model_to_dict(str(gnn_file_path))
    except FileNotFoundError:
        logger.error(f"    ❌ File not found: {gnn_file_display_name}. Skipping export for this file.")
        return False
    except Exception as e:
        logger.error(f"    ❌ Error parsing GNN file {gnn_file_display_name}: {e}. Skipping export for this file.", exc_info=verbose)
        return False

    # Create a dedicated output subdirectory for this GNN file's exports
    # e.g., <pipeline_output_dir>/gnn_exports/<model_name>/
    file_specific_export_dir = base_output_dir / "gnn_exports" / gnn_file_path.stem
    try:
        file_specific_export_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"    📤 Ensured export subdirectory: {file_specific_export_dir}")
    except OSError as e:
        logger.error(f"    ❌ Failed to create export subdirectory {file_specific_export_dir}: {e}. Skipping exports for {gnn_file_display_name}.")
        return False

    success_count = 0
    failure_count = 0

    if not formats_to_export:
        logger.info(f"    ℹ️ No specific formats requested for export for {gnn_file_display_name}. Nothing to do.")
        return True # No failures if nothing was requested

    for fmt_key in formats_to_export:
        if fmt_key not in AVAILABLE_EXPORT_FUNCTIONS:
            logger.warning(f"    ⚠️ Unknown or unavailable export format '{fmt_key}' requested for {gnn_file_display_name}. Skipping this format.")
            failure_count += 1
            continue

        export_func = AVAILABLE_EXPORT_FUNCTIONS[fmt_key]

        file_extension = "gnn" if fmt_key == "dsl" else fmt_key
        if fmt_key == "txt_summary": file_extension = "txt"
        # For json_adj, it will be .json_adj if fmt_key is 'json_adj'

        output_file = file_specific_export_dir / f"{gnn_file_path.stem}.{file_extension}"
        try:
            logger.debug(f"      📤 Exporting to {fmt_key.upper()} -> {_get_relative_path_if_possible(output_file, project_root_for_reporting)}")
            export_func(gnn_model_dict, str(output_file)) # Pass the parsed dict
            logger.info(f"      ✅ Successfully exported to {output_file.name} (format: {fmt_key})")
            success_count += 1
        except Exception as e:
            logger.error(f"      ❌ Error exporting to {fmt_key.upper()} ({output_file.name}): {e}", exc_info=verbose)
            failure_count += 1
            # Optionally, clean up partially written file if necessary, though most write ops are atomic or overwrite.

    if failure_count > 0 and success_count > 0:
        logger.warning(f"    ⚠️ Partially completed exports for {gnn_file_display_name}: {success_count} succeeded, {failure_count} failed.")
    elif failure_count > 0 and success_count == 0:
        logger.error(f"    ❌ Failed all {len(formats_to_export)} requested export(s) for {gnn_file_display_name}.")
    elif success_count == len(formats_to_export) and success_count > 0:
        logger.info(f"    ✅ All {success_count} requested exports for {gnn_file_display_name} completed successfully.")
    # If formats_to_export was empty, this point isn't reached for this file due to early return.

    return failure_count == 0

def generate_overall_pipeline_summary_report(target_dir_abs: Path, output_dir_abs: Path, project_root: Path, recursive: bool, verbose: bool):
    """Generates the main gnn_processing_summary.md for the pipeline."""
    logger.info("📄 Generating Overall GNN Processing Summary Report...")
    summary_file = output_dir_abs / "gnn_processing_summary.md"

    md_files = list(target_dir_abs.glob("**/*.md" if recursive else "*.md"))
    logger.debug(f"  📊 Found {len(md_files)} .md files for overall summary in '{target_dir_abs}'.")

    with open(summary_file, "w") as f:
        f.write("# 📊 GNN Processing Summary\n\n")
        f.write(f"🗓️ Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## ⚙️ Processing Configuration\n\n")

def main(cmd_args=None):
    """Main function for the GNN export step (Step 5).

    Handles argument parsing if run standalone. Orchestrates the process of finding GNN files,
    exporting them to selected formats, and generating summary reports for the export step
    and the overall GNN processing (a basic file listing).

    Args:
        cmd_args (list[str] | None):
            A list of command-line arguments (strings) to be parsed, or None to use sys.argv[1:].
            If called from main.py, this will typically be None, and argparse will use sys.argv.
            Expected parsed arguments on the Namespace object include:
            target_dir (Path), output_dir (Path), recursive (bool), formats (str), verbose (bool).
    """
    script_file_path = Path(__file__).resolve()
    project_root = script_file_path.parent.parent

    default_target_dir = project_root / "src" / "gnn" / "examples"
    default_output_dir = project_root / "output"
    default_formats_str = ",".join(DEFAULT_EXPORT_FORMATS)

    parser = argparse.ArgumentParser(description="GNN Processing Pipeline - Step 5: Export GNNs & Generate Summary Report")
    parser.add_argument("--target-dir", type=Path, default=default_target_dir, help=f"Target directory for GNN files (default: {default_target_dir.relative_to(project_root)})" )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help=f"Base directory to save outputs (default: {default_output_dir.relative_to(project_root)})" )
    parser.add_argument("--recursive", action="store_true", help="Recursively process GNN files in the target directory.")
    parser.add_argument("--formats", type=str, default=default_formats_str, help=f"Comma-separated list of formats to export (e.g., json,xml,dsl). Available: {', '.join(AVAILABLE_EXPORT_FUNCTIONS.keys())}. Default: '{default_formats_str}'. Use 'all' for all available.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output for this script.")

    parsed_args = parser.parse_args(cmd_args if cmd_args else sys.argv[1:])

    # Configure logger for this script
    current_logger = logging.getLogger(__name__) 
    log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
    # Basic config if no handlers for THIS logger, to ensure output if standalone
    if not current_logger.hasHandlers() or not any(isinstance(h, logging.StreamHandler) for h in current_logger.handlers):
        logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stdout, force=True)
        current_logger.info(f"Configured basic logging for {__name__} at level {log_level}.")
    else:
        current_logger.setLevel(log_level)
    
    current_logger.info(f"▶️ Starting Step 5: Export ({script_file_path.name})")
    current_logger.debug(f"  Raw args received: {cmd_args if cmd_args else sys.argv[1:]}")
    current_logger.debug(f"  Parsed args: {parsed_args}")

    target_dir_abs = parsed_args.target_dir.resolve()
    output_dir_abs = parsed_args.output_dir.resolve()

    try:
        output_dir_abs.mkdir(parents=True, exist_ok=True)
        (output_dir_abs / "gnn_exports").mkdir(parents=True, exist_ok=True)
        current_logger.info(f"  Ensured output directory for exports: {output_dir_abs / 'gnn_exports'}")
    except OSError as e:
        current_logger.error(f"  ❌ Failed to create output directories: {e}. Aborting export step.")
        return 1

    formats_input_str = parsed_args.formats.lower()
    if "all" in formats_input_str or not formats_input_str.strip():
        selected_formats = list(AVAILABLE_EXPORT_FUNCTIONS.keys())
        current_logger.info(f"  Exporting to all available formats: {selected_formats}")
    else:
        requested_formats_list = [f.strip() for f in formats_input_str.split(',') if f.strip()]
        selected_formats = []
        for req_f in requested_formats_list:
            if req_f in AVAILABLE_EXPORT_FUNCTIONS:
                selected_formats.append(req_f)
            else:
                current_logger.warning(f"  ⚠️ Requested export format '{req_f}' is not available/supported. Skipping.")
        current_logger.info(f"  Selected formats for export: {selected_formats}")

    if not FORMAT_EXPORTERS_LOADED:
        current_logger.critical("CRITICAL: Cannot perform GNN exports because 'format_exporters' module failed to load.")
        generate_overall_pipeline_summary_report(target_dir_abs, output_dir_abs, project_root, parsed_args.recursive, parsed_args.verbose)
        return 1
 
    overall_export_status_code = 0 # Default to success
    num_files_processed = 0
    num_successful_file_exports = 0
    num_files_with_failures = 0

    if not selected_formats:
        current_logger.warning("No valid/available export formats selected. Skipping GNN model exports.")
    else:
        glob_pattern = "**/*.md" if parsed_args.recursive else "*.md"
        gnn_files_to_export = list(target_dir_abs.glob(glob_pattern))
        num_files_processed = len(gnn_files_to_export)

        if not gnn_files_to_export:
            current_logger.info(f"No GNN files (.md) found in '{target_dir_abs}' (pattern: '{glob_pattern}') to export.")
        else:
            current_logger.info(f"Found {len(gnn_files_to_export)} GNN file(s) to process for export from '{target_dir_abs}'.")
            for gnn_file in gnn_files_to_export:
                file_all_formats_succeeded = export_gnn_file_to_selected_formats(
                    gnn_file,
                    output_dir_abs, 
                    selected_formats,
                    project_root, 
                    parsed_args.verbose
                )
                if file_all_formats_succeeded:
                    num_successful_file_exports += 1
                else:
                    num_files_with_failures += 1
            
            current_logger.info(f"Export processing complete. Files with all formats successful: {num_successful_file_exports}, Files with at least one failure: {num_files_with_failures}")
            if num_files_with_failures > 0:
                overall_export_status_code = 1
    
    # Generate the export-specific summary report
    generate_export_summary_report(str(target_dir_abs), str(output_dir_abs), num_files_processed, num_successful_file_exports + num_files_with_failures, num_files_with_failures, parsed_args.recursive, parsed_args.verbose)

    # Generate the main pipeline summary report (gnn_processing_summary.md)
    generate_overall_pipeline_summary_report(target_dir_abs, output_dir_abs, project_root, parsed_args.recursive, parsed_args.verbose)
    
    current_logger.info(f"✅ Step 5: Export operations finished.")
    return overall_export_status_code

if __name__ == '__main__':
    exit_code = main() 
    sys.exit(exit_code)

# The generate_summary_report_v2 and export_all_gnn_files_v2 functions will be defined
# by integrating and refactoring the existing logic into them. 