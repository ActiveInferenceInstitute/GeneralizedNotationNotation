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
import re
import argparse
import json
from typing import Dict, Any, List

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
logger = setup_step_logging("5_export", verbose=False)

# Attempt to import format exporters. Assumes 'src' is in PYTHONPATH or script is run correctly by main.py
try:
    from export.format_exporters import (
        _gnn_model_to_dict,
        export_to_json_gnn,
        export_to_xml_gnn,
        export_to_plaintext_summary,
        export_to_plaintext_dsl,
        export_to_gexf,
        export_to_graphml,
        export_to_json_adjacency_list,
        export_to_python_pickle,
        HAS_NETWORKX
    )
    FORMAT_EXPORTERS_LOADED = True
except ImportError as e:
    log_step_error(logger, f"Failed to import format_exporters. Individual GNN export formats will be unavailable. Error: {e}")
    FORMAT_EXPORTERS_LOADED = False
    HAS_NETWORKX = False
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
    "pkl": export_to_python_pickle
} if FORMAT_EXPORTERS_LOADED else {}

if FORMAT_EXPORTERS_LOADED and HAS_NETWORKX:
    AVAILABLE_EXPORT_FUNCTIONS.update({
        "gexf": export_to_gexf,
        "graphml": export_to_graphml,
        "json_adj": export_to_json_adjacency_list,
    })
elif FORMAT_EXPORTERS_LOADED and not HAS_NETWORKX:
    log_step_warning(logger, "NetworkX-dependent export formats (gexf, graphml, json_adj) are disabled because networkx is not installed.")

DEFAULT_EXPORT_FORMATS = ["json", "xml", "txt_summary", "dsl"] # A sensible default set

def _get_relative_path_if_possible(absolute_path_obj: Path, project_root: Path | None) -> str:
    if project_root:
        try:
            return str(absolute_path_obj.relative_to(project_root))
        except ValueError:
            return str(absolute_path_obj)
    return str(absolute_path_obj)

def generate_export_summary_report(target_dir: str, output_dir: str, num_files_processed: int, num_files_exported: int, num_export_failures: int, recursive: bool = False, verbose: bool = False):
    """Generates a summary report specifically for the export step."""
    log_step_start(logger, "Generating GNN Export Step Summary Report")
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
    
    log_step_success(logger, f"Export step summary report generated: {summary_file}")

def export_gnn_file_to_selected_formats(gnn_file_path: Path, base_output_dir: Path, formats_to_export: list[str], project_root_for_reporting: Path, verbose: bool = False):
    """
    Parses a single GNN file and exports it to selected formats.
    Outputs are saved in a subdirectory of base_output_dir/gnn_exports/, named after the GNN file.
    Returns True if all selected exports for this file were successful, False otherwise.
    """
    if not FORMAT_EXPORTERS_LOADED:
        log_step_error(logger, f"Skipping export for {_get_relative_path_if_possible(gnn_file_path, project_root_for_reporting)} as format exporters module is not loaded")
        return False

    # Use a relative path for logging if possible
    gnn_file_display_name = _get_relative_path_if_possible(gnn_file_path, project_root_for_reporting)
    logger.info(f"Processing exports for GNN file: {gnn_file_display_name}")

    try:
        # _gnn_model_to_dict expects a string path
        gnn_model_dict = _gnn_model_to_dict(str(gnn_file_path))
    except FileNotFoundError:
        log_step_error(logger, f"File not found: {gnn_file_display_name}. Skipping export for this file.")
        return False
    except Exception as e:
        log_step_error(logger, f"Error parsing GNN file {gnn_file_display_name}: {e}")
        return False

    # Create a dedicated output subdirectory for this GNN file's exports
    # e.g., <pipeline_output_dir>/gnn_exports/<model_name>/
    file_specific_export_dir = base_output_dir / "gnn_exports" / gnn_file_path.stem
    try:
        file_specific_export_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured export subdirectory: {file_specific_export_dir}")
    except OSError as e:
        log_step_error(logger, f"Failed to create export subdirectory {file_specific_export_dir}: {e}. Skipping exports for {gnn_file_display_name}.")
        return False

    success_count = 0
    failure_count = 0

    if not formats_to_export:
        logger.info(f"No specific formats requested for export for {gnn_file_display_name}. Nothing to do.")
        return True # No failures if nothing was requested

    for fmt_key in formats_to_export:
        if fmt_key not in AVAILABLE_EXPORT_FUNCTIONS:
            log_step_warning(logger, f"Unknown or unavailable export format '{fmt_key}' requested for {gnn_file_display_name}. Skipping this format.")
            failure_count += 1
            continue

        export_func = AVAILABLE_EXPORT_FUNCTIONS[fmt_key]

        file_extension = "gnn" if fmt_key == "dsl" else fmt_key
        if fmt_key == "txt_summary": file_extension = "txt"

        output_file = file_specific_export_dir / f"{gnn_file_path.stem}.{file_extension}"
        try:
            logger.debug(f"Exporting to {fmt_key.upper()} -> {_get_relative_path_if_possible(output_file, project_root_for_reporting)}")
            export_func(gnn_model_dict, str(output_file))
            
            try:
                file_size_bytes = output_file.stat().st_size
                file_size_kb = file_size_bytes / 1024
                size_str = f", size: {file_size_kb:.2f} KB"
            except (OSError, ZeroDivisionError):
                size_str = ""
            
            logger.info(f"✅ Successfully exported to {fmt_key.upper()}{size_str}")
            success_count += 1
        
        except Exception as e:
            log_step_error(logger, f"Failed to export {gnn_file_display_name} to {fmt_key.upper()}: {e}")
            failure_count += 1

    total_requested = len(formats_to_export)
    logger.info(f"Export summary for {gnn_file_display_name}: {success_count}/{total_requested} formats successful")
    
    # Return True only if all requested formats were successfully exported
    return failure_count == 0

def generate_overall_pipeline_summary_report(target_dir_abs: Path, output_dir_abs: Path, project_root: Path, recursive: bool, verbose: bool):
    """Generates the main gnn_processing_summary.md for the pipeline."""
    logger.info("📄 Generating Overall GNN Processing Summary Report...")
    summary_file = output_dir_abs / "gnn_processing_summary.md"

    md_files = list(target_dir_abs.glob("**/*.md" if recursive else "*.md"))
    logger.debug(f"  📊 Found {len(md_files)} .md files for overall summary in '{target_dir_abs}'.")

    # Read pipeline execution summary if available
    pipeline_summary_file = output_dir_abs / "pipeline_execution_summary.json"
    pipeline_data = None
    if pipeline_summary_file.exists():
        try:
            with open(pipeline_summary_file, 'r') as f:
                pipeline_data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read pipeline execution summary: {e}")

    with open(summary_file, "w") as f:
        f.write("# 📊 GNN Processing Summary\n\n")
        f.write(f"🗓️ Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## ⚙️ Processing Configuration\n\n")
        if pipeline_data and "arguments" in pipeline_data:
            args = pipeline_data["arguments"]
            f.write(f"- **Target Directory**: `{args.get('target_dir', 'N/A')}`\n")
            f.write(f"- **Output Directory**: `{args.get('output_dir', 'N/A')}`\n")
            f.write(f"- **Recursive Processing**: {args.get('recursive', False)}\n")
            f.write(f"- **Verbose Mode**: {args.get('verbose', False)}\n")
            f.write(f"- **Strict Mode**: {args.get('strict', False)}\n")
            f.write(f"- **Resource Estimation**: {args.get('estimate_resources', False)}\n")
        else:
            f.write("- Configuration details not available\n")
        f.write("\n")

        f.write("## 📁 GNN Files Discovered\n\n")
        f.write(f"Found **{len(md_files)}** GNN files for processing:\n\n")
        for md_file in md_files:
            relative_path = md_file.relative_to(project_root) if md_file.is_relative_to(project_root) else md_file
            f.write(f"- `{relative_path}`\n")
        f.write("\n")

        f.write("## 🔄 Pipeline Execution Status\n\n")
        if pipeline_data and "steps" in pipeline_data:
            steps = pipeline_data["steps"]
            total_steps = len(steps)
            successful_steps = sum(1 for step in steps if step.get("status") == "SUCCESS")
            failed_steps = sum(1 for step in steps if step.get("status") == "FAILED")
            
            f.write(f"**Overall Status**: {successful_steps}/{total_steps} steps completed successfully\n\n")
            
            if pipeline_data.get("start_time") and pipeline_data.get("end_time"):
                start_time = datetime.datetime.fromisoformat(pipeline_data["start_time"])
                end_time = datetime.datetime.fromisoformat(pipeline_data["end_time"])
                total_duration = (end_time - start_time).total_seconds()
                f.write(f"**Total Execution Time**: {total_duration:.2f} seconds\n\n")
            
            f.write("### Step-by-Step Results\n\n")
            f.write("| Step | Script | Status | Duration (s) | Details |\n")
            f.write("|------|--------|--------|--------------|----------|\n")
            
            for step in steps:
                step_num = step.get("step_number", "?")
                script_name = step.get("script_name", "unknown")
                status = step.get("status", "UNKNOWN")
                duration = step.get("duration_seconds", 0)
                details = step.get("details", "")
                
                status_emoji = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⚠️"
                f.write(f"| {step_num} | `{script_name}` | {status_emoji} {status} | {duration:.3f} | {details} |\n")
            
            f.write("\n")
            
            if failed_steps > 0:
                f.write("### ❌ Failed Steps Details\n\n")
                for step in steps:
                    if step.get("status") == "FAILED":
                        f.write(f"**Step {step.get('step_number')}: {step.get('script_name')}**\n")
                        f.write(f"- Duration: {step.get('duration_seconds', 0):.3f}s\n")
                        if step.get("details"):
                            f.write(f"- Details: {step.get('details')}\n")
                        f.write("\n")
        else:
            f.write("Pipeline execution data not available.\n\n")

        f.write("## 📊 Output Summary\n\n")
        
        # Check for various output directories and files
        output_sections = [
            ("GNN Processing", "gnn_processing_step"),
            ("Type Checking", "gnn_type_check"),
            ("Exports", "gnn_exports"),
            ("Visualizations", "visualization"),
            ("Rendered Simulators", "gnn_rendered_simulators"),
            ("Test Reports", "test_reports"),
        ]
        
        for section_name, dir_name in output_sections:
            section_dir = output_dir_abs / dir_name
            if section_dir.exists():
                files = list(section_dir.rglob("*"))
                file_count = len([f for f in files if f.is_file()])
                f.write(f"- **{section_name}**: {file_count} files in `{dir_name}/`\n")
            else:
                f.write(f"- **{section_name}**: No output directory found\n")
        
        f.write("\n")

        f.write("## 🔍 Key Findings\n\n")
        
        # Analyze test results if available
        test_report_file = output_dir_abs / "test_reports" / "pytest_report.xml"
        if test_report_file.exists():
            try:
                # Simple XML parsing to extract test results
                with open(test_report_file, 'r') as tf:
                    test_content = tf.read()
                    # Extract basic test statistics from XML
                    import re
                    tests_match = re.search(r'tests="(\d+)"', test_content)
                    failures_match = re.search(r'failures="(\d+)"', test_content)
                    errors_match = re.search(r'errors="(\d+)"', test_content)
                    
                    if tests_match:
                        total_tests = int(tests_match.group(1))
                        failures = int(failures_match.group(1)) if failures_match else 0
                        errors = int(errors_match.group(1)) if errors_match else 0
                        passed = total_tests - failures - errors
                        
                        f.write(f"- **Test Results**: {passed}/{total_tests} tests passed")
                        if failures > 0 or errors > 0:
                            f.write(f" ({failures} failures, {errors} errors)")
                        f.write("\n")
            except Exception as e:
                f.write(f"- **Test Results**: Could not parse test report ({e})\n")
        
        # Check for GNN parsing issues
        gnn_discovery_file = output_dir_abs / "gnn_processing_step" / "1_gnn_discovery_report.md"
        if gnn_discovery_file.exists():
            try:
                with open(gnn_discovery_file, 'r') as gf:
                    gnn_content = gf.read()
                    if "StateSpaceBlock section not found" in gnn_content:
                        f.write("- **GNN Parsing**: Some StateSpaceBlock sections not detected\n")
                    if "Connections section not found" in gnn_content:
                        f.write("- **GNN Parsing**: Some Connections sections not detected\n")
            except Exception as e:
                logger.debug(f"Could not analyze GNN discovery report: {e}")
        
        f.write("\n")

        f.write("## 📋 Recommendations\n\n")
        
        if pipeline_data and "steps" in pipeline_data:
            failed_steps = [step for step in pipeline_data["steps"] if step.get("status") == "FAILED"]
            if failed_steps:
                f.write("### Immediate Actions Required\n\n")
                for step in failed_steps:
                    f.write(f"- **Fix Step {step.get('step_number')} ({step.get('script_name')})**: {step.get('details', 'Check logs for details')}\n")
                f.write("\n")
        
        f.write("### General Improvements\n\n")
        f.write("- Review and fix any test failures in the PyMDP converter components\n")
        f.write("- Investigate GNN section parsing issues (StateSpaceBlock/Connections detection)\n")
        f.write("- Consider running pipeline with `--verbose` flag for detailed debugging\n")
        f.write("- Check individual step logs in the output directory for specific error details\n")
        
        f.write("\n")
        f.write("---\n")
        f.write(f"*Report generated by GNN Processing Pipeline Step 5 (Export)*\n")

    logger.info(f"📄 Overall pipeline summary report saved to: {summary_file}")

def main(parsed_args: argparse.Namespace):
    """Main execution function for the export script."""
    # Setup logger based on verbosity
    if parsed_args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)
    else:
        import logging
        logger.setLevel(logging.INFO)

    # --- Start of Diagnostic Logging ---
    logger.info("--- 5_export.py execution started ---")
    logger.info(f"Received arguments: {parsed_args}")
    target_dir = parsed_args.target_dir.resolve()
    logger.info(f"Resolved target directory: {target_dir}")
    logger.info(f"Checking if target directory exists: {target_dir.exists()}")
    logger.info(f"Checking if target directory is a directory: {target_dir.is_dir()}")
    is_recursive = parsed_args.recursive
    logger.info(f"Recursive mode: {is_recursive}")
    # --- End of Diagnostic Logging ---

    logger.debug(f"Export script started with arguments: {parsed_args}")

    output_dir = parsed_args.output_dir.resolve()
    project_root = Path(__file__).resolve().parent.parent

    # Corrected glob_pattern to look for standard markdown files.
    glob_pattern = "**/*.md" if is_recursive else "*.md"
    logger.info(f"Using glob pattern: '{glob_pattern}' in directory '{target_dir}'")

    discovered_files = []
    if target_dir.is_dir():
        discovered_files = list(target_dir.glob(glob_pattern))
    
    logger.info(f"Found {len(discovered_files)} files using glob pattern.")
    if discovered_files:
        logger.debug("Discovered files list:")
        for f in discovered_files:
            logger.debug(f"  - {f}")

    if not discovered_files:
        logger.info(f"No GNN markdown files found in '{target_dir}' with pattern '{glob_pattern}'. Nothing to export.")
        # Generate an empty summary report
        generate_export_summary_report(
            str(target_dir), str(output_dir), 0, 0, 0,
            recursive=parsed_args.recursive, verbose=parsed_args.verbose
        )
        return 0
    
    logger.info(f"Starting GNN export for {len(discovered_files)} discovered GNN files...")

    # Determine which formats to export
    formats_to_export_str = parsed_args.formats.lower()
    if "all" in formats_to_export_str:
        formats_to_export = list(AVAILABLE_EXPORT_FUNCTIONS.keys())
    else:
        formats_to_export = [fmt.strip() for fmt in formats_to_export_str.split(',') if fmt.strip() in AVAILABLE_EXPORT_FUNCTIONS]

    if not formats_to_export:
        logger.warning("No valid or recognized export formats specified. Nothing will be exported.")
        return 0

    logger.info(f"Will attempt to export to the following formats: {', '.join(formats_to_export)}")
    
    successful_exports = 0
    failed_exports = 0

    for gnn_file_path in discovered_files:
        if gnn_file_path.is_file():
            # The export function handles its own parsing and error logging now
            was_successful = export_gnn_file_to_selected_formats(
                gnn_file_path,
                output_dir,
                formats_to_export,
                project_root_for_reporting=project_root,
                verbose=parsed_args.verbose
            )
            if was_successful:
                successful_exports += 1
            else:
                failed_exports += 1
        else:
            logger.warning(f"  ⚠️ Path listed during discovery is not a file: {gnn_file_path}. Skipping.")
            failed_exports += 1
    
    # Generate summary report
    generate_export_summary_report(
        str(target_dir),
        str(output_dir),
        num_files_processed=len(discovered_files),
        num_files_exported=successful_exports,
        num_export_failures=failed_exports,
        recursive=parsed_args.recursive,
        verbose=parsed_args.verbose
    )

    logger.info(f"✅ GNN export step finished. {successful_exports} files fully exported, {failed_exports} had issues.")
    
    if failed_exports > 0:
        return 1  # Indicate partial or total failure
    return 0

if __name__ == '__main__':
    # Determine project root for default paths
    script_file_path = Path(__file__).resolve()
    project_root_for_defaults = script_file_path.parent.parent # src/ -> project_root

    default_target_dir = project_root_for_defaults / "src" / "gnn" / "examples"
    default_output_dir = project_root_for_defaults / "output"
    default_formats_str = ",".join(DEFAULT_EXPORT_FORMATS)

    parser = argparse.ArgumentParser(description="GNN Processing Pipeline - Step 5: Export GNNs & Generate Summary Report (Standalone)")
    parser.add_argument("--target-dir", type=Path, default=default_target_dir, help=f"Target directory for GNN files (default: {default_target_dir.relative_to(project_root_for_defaults) if default_target_dir.is_relative_to(project_root_for_defaults) else default_target_dir})" )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help=f"Base directory to save outputs (default: {default_output_dir.relative_to(project_root_for_defaults) if default_output_dir.is_relative_to(project_root_for_defaults) else default_output_dir})" )
    # Adjusted recursive default to False for standalone, True is often by main.py unless specified
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False, help="Recursively process GNN files in the target directory.")
    parser.add_argument("--formats", type=str, default=default_formats_str, help=f"Comma-separated list of formats to export (e.g., json,xml,dsl). Available: {', '.join(AVAILABLE_EXPORT_FUNCTIONS.keys()) if FORMAT_EXPORTERS_LOADED else 'Error loading formats'}. Default: '{default_formats_str}'. Use 'all' for all available.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Enable verbose output for this script.") # Default False for standalone

    cli_args = parser.parse_args() # Parses from sys.argv

    # Update logger for standalone execution
    if cli_args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)

    exit_code = main(cli_args) 
    sys.exit(exit_code)

# The generate_summary_report_v2 and export_all_gnn_files_v2 functions will be defined
# by integrating and refactoring the existing logic into them. 