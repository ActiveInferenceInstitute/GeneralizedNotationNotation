#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 5: Export

This script exports processed GNN data and generates comprehensive reports:
- Collects results from previous pipeline steps
- Generates summary reports
- Exports data to specified formats

Usage:
    python 5_export.py [options]
    
Options:
    Same as main.py
"""

import os
import sys
import datetime
import shutil
from pathlib import Path
import logging # Import logging
import re
import argparse

# --- Logger Setup ---
# Ensure this is relative if 5_export.py is in src/ and format_exporters.py is in src/export/
# If both are treated as part of a package 'src', then it might be:
# from .export.format_exporters import _gnn_model_to_dict, export_to_json_gnn, export_to_xml_gnn, \\
#                                     export_to_plaintext_summary, export_to_plaintext_dsl, \\
#                                     export_to_gexf, export_to_graphml, \\
#                                     export_to_json_adjacency_list, export_to_python_pickle
# However, given the direct script execution model, a sys.path modification might be more robust
# if main.py is not handling this perfectly for child scripts.
# For now, let's assume direct import works due to main.py's sys.path manipulations or PYTHONPATH.

# Attempting a robust import strategy
try:
    # If run as part of the 'src' package (e.g., by main.py that has added 'src' to sys.path)
    from export.format_exporters import (
        _gnn_model_to_dict, export_to_json_gnn, export_to_xml_gnn,
        export_to_plaintext_summary, export_to_plaintext_dsl,
        export_to_gexf, export_to_graphml,
        export_to_json_adjacency_list, export_to_python_pickle
    )
except ImportError:
    # Fallback if run standalone and 'export' is a subdirectory
    # This requires 'src' to be the current working directory or in PYTHONPATH
    try:
        # This assumes that the script is run from the `src` directory or `src` is in `PYTHONPATH`
        # Or that main.py correctly modifies sys.path to make `export` a discoverable module
        current_dir = Path(__file__).parent.resolve()
        export_module_path = current_dir / "export"
        if str(export_module_path) not in sys.path:
            sys.path.insert(0, str(export_module_path)) # Add 'src/export' to path
        if str(current_dir) not in sys.path: # Add 'src' to path
             sys.path.insert(0, str(current_dir))

        from format_exporters import (
            _gnn_model_to_dict, export_to_json_gnn, export_to_xml_gnn,
            export_to_plaintext_summary, export_to_plaintext_dsl,
            export_to_gexf, export_to_graphml,
            export_to_json_adjacency_list, export_to_python_pickle
        )
    except ImportError as e:
        # If this still fails, it's a significant issue.
        # We'll let the logger in main handle the more detailed error reporting.
        logger = logging.getLogger(__name__) # Ensure logger is available for this message
        logger.critical(f"CRITICAL: Failed to import format_exporters.py. Individual GNN export formats will be unavailable. Error: {e}")
        # Define dummy functions so the script doesn't crash if these are called later,
        # although the critical log should ideally stop execution or be noted.
        def _gnn_model_to_dict(*args, **kwargs): raise NotImplementedError("format_exporters not loaded")
        def export_to_json_gnn(*args, **kwargs): logger.error("export_to_json_gnn not available"); pass
        def export_to_xml_gnn(*args, **kwargs): logger.error("export_to_xml_gnn not available"); pass
        def export_to_plaintext_summary(*args, **kwargs): logger.error("export_to_plaintext_summary not available"); pass
        def export_to_plaintext_dsl(*args, **kwargs): logger.error("export_to_plaintext_dsl not available"); pass
        def export_to_gexf(*args, **kwargs): logger.error("export_to_gexf not available"); pass
        def export_to_graphml(*args, **kwargs): logger.error("export_to_graphml not available"); pass
        def export_to_json_adjacency_list(*args, **kwargs): logger.error("export_to_json_adjacency_list not available"); pass
        def export_to_python_pickle(*args, **kwargs): logger.error("export_to_python_pickle not available"); pass


logger = logging.getLogger(__name__)
# --- End Logger Setup ---

def generate_summary_report(target_dir, output_dir, recursive=False, verbose=False):
    """Generate a summary report of all processed GNN files."""
    # This script's logger level is set in its main() or by main.py.
    # verbose flag here is passed from args and can be used for conditional logic if needed,
    # but primary logging control is via the logger's level.

    logger.info("📄 Generating GNN Processing Summary Report...") # Was print if verbose

    # Build paths
    output_path = Path(output_dir)
    target_path = Path(target_dir)
    summary_file = output_path / "gnn_processing_summary.md"
    
    # Determine project root for relative paths in report
    project_root = Path(__file__).resolve().parent.parent

    logger.debug(f"  🎯 Target Directory (abs): {target_path.resolve()}") 
    logger.debug(f"  📁 Output Directory (abs): {output_path.resolve()}") 
    logger.debug(f"  📝 Summary Report Path: {summary_file}") # Was print if verbose

    # Get type checking report path
    type_check_path = output_path / "gnn_type_check" / "type_check_report.md"
    type_check_exists = type_check_path.exists()
    type_check_size = type_check_path.stat().st_size if type_check_exists else 0
    logger.debug(f"  🔍 Type Check Report: {type_check_path} (Exists: {type_check_exists}, Size: {type_check_size} bytes)") # Was print if verbose
    
    # Get visualization directory and check for actual files
    viz_dir = output_path / "gnn_examples_visualization"
    viz_exists = viz_dir.exists() and any(item for item in viz_dir.glob('**/*') if item.is_file())
    logger.debug(f"  🎨 Visualization Directory: {viz_dir} (Exists & has actual files: {viz_exists})") # Was print if verbose
    
    # Count processed files
    md_files = list(target_path.glob("**/*.md" if recursive else "*.md"))
    logger.debug(f"  📊 Found {len(md_files)} .md files for summary in '{target_path}'.") # Was print if verbose
    
    # Generate summary report
    logger.debug(f"  ✍️ Writing summary report content to: {summary_file}") # Was print if verbose
    with open(summary_file, "w") as f:
        f.write("# 📊 GNN Processing Summary\n\n")
        f.write(f"🗓️ Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## ⚙️ Processing Configuration\n\n")
        try:
            reported_target_dir = target_path.resolve().relative_to(project_root)
        except ValueError:
            reported_target_dir = target_path.resolve() # Keep absolute if not under project_root
        f.write(f"- 🎯 **Target Directory:** `{reported_target_dir}`\n")
        
        try:
            reported_output_dir = output_path.resolve().relative_to(project_root)
        except ValueError:
            reported_output_dir = output_path.resolve() # Keep absolute
        f.write(f"- 📁 **Output Directory:** `{reported_output_dir}`\n")
        f.write(f"- 🔄 **Recursive Processing:** {'✅ Enabled' if recursive else '❌ Disabled'}\n")
        f.write(f"- 🔢 **Total GNN Files Processed:** {len(md_files)}\n\n")
        
        f.write("## 🗂️ Processed GNN Source Files\n\n")
        if md_files:
            for md_file in md_files[:20]:  # Limit to first 20
                # md_file is an absolute path from target_path.glob()
                try:
                    relative_md_path = md_file.resolve().relative_to(project_root)
                except ValueError:
                    # Fallback if md_file is not under project_root (should not happen if target_path is)
                    relative_md_path = md_file.name 
                md_file_size = md_file.stat().st_size
                f.write(f"- 📄 `{relative_md_path}` ({md_file_size} bytes)\n")
            if len(md_files) > 20:
                f.write(f"- ... and {len(md_files) - 20} more\n")
        else:
            f.write("- INFO: No GNN source files found to list.\n")
        f.write("\n") # Add a newline for separation

        # --- Add Ontology Summary Section ---
        f.write("## 📝 Step-Specific Summaries\n\n")
        ontology_report_path = output_path / "ontology_processing" / "ontology_processing_report.md"
        ontology_summary_line = "- **Ontology Validation:** Not available (report not found or parse error)."
        if ontology_report_path.exists():
            try:
                with open(ontology_report_path, 'r', encoding='utf-8') as or_f:
                    content = or_f.read()
                    failed_match = re.search(r"- ❌ Failed: (\d+)", content)
                    passed_match = re.search(r"- ✅ Passed: (\d+)", content)
                    total_validated_match = re.search(r"- \*\*Total Annotations Validated:\*\* (\d+)", content)
                    
                    if failed_match and passed_match and total_validated_match:
                        failed_count = int(failed_match.group(1))
                        # passed_count = int(passed_match.group(1))
                        # total_validated = int(total_validated_match.group(1))
                        if failed_count > 0:
                            ontology_summary_line = f"- **Ontology Validation:** {failed_count} failed term(s). See `{ontology_report_path.relative_to(output_path.parent)}` for details."
                        else:
                            ontology_summary_line = f"- **Ontology Validation:** All terms passed. See `{ontology_report_path.relative_to(output_path.parent)}` for details."
                    elif failed_match: # Fallback if only failed count is found
                        failed_count = int(failed_match.group(1))
                        ontology_summary_line = f"- **Ontology Validation:** {failed_count} failed term(s) (summary incomplete). See `{ontology_report_path.relative_to(output_path.parent)}` for details."
                    else:
                         ontology_summary_line = f"- **Ontology Validation:** Summary could not be parsed from report. See `{ontology_report_path.relative_to(output_path.parent)}`."
            except Exception as e:
                logger.warning(f"Could not read or parse ontology report {ontology_report_path}: {e}")
                ontology_summary_line = f"- **Ontology Validation:** Error reading report. See `{ontology_report_path.relative_to(output_path.parent)}`."
        f.write(f"{ontology_summary_line}\n")
        # --- End Ontology Summary Section ---

    summary_file_size = summary_file.stat().st_size
    logger.info(f"📄 Summary report generated: {summary_file} ({summary_file_size} bytes)") # Was print if verbose
    
    return True

def export_gnn_to_all_formats(gnn_file_path: Path, base_output_dir: Path, verbose: bool = False):
    """
    Parses a single GNN file and exports it to all available formats.
    Outputs are saved in a subdirectory of base_output_dir, named after the GNN file.
    """
    if not hasattr(_gnn_model_to_dict, '__call__'): # Check if imports failed
        logger.error(f"Skipping export for {gnn_file_path.name} as format exporters could not be loaded.")
        return False

    logger.info(f"Processing exports for GNN file: {gnn_file_path.name}")
    try:
        gnn_model = _gnn_model_to_dict(str(gnn_file_path))
    except FileNotFoundError:
        logger.error(f"  ❌ File not found: {gnn_file_path}. Skipping export for this file.")
        return False
    except Exception as e:
        logger.error(f"  ❌ Error parsing GNN file {gnn_file_path}: {e}. Skipping export for this file.")
        return False

    # Create a dedicated output subdirectory for this GNN file's exports
    # e.g., output_dir/gnn_exports/my_model/
    file_specific_export_dir = base_output_dir / "gnn_exports" / gnn_file_path.stem
    try:
        file_specific_export_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"  📤 Ensured export subdirectory: {file_specific_export_dir}")
    except OSError as e:
        logger.error(f"  ❌ Failed to create export subdirectory {file_specific_export_dir}: {e}. Skipping exports for {gnn_file_path.name}.")
        return False

    export_functions = {
        "json": export_to_json_gnn,
        "xml": export_to_xml_gnn,
        "txt_summary": export_to_plaintext_summary,
        "dsl": export_to_plaintext_dsl, # .gnn or .md typically for DSL
        "gexf": export_to_gexf,
        "graphml": export_to_graphml,
        "json_adj": export_to_json_adjacency_list,
        "pkl": export_to_python_pickle
    }

    success_count = 0
    failure_count = 0

    for fmt, export_func in export_functions.items():
        # Use .gnn extension for the DSL output for clarity, others use their common extensions.
        file_extension = "gnn" if fmt == "dsl" else fmt
        # For txt_summary, use .txt
        if fmt == "txt_summary":
            file_extension = "txt"

        output_file = file_specific_export_dir / f"{gnn_file_path.stem}.{file_extension}"
        try:
            logger.debug(f"    📤 Exporting to {fmt.upper()} -> {output_file}")
            export_func(gnn_model, str(output_file))
            logger.info(f"    ✅ Successfully exported {gnn_file_path.name} to {output_file.name}")
            success_count += 1
        except Exception as e:
            logger.error(f"    ❌ Failed to export {gnn_file_path.name} to {fmt.upper()}: {e}")
            logger.debug(f"Traceback for {fmt.upper()} export failure:", exc_info=True)
            failure_count += 1
            # Optionally, clean up partially written file if necessary, though most write ops are atomic or overwrite.

    if failure_count > 0:
        logger.warning(f"  ⚠️ Finished exports for {gnn_file_path.name} with {failure_count} failure(s) out of {len(export_functions)} formats.")
    else:
        logger.info(f"  🎉 All {len(export_functions)} export formats successful for {gnn_file_path.name}.")
    return failure_count == 0

def main(args):
    """Main function for the export step."""
    logger.info(f"▶️ Starting Step 5: Export Operations ({Path(__file__).name})")
    logger.debug(f"  Parsed options: Target='{args.target_dir}', Output='{args.output_dir}', Recursive={args.recursive}, Verbose={args.verbose}")

    # Ensure output directory and main export subdirectory exist
    base_output_dir = Path(args.output_dir).resolve()
    main_gnn_export_dir = base_output_dir / "gnn_exports" # Specific subdir for GNN structured exports
    try:
        main_gnn_export_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured GNN export directory exists: {main_gnn_export_dir}")
    except OSError as e:
        logger.error(f"CRITICAL: Failed to create GNN export directory {main_gnn_export_dir}: {e}")
        return 1 # Critical failure if this directory cannot be made

    # Generate the main summary report first, as it summarizes all GNN files found
    # It uses args.target_dir (e.g. src/gnn/examples) to find .md files
    try:
        summary_success = generate_summary_report(args.target_dir, str(base_output_dir), args.recursive, args.verbose)
        if not summary_success:
            # Log this but don't necessarily make the whole step fail if exports can proceed.
            logger.warning("Summary report generation encountered issues (see logs above).")
    except Exception as e:
        logger.error(f"Error during summary report generation: {e}", exc_info=True)
        # Continue to exports if possible

    # Process individual GNN files for export to multiple formats
    # These GNN files are the source .md files
    # Their exports will go into main_gnn_export_dir / gnn_file_stem / ...
    search_pattern = "**/*.md" if args.recursive else "*.md"
    gnn_source_files = list(Path(args.target_dir).glob(search_pattern))

    if not gnn_source_files:
        logger.info(f"No GNN source files (.md) found in '{args.target_dir}' with pattern '{search_pattern}'. No specific GNN exports will be generated.")
    else:
        logger.info(f"Found {len(gnn_source_files)} GNN source files to export.")

    successful_exports = 0
    failed_exports = 0

    for gnn_file_path in gnn_source_files:
        logger.info(f"--- Processing exports for: {gnn_file_path.name} ---")
        # Note: export_gnn_to_all_formats uses 'base_output_dir' and internally appends 'gnn_exports'
        # and then the gnn_file_path.stem. This is consistent with main_gnn_export_dir being pre-created.
        export_result = export_gnn_to_all_formats(gnn_file_path, base_output_dir, args.verbose)
        if export_result:
            successful_exports += 1
        else:
            failed_exports += 1
            logger.error(f"Export failed for {gnn_file_path.name}. Check logs for details.")

    if failed_exports > 0:
        logger.warning(f"Step 5: Export operations completed with {failed_exports} GNN file(s) failing to export some/all formats.")
        # Decide if this constitutes a partial or full failure for the step's exit code.
        # For now, if any GNN files were found and attempted, let's consider it a soft failure/warning.
        # If generate_summary_report failed critically earlier, that might be a harder failure.
        # Let's return 0 if some exports were made, or if no files were found (nothing to fail on).
        # Return 1 only for critical setup failures like directory creation.
    
    if successful_exports > 0 and failed_exports == 0:
        logger.info(f"✅ Step 5: Export Operations ({Path(__file__).name}) - COMPLETED successfully for all {successful_exports} GNN file(s).")
    elif successful_exports > 0 and failed_exports > 0:
        logger.info(f"✅ Step 5: Export Operations ({Path(__file__).name}) - COMPLETED for {successful_exports} GNN file(s), but {failed_exports} had issues.")
    elif not gnn_source_files:
        logger.info(f"✅ Step 5: Export Operations ({Path(__file__).name}) - COMPLETED (no GNN source files to export).")
    else: # No successful exports, but files were present and all failed
        logger.error(f"❌ Step 5: Export Operations ({Path(__file__).name}) - FAILED (all {failed_exports} GNN files encountered export errors).")
        return 1 # If files were there but all failed to export, it's a failure of the step.

    return 0 # Default to success if it reaches here

if __name__ == "__main__":
    # This block allows 5_export.py to be run standalone.
    # Basic configuration for running this script standalone
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Mimic the args object that main.py would provide
    parser = argparse.ArgumentParser(description="GNN Processing Pipeline - Step 5: Export")
    parser.add_argument("--target-dir", default="gnn/examples",
                        help="Target directory containing GNN files (.gnn.md) to export.")
    parser.add_argument("--output-dir", default="../output",
                        help="Base directory to save exported files.")
    parser.add_argument("--recursive", action="store_true", default=True, # Changed default to True to match main.py
                        help="Recursively search for GNN files in subdirectories.")
    # Removed --formats argument as it was not used by the export_gnn_to_all_formats function.
    # The function currently exports to all compiled-in formats.
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output for this export step.")
    # Add any other args from main.py that this script's main() might expect

    parsed_args = parser.parse_args()

    # Update log level if --verbose is used in standalone mode
    if parsed_args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled for standalone run of 5_export.py.")

    sys.exit(main(parsed_args)) 