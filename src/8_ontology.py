#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 8: Ontology Processing

This script handles ontology-related operations:
- Processes Active Inference Ontology annotations
- Validates ontology mappings in GNN files
- Generates ontology analysis reports

Usage:
    python 8_ontology.py [options]
    (Typically called by main.py)
"""

import argparse
import os
import json
import datetime 
from pathlib import Path
import sys

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
logger = setup_step_logging("8_ontology", verbose=False)

# Attempt to import MCP functionalities from the ontology module
# This assumes 8_ontology.py is in src/ and mcp.py is in src/ontology/
try:
    from ontology import mcp as ontology_mcp
except ImportError:
    # Fallback if the script is run in a context where src/ontology is not directly importable
    # This might happen if PYTHONPATH is not set up as expected for direct script execution.
    try:
        # Adjust path to include src directory if running from src/ or similar context
        sys.path.append(str(Path(__file__).parent.resolve()))
        from ontology import mcp as ontology_mcp
    except ImportError as e:
        log_step_error(logger, f"Could not import 'mcp' from src/ontology/mcp.py: {e}")
        logger.error("Ensure src/ontology/mcp.py exists and src/ is discoverable.")
        ontology_mcp = None

def process_ontology_operations(target_dir_str: str, output_dir_str: str, ontology_terms_file: str = None, recursive: bool = False, verbose: bool = False):
    """Processes GNN files to extract, validate, and report on ontology annotations."""
    log_step_start(logger, f"Processing ontology operations for {target_dir_str}")
    
    if not ontology_mcp:
        log_step_error(logger, "MCP for ontology (ontology.mcp) not available. Cannot process ontology operations.")
        return False, 0

    # Log initial parameters at DEBUG level
    logger.info(f"Processing ontology related tasks...")
    logger.debug(f"Target GNN files in: {Path(target_dir_str).resolve()}")
    logger.debug(f"Output directory for ontology report: {Path(output_dir_str).resolve()}")
    logger.debug(f"Recursive mode: {'Enabled' if recursive else 'Disabled'}")
    if ontology_terms_file:
        logger.debug(f"Using ontology terms definition from: {Path(ontology_terms_file).resolve()}")
    else:
        log_step_warning(logger, "No ontology terms definition file provided. Validation will be skipped.")

    # Conceptual Ontology Logging (if verbose)
    logger.debug("Conceptual Note: Ontologies provide a formal way to represent knowledge.")
    logger.debug("  - Informal ontologies (like folksonomies or taxonomies) help organize concepts.")
    logger.debug("  - Formal ontologies (e.g., in OWL, RDF) allow for logical reasoning and consistency checks.")
    logger.debug("  - This script focuses on extracting and validating terms based on a predefined JSON schema.")
    logger.debug("  - Different ontology languages (OWL, RDF, SKOS) offer varying expressiveness.")

    target_dir = Path(target_dir_str)
    output_dir = Path(output_dir_str)
    ontology_output_path = output_dir / "ontology_processing"
    ontology_output_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ontology report will be saved in: {ontology_output_path.resolve()}")

    # Determine project root for relative paths in report
    project_root = Path(__file__).resolve().parent.parent

    defined_ontology_terms = {}
    if ontology_terms_file:
        logger.debug(f"Loading defined ontology terms from: {ontology_terms_file}")
        defined_ontology_terms = ontology_mcp.load_defined_ontology_terms(ontology_terms_file, verbose=verbose)
        if defined_ontology_terms:
            logger.debug(f"Loaded {len(defined_ontology_terms)} ontology terms successfully.")
        else:
            log_step_warning(logger, f"Could not load or no terms found in {ontology_terms_file}. Validation may be limited.")

    report_title = "# 🧬 GNN Ontological Annotations Report"
    all_reports_parts = [report_title]
    all_reports_parts.append(f"🗓️ Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Make paths relative to project root for the report
    try:
        reported_target_dir = target_dir.relative_to(project_root)
    except ValueError:
        reported_target_dir = target_dir # Keep absolute if not under project_root (e.g. symlink or unusual setup)
    all_reports_parts.append(f"🎯 GNN Source Directory: `{reported_target_dir}`")
    
    if ontology_terms_file:
        try:
            reported_ontology_file = Path(ontology_terms_file).resolve().relative_to(project_root)
        except ValueError:
            reported_ontology_file = Path(ontology_terms_file).resolve() # Keep absolute if not under project root
        all_reports_parts.append(f"📖 Ontology Terms Definition: `{reported_ontology_file}` (Loaded: {len(defined_ontology_terms)} terms)")
    else:
        all_reports_parts.append("⚠️ Ontology Terms Validation: Skipped (no definition file provided)")
    all_reports_parts.append("\n---\n")

    search_pattern = "**/*.md" if recursive else "*.md"
    gnn_files = list(target_dir.glob(search_pattern))

    if not gnn_files:
        logger.info(f"No .md files found in '{target_dir}' with search pattern '{search_pattern}'.")
        all_reports_parts.append("**No GNN (.md) files found to process in the specified target directory.**\n")
    else:
        logger.debug(f"Found {len(gnn_files)} GNN (.md) files to process.")

    processed_file_count = 0
    total_annotations_found = 0
    total_validations_passed = 0
    total_validations_failed = 0

    for gnn_file_path in gnn_files:
        logger.debug(f"Processing file: {gnn_file_path.name} ({gnn_file_path.stat().st_size} bytes)")
        try:
            with open(gnn_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Assuming parse_gnn_ontology_section returns a list/dict of annotations
            parsed_annotations = ontology_mcp.parse_gnn_ontology_section(content, verbose=verbose)
            num_file_annotations = len(parsed_annotations) if parsed_annotations else 0
            total_annotations_found += num_file_annotations
            if num_file_annotations > 0:
                logger.debug(f"Found {num_file_annotations} ontology annotations in {gnn_file_path.name}.")
            
            validation_results = None
            file_valid = 0
            file_invalid = 0
            if defined_ontology_terms and parsed_annotations:
                validation_results = ontology_mcp.validate_annotations(parsed_annotations, defined_ontology_terms, verbose=verbose)
                if validation_results:
                    # Directly use the counts from the validation_results structure
                    file_valid = len(validation_results.get("valid_mappings", {}))
                    file_invalid = len(validation_results.get("invalid_terms", {}))
                    
                    total_validations_passed += file_valid
                    total_validations_failed += file_invalid
                    if file_valid > 0 or file_invalid > 0:
                        logger.debug(f"Validated for {gnn_file_path.name}: {file_valid} passed, {file_invalid} failed.")
            
            # The path passed to generate_ontology_report_for_file needs to be relative to project root.
            try:
                report_file_display_path = gnn_file_path.resolve().relative_to(project_root)
            except ValueError:
                report_file_display_path = gnn_file_path.name # Fallback to just filename if not in project root

            file_report_str = ontology_mcp.generate_ontology_report_for_file(
                str(report_file_display_path),
                parsed_annotations, 
                validation_results
            )
            all_reports_parts.append(file_report_str)
            processed_file_count +=1
        except Exception as e:
            log_step_error(logger, f"Error processing file {gnn_file_path.name}: {e}")
            all_reports_parts.append(f"### Error processing `{gnn_file_path.name}`\n - {str(e)}\n\n---\n")

    # Add a summary section to the report
    all_reports_parts.insert(1, f"\n## 📊 Summary of Ontology Processing\n")
    all_reports_parts.insert(2, f"- **Files Processed:** {processed_file_count}")
    all_reports_parts.insert(3, f"- **Total Annotations Found:** {total_annotations_found}")
    if defined_ontology_terms:
        all_reports_parts.insert(4, f"- **Validations Passed:** {total_validations_passed}")
        all_reports_parts.insert(5, f"- **Validations Failed:** {total_validations_failed}")
    else:
        all_reports_parts.insert(4, "- **Validation:** Skipped (no terms definition provided)")
    all_reports_parts.insert(-1, "\n")

    # Write the final report
    final_report_str = "\n".join(all_reports_parts)
    report_file_path = ontology_output_path / "ontology_processing_report.md"
    
    try:
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write(final_report_str)
        
        log_step_success(logger, f"Ontology processing completed successfully. Report saved: {report_file_path}")
    except Exception as e:
        log_step_error(logger, f"Failed to write ontology report: {e}")
        return False, processed_file_count

    return True, processed_file_count

def main(args):
    """Main function for the ontology step (Step 8).

    This function orchestrates the ontology processing for GNN files.
    It is typically called by the main pipeline.

    Args:
        args (argparse.Namespace): 
            Parsed command-line arguments. Expected attributes include:
            target_dir, output_dir, ontology_terms_file, recursive, verbose.
    """
    
    # Update logger if verbose mode enabled
    if hasattr(args, 'verbose') and args.verbose and UTILS_AVAILABLE:
        global logger
        logger = setup_step_logging("8_ontology", verbose=True)

    log_step_start(logger, f"Starting Step 8: Ontology Operations ({Path(__file__).name})")

    # Get ontology terms file, using default if not provided
    ontology_terms_file = getattr(args, 'ontology_terms_file', None)
    if not ontology_terms_file:
        # Default to the standard location
        default_ontology_file = Path(__file__).parent / "ontology" / "act_inf_ontology_terms.json"
        if default_ontology_file.exists():
            ontology_terms_file = str(default_ontology_file)
            logger.debug(f"Using default ontology terms file: {ontology_terms_file}")

    success, files_processed = process_ontology_operations(
        args.target_dir, 
        args.output_dir, 
        ontology_terms_file=ontology_terms_file,
        recursive=args.recursive if hasattr(args, 'recursive') else False,
        verbose=args.verbose
    )
    
    if success:
        log_step_success(logger, f"Step 8: Ontology Operations completed successfully. Processed {files_processed} files.")
        return 0
    else:
        log_step_error(logger, f"Step 8: Ontology Operations failed.")
        return 1

if __name__ == "__main__":
    # Basic configuration for running this script standalone
    # In a pipeline, main.py should configure logging.
    # log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    # log_level = getattr(logging, log_level_str, logging.INFO)
    # logging.basicConfig(
    #     level=log_level,
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S"
    # )
    # Simplified arg parsing for standalone run
    parser = argparse.ArgumentParser(description="Standalone Ontology Processing.")
    
    # Define defaults for standalone execution relative to this script's project root
    script_file_path = Path(__file__).resolve()
    project_root_for_defaults = script_file_path.parent.parent # src/ -> project_root
    default_target_dir_standalone = project_root_for_defaults / "output" / "gnn_exports" # Example, adjust if needed
    default_output_dir_standalone = project_root_for_defaults / "output"
    default_ontology_terms_file_standalone = project_root_for_defaults / "src" / "ontology" / "act_inf_ontology_terms.json"

    parser.add_argument("--target-dir", type=Path, default=default_target_dir_standalone, help="Directory with GNN exports.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir_standalone, help="Output directory for reports.")
    parser.add_argument("--ontology-terms-file", type=Path, default=default_ontology_terms_file_standalone, help="JSON file with defined ontology terms.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Enable verbose output.")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False, help="Recursively process GNN files in the target directory.")
    
    cli_args = parser.parse_args()
    
    # Update logger for standalone execution
    if cli_args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)

    result_code = main(cli_args) 
    sys.exit(result_code) 