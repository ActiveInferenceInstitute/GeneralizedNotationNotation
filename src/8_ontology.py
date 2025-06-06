#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 8: Ontology Operations

This script handles ontology-specific operations, such as:
- Loading and parsing ontology files.
- Performing ontology-based analysis or validation.
- Integrating ontological information with GNN models.

Usage:
    python 8_ontology.py [options]
    
Options:
    Same as main.py (though many may not be relevant for this specific step)
"""

import os
import sys
import glob # For finding .md files
from pathlib import Path
import argparse
import datetime
import logging # Import logging

# Attempt to import the new logging utility
try:
    from utils.logging_utils import setup_standalone_logging
except ImportError:
    # Fallback for standalone execution or if src is not directly in path
    current_script_path_for_util = Path(__file__).resolve()
    project_root_for_util = current_script_path_for_util.parent.parent
    # Try adding project root, then src, to sys.path for utils
    paths_to_try = [str(project_root_for_util), str(project_root_for_util / "src")]
    original_sys_path = list(sys.path)
    for p in paths_to_try:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from utils.logging_utils import setup_standalone_logging
    except ImportError:
        setup_standalone_logging = None
        _temp_logger_name = __name__ if __name__ != "__main__" else "src.8_ontology_import_warning"
        _temp_logger = logging.getLogger(_temp_logger_name)
        if not _temp_logger.hasHandlers():
            if not logging.getLogger().hasHandlers():
                logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
            else:
                 _temp_logger.addHandler(logging.StreamHandler(sys.stderr))
                 _temp_logger.propagate = False
        _temp_logger.warning(
            "Could not import setup_standalone_logging from utils.logging_utils. Standalone logging might be basic."
        )
    finally:
        # Restore original sys.path to avoid side effects if this script is imported elsewhere
        sys.path = original_sys_path

# Logger for this module
logger = logging.getLogger(__name__)

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
        logger.error(f"Error: Could not import 'mcp' from src/ontology/mcp.py: {e}")
        logger.error("Ensure src/ontology/mcp.py exists and src/ is discoverable.")
        ontology_mcp = None

def process_ontology_operations(target_dir_str: str, output_dir_str: str, ontology_terms_file: str = None, recursive: bool = False, verbose: bool = False):
    """Processes GNN files to extract, validate, and report on ontology annotations."""
    if not ontology_mcp:
        logger.error("❌🧬 MCP for ontology (ontology.mcp) not available. Cannot process ontology operations.")
        return False, 0

    # Log initial parameters at DEBUG level
    logger.info(f"  🔎 Processing ontology related tasks...")
    logger.debug(f"    🎯 Target GNN files in: {Path(target_dir_str).resolve()}")
    logger.debug(f"    ել Output directory for ontology report: {Path(output_dir_str).resolve()}")
    logger.debug(f"    🔄 Recursive mode: {'Enabled' if recursive else 'Disabled'}")
    if ontology_terms_file:
        logger.debug(f"    📖 Using ontology terms definition from: {Path(ontology_terms_file).resolve()}")
    else:
        logger.warning("    ⚠️ No ontology terms definition file provided. Validation will be skipped.")

    # Conceptual Ontology Logging (if verbose)
    logger.debug("    🧠 Conceptual Note: Ontologies provide a formal way to represent knowledge.")
    logger.debug("      - Informal ontologies (like folksonomies or taxonomies) help organize concepts.")
    logger.debug("      - Formal ontologies (e.g., in OWL, RDF) allow for logical reasoning and consistency checks.")
    logger.debug("      - This script focuses on extracting and validating terms based on a predefined JSON schema.")
    logger.debug("      - Different ontology languages (OWL, RDF, SKOS) offer varying expressiveness.")

    target_dir = Path(target_dir_str)
    output_dir = Path(output_dir_str)
    ontology_output_path = output_dir / "ontology_processing"
    ontology_output_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"    ✍️ Ontology report will be saved in: {ontology_output_path.resolve()}")

    # Determine project root for relative paths in report
    project_root = Path(__file__).resolve().parent.parent

    defined_ontology_terms = {}
    if ontology_terms_file:
        logger.debug(f"    🧐 Loading defined ontology terms from: {ontology_terms_file}")
        defined_ontology_terms = ontology_mcp.load_defined_ontology_terms(ontology_terms_file, verbose=verbose)
        if defined_ontology_terms:
            logger.debug(f"      📚 Loaded {len(defined_ontology_terms)} ontology terms successfully.")
        else:
            logger.warning(f"      ⚠️ Could not load or no terms found in {ontology_terms_file}. Validation may be limited.")

    report_title = "# 🧬 GNN Ontological Annotations Report"
    all_reports_parts = [report_title]
    all_reports_parts.append(f"��️ Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        logger.info(f"    ℹ️ No .md files found in '{target_dir}' with search pattern '{search_pattern}'.")
        all_reports_parts.append("**No GNN (.md) files found to process in the specified target directory.**\n")
    else:
        logger.debug(f"    📊 Found {len(gnn_files)} GNN (.md) files to process.")

    processed_file_count = 0
    total_annotations_found = 0
    total_validations_passed = 0
    total_validations_failed = 0

    for gnn_file_path in gnn_files:
        logger.debug(f"    📄 Processing file: {gnn_file_path.name} ({gnn_file_path.stat().st_size} bytes)")
        try:
            with open(gnn_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Assuming parse_gnn_ontology_section returns a list/dict of annotations
            parsed_annotations = ontology_mcp.parse_gnn_ontology_section(content, verbose=verbose)
            num_file_annotations = len(parsed_annotations) if parsed_annotations else 0
            total_annotations_found += num_file_annotations
            if num_file_annotations > 0:
                logger.debug(f"      Found {num_file_annotations} ontology annotations in {gnn_file_path.name}.")
            
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
                        logger.debug(f"        Validated for {gnn_file_path.name}: {file_valid} passed, {file_invalid} failed.")
            
            # The path passed to generate_ontology_report_for_file needs to be relative to project root.
            # Original: str(gnn_file_path.relative_to(target_dir.parent if target_dir.is_dir() else target_dir.parent.parent))
            # gnn_file_path is absolute here. target_dir is also absolute.
            # We want path relative to project_root, e.g. src/gnn/examples/file.md
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
            logger.error(f"    ❌ Error processing file {gnn_file_path.name}: {e}", exc_info=True)
            all_reports_parts.append(f"### Error processing `{gnn_file_path.name}`\n - {str(e)}\n\n---\n")

    # Add a summary section to the report
    all_reports_parts.insert(1, f"\n## 📊 Summary of Ontology Processing\n")
    all_reports_parts.insert(2, f"- **Files Processed:** {processed_file_count} / {len(gnn_files)}")
    all_reports_parts.insert(3, f"- **Total Ontological Annotations Found:** {total_annotations_found}")
    if defined_ontology_terms:
        all_reports_parts.insert(4, f"- **Total Annotations Validated:** {total_validations_passed + total_validations_failed}")
        all_reports_parts.insert(5, f"  - ✅ Passed: {total_validations_passed}")
        all_reports_parts.insert(6, f"  - ❌ Failed: {total_validations_failed}")
    all_reports_parts.insert(7, "\n---\n")
            
    report_file_path = ontology_output_path / "ontology_processing_report.md"
    try:
        with open(report_file_path, 'w', encoding='utf-8') as f_report:
            f_report.write("\n".join(all_reports_parts))
        report_size = report_file_path.stat().st_size
        logger.debug(f"  ✅ Ontology processing report saved: {report_file_path.resolve()} ({report_size} bytes)")
    except Exception as e:
        logger.error(f"❌ Failed to write ontology report to {report_file_path}: {e}", exc_info=True)
        return False, 0
        
    return True, total_validations_failed

def main(args):
    """Main function for the ontology operations step (Step 8).

    This function is the entry point for ontology processing. It logs the start
    of the step and calls `process_ontology_operations` with the necessary arguments
    derived from the `args` Namespace object.

    Args:
        args (argparse.Namespace): 
            Parsed command-line arguments from `main.py` or standalone execution.
            Expected attributes include: target_dir, output_dir, ontology_terms_file,
            recursive, verbose.
    """
    # Set this script's logger level based on pipeline's args.verbose
    # This is typically handled by main.py for child modules.
    # The __main__ block handles it for standalone execution.
    # if args.verbose:
    #     logger.setLevel(logging.DEBUG)
    # else:
    #     logger.setLevel(logging.INFO)

    logger.info(f"▶️ Starting Step 8: Ontology Operations ({Path(__file__).name})")
    logger.debug(f"  Parsed options (from main.py or standalone):")
    logger.debug(f"    Target GNN files directory: {args.target_dir}")
    logger.debug(f"    Output directory for report: {args.output_dir}")
    logger.debug(f"    Recursive: {args.recursive}")
    if args.ontology_terms_file:
        logger.debug(f"    Ontology terms definition file: {args.ontology_terms_file}")
    else:
        logger.debug(f"    Ontology terms file: Not provided (validation will be skipped)")
    logger.debug(f"    Verbose flag from args: {args.verbose}")

    success, num_failed_validations = process_ontology_operations(
        args.target_dir, 
        args.output_dir, 
        args.ontology_terms_file,
        args.recursive if hasattr(args, 'recursive') else False, 
        args.verbose
    )

    if not success:
        logger.error(f"❌ Step 8: Ontology Operations ({Path(__file__).name}) FAILED critically.")
        return 1 # Critical failure
    
    if num_failed_validations > 0:
        warning_message = (
            f"Ontology validation completed with {num_failed_validations} failed term(s). "
            f"Check '{Path(args.output_dir) / 'ontology_processing/ontology_processing_report.md'}' "
            f"for details."
        )
        logger.warning(f"⚠️ Step 8: {warning_message}")
        # Return 0 for success with warnings, allowing the pipeline to continue
        return 0
        
    logger.info(f"✅ Step 8: Ontology Operations ({Path(__file__).name}) - COMPLETED without validation errors.")
    return 0

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
    
    # Setup logging for standalone execution
    log_level_to_set = logging.DEBUG if cli_args.verbose else logging.INFO
    if setup_standalone_logging:
        setup_standalone_logging(level=log_level_to_set, logger_name=__name__)
    else:
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=log_level_to_set,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                # datefmt="%Y-%m-%d %H:%M:%S", # Use default datefmt
                stream=sys.stdout
            )
        logging.getLogger(__name__).setLevel(log_level_to_set) 
        logging.getLogger(__name__).warning("Using fallback basic logging due to missing setup_standalone_logging utility.")

    # Quieten noisy libraries if run standalone
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    result_code = main(cli_args) 
    sys.exit(result_code) 