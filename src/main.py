#!/usr/bin/env python3
"""
GNN Processing Pipeline

This script orchestrates the full processing pipeline for GNN files and related artifacts.
It dynamically discovers and runs numbered scripts in the src/ directory, where each script
corresponds to a top-level folder and a specific processing stage.

Pipeline Steps (Dynamically Discovered and Ordered):
- 1_gnn.py (Corresponds to gnn/ folder)
- 2_setup.py (Corresponds to setup/ folder)
- 3_tests.py (Corresponds to tests/ folder)
- 4_gnn_type_checker.py (Corresponds to gnn_type_checker/ folder, uses gnn_type_checker.py)
- 5_export.py (Corresponds to export/ folder)
- 6_visualization.py (Corresponds to visualization/ folder, uses visualize_gnn.py)
- 7_mcp.py (Corresponds to mcp/ folder)
- 8_ontology.py (Corresponds to ontology/ folder)

Usage:
    python main.py [options]
    
Options:
    --target-dir DIR        Target directory for GNN files (default: gnn/examples)
                            (Note: Individual scripts might target their specific folders e.g. src/mcp)
    --output-dir DIR        Directory to save outputs (default: ../output)
    --recursive             Recursively process directories (passed to relevant steps)
    --skip-steps LIST       Comma-separated list of steps to skip (e.g., "1_gnn,7_mcp" or "1,7")
    --only-steps LIST       Comma-separated list of steps to run (e.g., "4_gnn_type_checker,6_visualization")
    --verbose               Enable verbose output
    --strict                Enable strict type checking mode (for 4_gnn_type_checker)
    --estimate-resources    Estimate computational resources (for 4_gnn_type_checker)
    --ontology-terms-file   Path to the ontology terms file (e.g., src/ontology/act_inf_ontology_terms.json, for 8_ontology)
"""

import os
import sys
import importlib
import argparse
import glob
from pathlib import Path
import logging
import traceback

# --- Logger Setup ---
# The logger for the GNN Pipeline orchestrator itself.
# Configuration will be done by basicConfig in the __main__ block.
logger = logging.getLogger("GNN_Pipeline")
# --- End Logger Setup ---

def parse_arguments():
    # Calculate default output directory relative to project root
    # Assuming main.py is in src/ and project root is its parent.
    script_file_path = Path(__file__).resolve() # Full path to src/main.py
    project_root = script_file_path.parent.parent # Project root
    default_output_dir = project_root / "output"

    parser = argparse.ArgumentParser(description="GNN Processing Pipeline",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--target-dir", default="gnn/examples",
                        help=("Target directory for GNN files (default: gnn/examples)\\n"
                             "(Note: Individual scripts might target their specific folders e.g. src/mcp)"))
    parser.add_argument("--output-dir", default=str(default_output_dir),
                        help=f"Directory to save outputs (default: {default_output_dir})")
    parser.add_argument("--recursive", default=True, action=argparse.BooleanOptionalAction,
                        help="Recursively process directories (passed to relevant steps). Enabled by default. Use --no-recursive to disable.")
    parser.add_argument("--skip-steps", default="",
                        help=("Comma-separated list of steps to skip (e.g., \"1_gnn,7_mcp\" or \"1,7\")"))
    parser.add_argument("--only-steps", default="",
                        help=("Comma-separated list of steps to run (e.g., \"4_gnn_type_checker,6_visualization\")"))
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--strict", action="store_true",
                        help="Enable strict type checking mode (for 4_gnn_type_checker)")
    parser.add_argument("--estimate-resources", default=True, action=argparse.BooleanOptionalAction,
                        help="Estimate computational resources (for 4_gnn_type_checker) (default: True)")
    parser.add_argument("--ontology-terms-file", default="ontology/act_inf_ontology_terms.json",
                        help="Path to the ontology terms file (e.g., src/ontology/act_inf_ontology_terms.json, for 8_ontology)")
    return parser.parse_args()

def get_pipeline_scripts():
    """Get all numbered pipeline scripts in sorted order based on the leading number."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_pattern = os.path.join(current_dir, "[0-9]_*.py")
    logger.debug(f"ℹ️ Discovering pipeline scripts using pattern: {script_pattern}")
    scripts_with_paths = sorted(glob.glob(script_pattern))
    
    if logger.isEnabledFor(logging.DEBUG): # Log full paths only if DEBUG is enabled (typically via --verbose)
        logger.debug(f"ℹ️ Found raw script files (full paths): {scripts_with_paths}")
        
    return [os.path.basename(script) for script in scripts_with_paths]

def run_pipeline(args):
    """Run the full GNN processing pipeline."""
    # Logger level is set by basicConfig based on args.verbose before this function is called.
    
    # --- ASCII Art Intro ---
    logger.info("""

      ____ ____ ____   ____ ___  _ ____ ___ _ ____ _  _ ____
     / ___| __ |__/   |  | |  \\ | |__|  |  | |___ |\\/| |___
     \\___|__| |  \\   |__| |__/ | |  |  |  | |___ |  | |___ 

    Initializing GNN Processing Pipeline...
    """)
    # --- End ASCII Art Intro ---

    # Resolve and log paths
    abs_target_dir = Path(args.target_dir).resolve()
    abs_output_dir = Path(args.output_dir).resolve()

    if args.verbose: # Or use logger.isDebugEnabled()
        logger.debug(f"ℹ️ Resolved Target Directory: {abs_target_dir}")
        logger.debug(f"ℹ️ Resolved Output Directory: {abs_output_dir}")
        logger.debug(f"ℹ️ Full Arguments: {args}") # Moved from later

    # Create output directory
    try:
        abs_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Ensured output directory exists: {abs_output_dir}")
    except OSError as e:
        logger.error(f"❌ Failed to create output directory {abs_output_dir}: {e}")
        return 1 # Cannot proceed without output directory

    # --- BEGIN VENV PATH MODIFICATION ---
    # Attempt to add the virtual environment's site-packages to sys.path
    # This ensures that modules imported by importlib can find packages installed
    # by the setup script (e.g., 2_setup.py) into the .venv.
    current_script_dir = Path(__file__).resolve().parent # Should be src/
    venv_path = current_script_dir / ".venv"
    site_packages_found_path = None

    if venv_path.is_dir(): # Check if .venv exists and is a directory
        lib_path = venv_path / "lib"
        if lib_path.is_dir():
            # Iterate through subdirectories in lib/ (e.g., python3.10, python3.12)
            for python_version_dir in lib_path.iterdir():
                if python_version_dir.is_dir() and python_version_dir.name.startswith("python"):
                    current_site_packages = python_version_dir / "site-packages"
                    if current_site_packages.is_dir():
                        site_packages_found_path = current_site_packages
                        logger.debug(f"Found site-packages at: {site_packages_found_path}")
                        break # Found one, assume it's the correct one for the venv
    
    if site_packages_found_path:
        site_packages_str = str(site_packages_found_path.resolve())
        if site_packages_str not in sys.path:
            logger.info(f"ℹ️ Adding to sys.path for dynamic module loading: {site_packages_str}")
            sys.path.insert(0, site_packages_str) # Prepend to give it priority
        else:
            logger.debug(f"ℹ️ Virtual env site-packages already in sys.path: {site_packages_str}")
    elif venv_path.is_dir(): # .venv exists but site-packages not found as expected
        logger.debug(
            f"ℹ️ Virtual environment at {venv_path} exists, "
            f"but its site-packages directory could not be automatically determined or found. "
            f"Dynamically loaded modules might not find their dependencies if 2_setup.py is skipped or fails."
        )
    else: # .venv directory itself does not exist
        logger.debug(
            f"ℹ️ No .venv directory found at {venv_path}. "
            f"Assuming system Python or environment is managed externally, "
            f"or will be created by a setup script in the pipeline."
        )
    # --- END VENV PATH MODIFICATION ---

    all_scripts = get_pipeline_scripts()
    
    if not all_scripts:
        logger.error("❌ No pipeline scripts found (e.g., 1_gnn.py, 2_setup.py, etc.).")
        logger.error("Please ensure numbered scripts matching folder names exist in the src/ directory.")
        return 1
    
    logger.info("🚀 Starting GNN Processing Pipeline...")
    if args.verbose: # Log discovered scripts only in verbose mode for cleaner default output
        logger.debug(f"ℹ️ All discovered script modules (basenames): {all_scripts}")

    skip_steps_input = [s.strip() for s in args.skip_steps.split(",") if s.strip()] if args.skip_steps else []
    only_steps_input = [s.strip() for s in args.only_steps.split(",") if s.strip()] if args.only_steps else []
    
    processed_scripts_count = 0
    successful_scripts_count = 0
    skipped_due_to_args_count = 0
    skipped_due_to_no_main_count = 0
    failed_steps_details = [] # Stores tuples of (script_name, reason, exit_code_or_exception)
    pipeline_level_warnings = [] # To store summaries and detailed warnings from steps
    critical_step_failed = False
    critical_step_name = ""

    for i, script_file in enumerate(all_scripts):
        script_name_no_ext = os.path.splitext(script_file)[0]
        step_num_str = script_name_no_ext.split("_")[0]
        
        step_header = f"Step {i+1}/{len(all_scripts)}: {script_name_no_ext}"
        # Consider making other steps critical, e.g., 1_gnn if its failure means no data for subsequent steps.
        is_critical_step = (script_name_no_ext == "2_setup") 

        should_skip_due_to_args = False
        if only_steps_input:
            if not (script_name_no_ext in only_steps_input or 
                      script_file in only_steps_input or 
                      step_num_str in only_steps_input):
                should_skip_due_to_args = True
        elif skip_steps_input:
            if (script_name_no_ext in skip_steps_input or 
                script_file in skip_steps_input or 
                step_num_str in skip_steps_input):
                should_skip_due_to_args = True
        
        if should_skip_due_to_args:
            logger.info(f"⏭️ {step_header} - SKIPPED (due to --skip-steps or --only-steps filter)")
            skipped_due_to_args_count += 1
            continue
        
        # If not skipped by args, it's considered for processing.
        processed_scripts_count += 1
        logger.info(f"⚙️ {step_header} (from {script_file}) - STARTING")
        
        try:
            module = importlib.import_module(script_name_no_ext)
            
            if hasattr(module, 'main'):
                result = module.main(args) # This is where individual scripts execute
                
                if isinstance(result, dict) and result.get('status') == 'success_with_warnings':
                    successful_scripts_count += 1
                    logger.info(f"✅ {step_header} - COMPLETED with warnings.")
                    if 'summary' in result and 'warnings' in result:
                        pipeline_level_warnings.append({
                            'step': script_name_no_ext,
                            'summary': result['summary'],
                            'details': result['warnings']
                        })
                        # Log the detailed warnings from the step immediately if verbose, 
                        # or a summary otherwise. The main summary will show all summaries later.
                        if args.verbose:
                            for warn_detail in result['warnings']:
                                logger.warning(f"   Step Warning: {warn_detail}")
                        else:
                            logger.warning(f"   Step Summary: {result['summary']}")
                    else:
                        logger.warning(f"   Step returned 'success_with_warnings' but without standard 'summary' or 'warnings' keys.")

                elif result not in [0, None]: # Step reported a controlled failure via return code
                    error_message = f"❌ {step_header} - FAILED (Reported exit code: {result})."
                    logger.error(error_message)
                    failed_steps_details.append({'name': script_name_no_ext, 'reason': 'NonZeroExitCode', 'code': result})
                    
                    if is_critical_step:
                        logger.critical(f"🔥 This ({script_name_no_ext}) was a CRITICAL step. Halting pipeline.")
                        critical_step_failed = True
                        critical_step_name = script_name_no_ext
                        # No return here; summary will handle final exit code
                        break # Exit loop to go to summary
                    else:
                        logger.warning(f"⚠️ This was a non-critical step. Pipeline will attempt to continue.")
                else: # Successful run (exit code 0 or None)
                    successful_scripts_count += 1
                    logger.info(f"✅ {step_header} - COMPLETED successfully.")
                    if args.verbose: # Detailed success only in verbose
                         logger.debug(f"   (Script returned: {result})")
            else:
                logger.warning(f"⚠️ {step_header} (from {script_file}) - SKIPPED (no main() function found).")
                skipped_due_to_no_main_count +=1
                # It was an attempt to process, so don't decrement processed_scripts_count here.
                # We'll account for it in the summary.
                
        except ImportError as e:
            logger.error(f"❌ Error importing module {script_name_no_ext} for script {script_file}: {e}")
            logger.debug(traceback.format_exc(), exc_info=True)
            failed_steps_details.append({'name': script_name_no_ext, 'reason': 'ImportError', 'details': str(e)})
            if is_critical_step:
                 logger.critical(f"🔥 Critical step {script_name_no_ext} failed to import. Halting pipeline.")
                 critical_step_failed = True
                 critical_step_name = script_name_no_ext
                 break # Exit loop
            else:
                logger.warning(f"⚠️ Module import failure for non-critical step {script_name_no_ext}. Pipeline will attempt to continue.")

        except Exception as e:
            logger.error(f"❌ Unhandled exception in step {script_name_no_ext} (from {script_file}): {e}")
            logger.debug(traceback.format_exc(), exc_info=True) # Log full traceback in debug
            failed_steps_details.append({'name': script_name_no_ext, 'reason': 'UnhandledException', 'details': str(e)})
            if is_critical_step:
                logger.critical(f"🔥 Critical step {script_name_no_ext} encountered an unhandled exception. Halting pipeline.")
                critical_step_failed = True
                critical_step_name = script_name_no_ext
                break # Exit loop
            else:
                logger.warning(f"⚠️ Step {script_name_no_ext} failed due to an unhandled exception. Pipeline will attempt to continue.")
    
    # --- Pipeline Summary ---
    logger.info("\n--- Pipeline Execution Summary ---")
    total_discovered_scripts = len(all_scripts)
    total_attempted_to_run = processed_scripts_count # Scripts not skipped by args
    total_skipped = skipped_due_to_args_count + skipped_due_to_no_main_count
    
    logger.info(f"📊 Total Scripts Discovered: {total_discovered_scripts}")
    logger.info(f"   Attempted to Run:      {total_attempted_to_run}")
    logger.info(f"   Successfully Completed: {successful_scripts_count}")
    logger.info(f"   Skipped:                {total_skipped}")
    if total_skipped > 0:
        logger.info(f"     - Skipped by arguments (--skip-steps/--only-steps): {skipped_due_to_args_count}")
        logger.info(f"     - Skipped (no main() function):                   {skipped_due_to_no_main_count}")
    
    num_failed_steps = len(failed_steps_details)
    logger.info(f"   Failed:                 {num_failed_steps}")

    final_exit_code = 0

    if pipeline_level_warnings:
        logger.warning("\n--- Pipeline Step Warnings Summary ---")
        for warning_info in pipeline_level_warnings:
            logger.warning(f"  - Step '{warning_info['step']}': {warning_info['summary']}")
            if args.verbose: # In verbose mode, re-iterate detailed warnings here or ensure they were logged before.
                for detail in warning_info['details']:
                    logger.warning(f"    Details: {detail}")
        logger.warning("  (For full details, check individual step logs or reports mentioned in warnings.)")

    if critical_step_failed:
        logger.error(f"🛑 PIPELINE HALTED due to critical step failure: '{critical_step_name}'.")
        if failed_steps_details:
            last_failure = failed_steps_details[-1] # Assumes the critical failure is the last one if loop broke
            if last_failure['name'] == critical_step_name:
                 logger.error(f"   Reason: {last_failure['reason']}, Details: {last_failure.get('code') or last_failure.get('details')}")
        final_exit_code = 1 
    elif num_failed_steps > 0:
        logger.warning(f"⚠️ PIPELINE COMPLETED, but with {num_failed_steps} FAILED non-critical step(s):")
        for failure in failed_steps_details:
            logger.warning(f"   - Script: {failure['name']}, Reason: {failure['reason']}, Details: {failure.get('code') or failure.get('details')}")
        logger.info("   Please review the logs for details on these failed steps.")
        final_exit_code = 1 # Non-critical failure, but still an error exit code
    elif total_attempted_to_run > 0 and successful_scripts_count == total_attempted_to_run:
        if pipeline_level_warnings:
            logger.warning("🎉 PIPELINE FINISHED, but with warnings from some steps. Please review above.")
            # Optionally, could set final_exit_code to a specific value for 'success with warnings'
        else:
            logger.info("🎉 PIPELINE FINISHED SUCCESSFULLY. All attempted steps completed without errors.")
    elif total_attempted_to_run == 0 and total_discovered_scripts > 0 : # All scripts skipped by args or no main
        logger.info("✅ PIPELINE FINISHED. All discovered steps were skipped (by arguments or missing main() function).")
    elif total_discovered_scripts == 0: # No scripts found at all
         logger.info("✅ PIPELINE FINISHED. No scripts were found to process.")
    else: # Mixed outcome, possibly some skipped, some successful, no failures that weren't critical
        logger.info("✅ PIPELINE FINISHED.") # General completion message
    
    return final_exit_code

def main():
    args = parse_arguments()
    
    # --- Centralized Logging Configuration ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format)
    # --- End Centralized Logging Configuration ---

    # Log initial arguments, now that logging is configured.
    logger.debug(f"Parsed arguments: {args}")

    # Resolve paths after argument parsing and logging setup
    # Ensure target_dir and output_dir are absolute for clarity and consistency
    # Note: Individual pipeline scripts will receive the args object and should resolve paths as needed,
    # potentially using their own script's location as a base if args.target_dir is relative for them.
    original_target_dir = args.target_dir
    original_output_dir = args.output_dir

    # Attempt to resolve relative to CWD first, which is common for CLI tools.
    # If they are already absolute, resolve() does no harm.
    args.target_dir = str(Path(args.target_dir).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    
    # Log resolved paths. Important if the original was relative.
    if Path(original_target_dir) != Path(args.target_dir):
        logger.debug(f"Original target_dir '{original_target_dir}' resolved to '{args.target_dir}'")
    if Path(original_output_dir) != Path(args.output_dir):
        logger.debug(f"Original output_dir '{original_output_dir}' resolved to '{args.output_dir}'")

    if not Path(args.target_dir).exists():
        logger.warning(f"⚠️ Target directory {args.target_dir} does not exist. Some steps might fail or create it.")

    # Before running the pipeline, ensure critical paths used by main.py itself are set up.
    # For example, if `ontology_terms_file` is relative, resolve it against a sensible base.
    # Here, we assume it might be relative to the project root if not absolute.
    if args.ontology_terms_file and not Path(args.ontology_terms_file).is_absolute():
        script_file_path = Path(__file__).resolve()
        project_root = script_file_path.parent.parent
        resolved_ontology_file = project_root / args.ontology_terms_file
        if resolved_ontology_file.exists():
            logger.debug(f"Ontology terms file '{args.ontology_terms_file}' resolved to '{resolved_ontology_file}'")
            args.ontology_terms_file = str(resolved_ontology_file)
        else:
            # If not found relative to project root, try relative to CWD (implicitly handled by Path.resolve() if not absolute)
            # However, if it was meant to be relative to src/, this simple resolve might not be enough.
            # For now, we assume project root or CWD is sufficient for this specific argument if relative.
            # An alternative is to resolve it against Path(__file__).parent if it's meant to be in src/ by default.
            cwd_resolved_ontology_file = Path(args.ontology_terms_file).resolve()
            if cwd_resolved_ontology_file.exists():
                 logger.debug(f"Ontology terms file '{args.ontology_terms_file}' resolved via CWD to '{cwd_resolved_ontology_file}'")
                 args.ontology_terms_file = str(cwd_resolved_ontology_file)
            else:
                 logger.warning(f"Ontology terms file '{args.ontology_terms_file}' not found at project root or CWD. Step 8 might fail if it relies on this relative path.")
    
    exit_code = run_pipeline(args)
    sys.exit(exit_code)

if __name__ == "__main__":
    # Note: The basicConfig for logging is now inside main() after args are parsed.
    main() 