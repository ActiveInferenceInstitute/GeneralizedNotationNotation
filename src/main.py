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
    --target-dir DIR        Target directory for GNN files (default: src/gnn/examples)
                            (Note: Individual scripts might target their specific folders e.g. src/mcp)
    --output-dir DIR        Directory to save outputs (default: ../output)
    --recursive             Recursively process directories (passed to relevant steps)
    --skip-steps LIST       Comma-separated list of steps to skip (e.g., "1_gnn,7_mcp" or "1,7")
    --only-steps LIST       Comma-separated list of steps to run (e.g., "4_gnn_type_checker,6_visualization")
    --verbose               Enable verbose output
    --quiet                 Disable verbose output, making it quiet (overrides the default verbose behavior)
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
import re # Added import for regular expressions
import subprocess

# --- Logger Setup ---
# The logger for the GNN Pipeline orchestrator itself.
# Configuration will be done by basicConfig in the __main__ block.
logger = logging.getLogger("GNN_Pipeline")
# --- End Logger Setup ---

def parse_arguments():
    # Calculate default paths relative to project root
    # Assuming main.py is in src/ and project root is its parent.
    project_root = Path(__file__).resolve().parent.parent
    
    default_output_dir = project_root / "output"
    # Default target_dir for GNN source files (e.g., .md files)
    # 'src/gnn/examples' is the typical location for example GNN files within this project structure.
    default_target_dir = project_root / "src" / "gnn" / "examples"
    
    default_ontology_terms_file = project_root / "src" / "ontology" / "act_inf_ontology_terms.json"

    parser = argparse.ArgumentParser(
        description="GNN Processing Pipeline: Orchestrates GNN file processing, analysis, and export.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=default_target_dir,
        help=f"Target directory for GNN source files (e.g., .md GNN specifications).\nDefault: {default_target_dir.relative_to(project_root) if default_target_dir.is_relative_to(project_root) else default_target_dir}"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Root directory to save all pipeline outputs.\nDefault: {default_output_dir.relative_to(project_root) if default_output_dir.is_relative_to(project_root) else default_output_dir}"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True, # Defaulting to True as it's common for processing
        help="Recursively process GNN files in subdirectories of the target directory."
    )
    parser.add_argument(
        "--no-recursive",
        dest='recursive',
        action='store_false',
        help="Disable recursive processing (process only top-level files in target directory)."
    )
    parser.add_argument(
        "--skip-steps",
        type=str,
        default="",
        help="Comma-separated list of step scripts or numbers to skip (e.g., \"1_gnn.py,7_mcp.py\" or \"1,7\")."
    )
    parser.add_argument(
        "--only-steps",
        type=str,
        default="",
        help="Comma-separated list of step scripts or numbers to run exclusively (e.g., \"4_gnn_type_checker.py,6_visualization.py\"). Overrides --skip-steps."
    )
    parser.add_argument(
        "--ontology-terms-file",
        type=Path,
        default=default_ontology_terms_file,
        help=f"Path to a JSON file defining valid ontological terms for validation in Step 8.\nDefault: {default_ontology_terms_file.relative_to(project_root) if default_ontology_terms_file.is_relative_to(project_root) else default_ontology_terms_file}"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True, # Make verbose True by default
        help="Enable verbose output (default: True). This is the default behavior."
    )
    parser.add_argument(
        "--quiet",
        action="store_false",
        dest="verbose", # Set args.verbose to False if --quiet is used
        help="Disable verbose output, making it quiet (overrides the default verbose behavior)."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict type checking mode (passed to 4_gnn_type_checker.py)."
    )
    parser.add_argument(
        "--estimate-resources",
        action="store_true",
        help="Estimate computational resources (passed to 4_gnn_type_checker.py)."
    )
    # Add a new argument for the pipeline summary file name/path
    parser.add_argument(
        "--pipeline-summary-file",
        type=Path,
        help="Custom name/path for the main pipeline summary report. If not set, defaults to '<output_dir>/pipeline_execution_summary.md'."
    )
    # Argument for LLM related operations (Step 11)
    parser.add_argument(
        "--llm-tasks",
        type=str,
        default="all", # Default to all tasks for Step 11
        help="Comma-separated list of LLM tasks to perform in Step 11 (e.g., 'summarize,explain'). 'all' for all defined tasks. Passed to 11_llm.py."
    )
    
    args = parser.parse_args()

    # Post-process the pipeline_summary_file argument
    if args.pipeline_summary_file is None:
        args.pipeline_summary_file = args.output_dir / "pipeline_execution_summary.md"
    else:
        # If a relative path is given, make it relative to the output_dir, or resolve if absolute
        if not args.pipeline_summary_file.is_absolute():
            args.pipeline_summary_file = args.output_dir / args.pipeline_summary_file
        # Ensure the parent directory for the summary file exists
        args.pipeline_summary_file.parent.mkdir(parents=True, exist_ok=True)


    return args

def get_pipeline_scripts():
    """Get all numbered pipeline scripts in sorted order based on the leading number."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use a general pattern to find potential scripts, then filter and sort with regex
    # This pattern finds files like 'anyname_anything.py' in the current directory.
    potential_scripts_pattern = os.path.join(current_dir, "*_*.py") 
    logger.debug(f"ℹ️ Discovering potential pipeline scripts using pattern: {potential_scripts_pattern}")
    
    all_candidate_files = glob.glob(potential_scripts_pattern)
    
    pipeline_scripts_info = [] # Store dicts of {'num': int, 'basename': str}
    
    # Regex to match basenames starting with one or more digits, followed by an underscore, and ending with .py
    script_name_regex = re.compile(r"^(\d+)_.*\.py$")

    for script_path in all_candidate_files:
        script_basename = os.path.basename(script_path)
        match = script_name_regex.match(script_basename)
        if match:
            script_num = int(match.group(1)) # Extract the number part
            pipeline_scripts_info.append({'num': script_num, 'basename': script_basename})
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"ℹ️ Matched script for pipeline: {script_basename} (Number: {script_num})")
        # Optionally log files that did not match if needed for debugging:
        # else:
            # if logger.isEnabledFor(logging.DEBUG):
            #      logger.debug(f"ℹ️ File {script_basename} did not match numbered pipeline script pattern.")

    # Sort scripts primarily by the extracted number, then by basename for tie-breaking (e.g. 1_foo.py vs 1_bar.py)
    pipeline_scripts_info.sort(key=lambda x: (x['num'], x['basename']))
    
    # Extract just the basenames in the now correctly sorted order
    sorted_script_basenames = [info['basename'] for info in pipeline_scripts_info]

    if logger.isEnabledFor(logging.DEBUG): 
        logger.debug(f"ℹ️ Found and sorted script basenames: {sorted_script_basenames}")
        # For more detailed debugging, log the full paths that were considered:
        # relevant_full_paths = [info['path'] for info in pipeline_scripts_info] # Assuming 'path' was stored
        # logger.debug(f"ℹ️ Corresponding full paths for sorted scripts: {relevant_full_paths}")
        
    return sorted_script_basenames

def run_pipeline(args):
    """Run the full GNN processing pipeline."""
    # Logger level is set by basicConfig based on args.verbose before this function is called.
    
    # --- ASCII Art Intro ---
    logger.info("""

 ██████╗ ███╗   ██╗███╗   ██╗  
██╔════╝ ████╗  ██║████╗  ██║  
██║  ███╗██╔██╗ ██║██╔██╗ ██║
██║   ██║██║╚██╗██║██║╚██╗██║    
╚██████╔╝██║ ╚████║██║ ╚████║    
                                                                                 
    Generalized Notation Notation Processing Pipeline 
    
    Version 0.1.0 ~ May 15, 2025
    
    https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation
            
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
            # Try to import and run the script's main function
            try:
                module_name = script_name_no_ext
                module = importlib.import_module(module_name)
                
                # Check if the module has a main function that accepts args
                if hasattr(module, "main") and callable(module.main):
                    logger.info(f"⚙️ {step_header} - STARTING")
                    processed_scripts_count += 1 # Increment when we actually attempt to run it
                    
                    # Prepare environment for the subprocess
                    sub_env = os.environ.copy()
                    sub_env["PYTHONNOUSERSITE"] = "1" # Isolate from user site-packages

                    venv_python_path = current_script_dir / ".venv" / "bin" / "python3" # Path to venv python
                    
                    # Determine venv site-packages path
                    _venv_site_packages_path_for_subproc = None
                    _venv_path_for_subproc = current_script_dir / ".venv"
                    if _venv_path_for_subproc.is_dir():
                        _lib_path_for_subproc = _venv_path_for_subproc / "lib"
                        if _lib_path_for_subproc.is_dir():
                            for _py_ver_dir in _lib_path_for_subproc.iterdir():
                                if _py_ver_dir.is_dir() and _py_ver_dir.name.startswith("python"):
                                    _site_pkg = _py_ver_dir / "site-packages"
                                    if _site_pkg.is_dir():
                                        _venv_site_packages_path_for_subproc = str(_site_pkg.resolve())
                                        break
                    
                    if _venv_site_packages_path_for_subproc:
                        sub_env["PYTHONPATH"] = _venv_site_packages_path_for_subproc
                        if args.verbose:
                            logger.debug(f"  Subprocess PYTHONPATH explicitly set to: {_venv_site_packages_path_for_subproc}")
                    else: # If venv site-packages not found, clear PYTHONPATH to avoid contamination
                        if "PYTHONPATH" in sub_env:
                            del sub_env["PYTHONPATH"]
                        if args.verbose:
                            logger.debug(f"  Subprocess PYTHONPATH { 'cleared' if _venv_site_packages_path_for_subproc is None and 'PYTHONPATH' in os.environ else 'was not set, no changes needed'}.")

                    # Construct arguments for the script based on what it might need
                    script_args_for_subprocess = []
                    
                    # Common arguments passed to most/all scripts
                    if hasattr(args, 'output_dir'): 
                        script_args_for_subprocess.extend(["--output-dir", str(args.output_dir)]) # Ensure path is string
                    if hasattr(args, 'verbose') and args.verbose: 
                        script_args_for_subprocess.append("--verbose")

                    # Script-specific arguments
                    if script_name_no_ext == "1_gnn.py":
                        if hasattr(args, 'target_dir'): script_args_for_subprocess.extend(["--target-dir", str(args.target_dir)])
                        if hasattr(args, 'recursive') and args.recursive: script_args_for_subprocess.append("--recursive")
                        # No need for --no-recursive if it's the default or handled by presence of --recursive

                    elif script_name_no_ext == "2_setup.py":
                        # 2_setup.py usually doesn't need many specific args from main.py,
                        # as it sets up the environment. Output_dir is already passed.
                        pass

                    elif script_name_no_ext == "3_tests.py":
                        # Pass the venv python path for tests to use the correct interpreter if needed
                        script_args_for_subprocess.extend(["--venv-python-path", str(venv_python_path)])
                        if hasattr(args, 'target_dir'): script_args_for_subprocess.extend(["--target-dir", str(args.target_dir)])

                    elif script_name_no_ext == "4_gnn_type_checker.py":
                        if hasattr(args, 'target_dir'): script_args_for_subprocess.extend(["--target-dir", str(args.target_dir)])
                        if hasattr(args, 'recursive') and args.recursive: script_args_for_subprocess.append("--recursive")
                        if hasattr(args, 'strict') and args.strict: script_args_for_subprocess.append("--strict")
                        if hasattr(args, 'estimate_resources'):
                            if args.estimate_resources:
                                script_args_for_subprocess.append("--estimate-resources")
                            # No --no-estimate-resources needed if false is default in script
                    
                    elif script_name_no_ext == "5_export.py":
                        if hasattr(args, 'target_dir'): script_args_for_subprocess.extend(["--target-dir", str(args.target_dir)])
                        if hasattr(args, 'recursive') and args.recursive: script_args_for_subprocess.append("--recursive")

                    elif script_name_no_ext == "6_visualization.py":
                         if hasattr(args, 'target_dir'): script_args_for_subprocess.extend(["--target-dir", str(args.target_dir)])
                         if hasattr(args, 'recursive') and args.recursive: script_args_for_subprocess.append("--recursive")
                    
                    elif script_name_no_ext == "7_mcp.py":
                        # 7_mcp.py uses overrides for its directories if provided, but typically takes them from its own logic relative to src/
                        # output_dir and verbose are already common args.
                        pass

                    elif script_name_no_ext == "8_ontology.py":
                        if hasattr(args, 'target_dir'): script_args_for_subprocess.extend(["--target-dir", str(args.target_dir)])
                        if hasattr(args, 'recursive') and args.recursive: script_args_for_subprocess.append("--recursive")
                        if hasattr(args, 'ontology_terms_file') and args.ontology_terms_file:
                            script_args_for_subprocess.extend(["--ontology-terms-file", str(args.ontology_terms_file)])
                    
                    elif script_name_no_ext == "9_render.py":
                        # 9_render expects output_dir (main pipeline output) to find gnn_exports, not target_dir (GNN sources)
                        # verbose is common. target_format not set by main.py, uses defaults in 9_render.
                        pass # output_dir and verbose are already common

                    elif script_name_no_ext == "10_execute.py":
                        # 10_execute also uses output_dir (main pipeline output) to find rendered simulators.
                        # verbose is common.
                        pass # output_dir and verbose are already common

                    elif script_name_no_ext == "11_llm.py":
                        if hasattr(args, 'target_dir'): script_args_for_subprocess.extend(["--target-dir", str(args.target_dir)])
                        if hasattr(args, 'recursive') and args.recursive: script_args_for_subprocess.append("--recursive")
                        if hasattr(args, 'llm_tasks'): script_args_for_subprocess.extend(["--llm-tasks", args.llm_tasks])

                    # Execute the script as a subprocess
                    script_full_path = current_script_dir / script_file
                    
                    if args.verbose:
                        # Construct a clean command string for logging, quoting paths with spaces if any
                        cmd_parts_for_log = [str(venv_python_path), str(script_full_path)] + \
                                            [str(arg) for arg in script_args_for_subprocess]
                        # Basic quoting for display, real subprocess call handles spaces correctly
                        cmd_log_str = ' '.join([f'"{part}"' if ' ' in part else part for part in cmd_parts_for_log])
                        logger.debug(f"  Running subprocess: {cmd_log_str}")
                        logger.debug(f"  Subprocess environment will include: PYTHONNOUSERSITE={sub_env.get('PYTHONNOUSERSITE')}, PYTHONPATH='{sub_env.get('PYTHONPATH', '')}'")

                    step_process = subprocess.run(
                        [str(venv_python_path), str(script_full_path)] + [str(arg) for arg in script_args_for_subprocess],
                        capture_output=True, text=True, check=False, cwd=current_script_dir,
                        env=sub_env
                    )

                    if step_process.returncode == 0:
                        successful_scripts_count += 1
                        logger.info(f"✅ {step_header} - COMPLETED successfully.")
                        if args.verbose: # Detailed success only in verbose
                             logger.debug(f"   (Script returned: {step_process.returncode})")
                             if step_process.stdout and step_process.stdout.strip():
                                 logger.debug(f"   --- {script_name_no_ext} STDOUT ---")
                                 logger.debug(step_process.stdout.strip())
                                 logger.debug(f"   --- END {script_name_no_ext} STDOUT ---")
                             if step_process.stderr and step_process.stderr.strip():
                                 logger.warning(f"   --- {script_name_no_ext} STDERR (on success) ---")
                                 logger.warning(step_process.stderr.strip())
                                 logger.warning(f"   --- END {script_name_no_ext} STDERR (on success) ---")
                    else: # Script failed
                        error_message = f"❌ {step_header} - FAILED (Reported exit code: {step_process.returncode})."
                        logger.error(error_message)
                        if step_process.stdout and step_process.stdout.strip():
                            logger.error(f"   --- {script_name_no_ext} STDOUT ---")
                            logger.error(step_process.stdout.strip())
                            logger.error(f"   --- END {script_name_no_ext} STDOUT ---")
                        if step_process.stderr and step_process.stderr.strip():
                            logger.error(f"   --- {script_name_no_ext} STDERR ---")
                            logger.error(step_process.stderr.strip())
                            logger.error(f"   --- END {script_name_no_ext} STDERR ---")
                        
                        failed_steps_details.append({'name': script_name_no_ext, 'reason': 'NonZeroExitCode', 'code': step_process.returncode})
                        
                        if is_critical_step:
                            logger.critical(f"🔥 This ({script_name_no_ext}) was a CRITICAL step. Halting pipeline.")
                            critical_step_failed = True
                            critical_step_name = script_name_no_ext
                            break # Exit loop to go to summary
                        else:
                            logger.warning(f"⚠️ This was a non-critical step. Pipeline will attempt to continue.")
                else:
                    logger.warning(f"⚠️ {step_header} (from {script_file}) - SKIPPED (no main() function found).")
                    skipped_due_to_no_main_count +=1
                
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
    total_attempted_to_run = processed_scripts_count # Correctly reflects scripts with main() that were not skipped by args
    total_skipped_by_args = skipped_due_to_args_count
    total_skipped_no_main = skipped_due_to_no_main_count
    
    logger.info(f"📊 Total Scripts Discovered: {total_discovered_scripts}")
    logger.info(f"   Attempted to Run:      {total_attempted_to_run}")
    logger.info(f"   Successfully Completed: {successful_scripts_count}")
    logger.info(f"   Skipped:                {total_skipped_by_args + total_skipped_no_main}")
    if (total_skipped_by_args + total_skipped_no_main) > 0:
        logger.info(f"     - Skipped by arguments (--skip-steps/--only-steps): {total_skipped_by_args}")
        logger.info(f"     - Skipped (no main() function):                   {total_skipped_no_main}")
    
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
        else:
            logger.info("🎉 PIPELINE FINISHED SUCCESSFULLY. All attempted steps completed without errors.")
    elif total_discovered_scripts == 0: # No scripts found at all
         logger.info("✅ PIPELINE FINISHED. No scripts were found to process.")
    else: # Mixed outcome
        if total_attempted_to_run == 0 and total_skipped_by_args > 0:
             logger.info("✅ PIPELINE FINISHED. All relevant steps were skipped by arguments.")
        elif total_attempted_to_run == 0 and total_skipped_no_main > 0 and total_skipped_by_args == 0:
            logger.info("✅ PIPELINE FINISHED. No runnable steps found (all discovered scripts were missing main() or skipped).")
        else: # Default case for completion with mixed results not already covered
             logger.info("✅ PIPELINE FINISHED.") 
    
    return final_exit_code

def main():
    args = parse_arguments()
    
    # --- Centralized Logging Configuration ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format, stream=sys.stdout) # Ensure logs go to stdout
    # --- End Centralized Logging Configuration ---

    # Log initial arguments, now that logging is configured.
    logger.debug(f"Parsed arguments: {args}")

    # Resolve paths after argument parsing and logging setup
    original_target_dir_str = str(args.target_dir)
    original_output_dir_str = str(args.output_dir)

    args.target_dir = Path(args.target_dir).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    
    if original_target_dir_str != str(args.target_dir):
        logger.debug(f"Original target_dir '{original_target_dir_str}' resolved to '{args.target_dir}'")
    if original_output_dir_str != str(args.output_dir):
        logger.debug(f"Original output_dir '{original_output_dir_str}' resolved to '{args.output_dir}'")

    if not Path(args.target_dir).exists():
        logger.warning(f"⚠️ Target directory {args.target_dir} does not exist. Some steps might fail or create it.")

    if args.ontology_terms_file:
        original_ontology_terms_file_str = str(args.ontology_terms_file)
        args.ontology_terms_file = Path(args.ontology_terms_file).resolve()
        if original_ontology_terms_file_str != str(args.ontology_terms_file):
            logger.debug(f"Ontology terms file '{original_ontology_terms_file_str}' resolved to '{args.ontology_terms_file}'")
        if not args.ontology_terms_file.exists():
            logger.warning(f"⚠️ Ontology terms file {args.ontology_terms_file} does not exist. Step 8 may proceed with default behavior or fail if it's critical.")
    
    # Ensure pipeline_summary_file is resolved
    original_pipeline_summary_file_str = str(args.pipeline_summary_file)
    args.pipeline_summary_file = Path(args.pipeline_summary_file).resolve()
    if original_pipeline_summary_file_str != str(args.pipeline_summary_file):
        logger.debug(f"Pipeline summary file '{original_pipeline_summary_file_str}' resolved to '{args.pipeline_summary_file}'")

    exit_code = run_pipeline(args)
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 