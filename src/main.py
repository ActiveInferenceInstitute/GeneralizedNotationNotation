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
- 9_render.py (Corresponds to render/ folder)
- 10_execute.py (Corresponds to execute/ folder)
- 11_llm.py (Corresponds to llm/ folder)
- 12_site.py (Corresponds to site/ folder, generates HTML summary site)
- 12_discopy.py (Corresponds to discopy_translator_module/ folder, generates DisCoPy diagrams from GNN)
- 13_discopy_jax_eval.py (Corresponds to discopy_translator_module/ folder, generates DisCoPy diagrams from GNN using JAX)
- 15_site.py (Corresponds to site/ folder, generates HTML summary site)


Usage:
    python main.py [options]
    
Options:
    --target-dir DIR        Target directory for GNN files (default: src/gnn/examples)
    --output-dir DIR        Directory to save outputs (default: ../output)
    --recursive / --no-recursive    Recursively process directories (default: --recursive)
    --skip-steps LIST       Comma-separated list of steps to skip (e.g., "1_gnn,7_mcp" or "1,7")
    --only-steps LIST       Comma-separated list of steps to run (e.g., "4_gnn_type_checker,6_visualization")
    --verbose               Enable verbose output
    --strict                Enable strict type checking mode (for 4_gnn_type_checker)
    --estimate-resources / --no-estimate-resources 
                            Estimate computational resources (for 4_gnn_type_checker) (default: --estimate-resources)
    --ontology-terms-file   Path to the ontology terms file (default: src/ontology/act_inf_ontology_terms.json)
    --llm-tasks LIST        Comma-separated list of LLM tasks to run for 11_llm.py 
                            (e.g., "summarize,explain_structure")
    --llm-timeout           Timeout in seconds for the LLM processing step (11_llm.py)
    --pipeline-summary-file FILE
                            Path to save the final pipeline summary report (default: output/pipeline_execution_summary.json)
    --site-html-filename NAME
                            Filename for the generated HTML summary site (for 15_site.py, saved in output-dir, default: gnn_pipeline_summary_site.html)
    --discopy-gnn-input-dir DIR
                            Directory containing GNN files for DisCoPy processing (for 12_discopy.py, default: src/gnn/examples or --target-dir if specified)
    --discopy-jax-gnn-input-dir DIR
                            Directory containing GNN files for DisCoPy JAX evaluation (13_discopy_jax_eval.py)
    --discopy-jax-seed      Seed for JAX PRNG in 13_discopy_jax_eval.py
    --recreate-venv         Recreate virtual environment even if it already exists (for 2_setup.py)
    --dev                   Also install development dependencies from requirements-dev.txt (for 2_setup.py)

"""

import os
import sys
import subprocess
import argparse
import glob
from pathlib import Path
import logging
import traceback
import re
import datetime # For pipeline summary timestamp
import json # For pipeline summary structured data
import time # For tracking subprocess execution time
import signal # For timeout handling
from typing import TypedDict, List, Union, Dict, Any, cast # Add typing for clarity, added cast

# Try to import the centralized logging utilities
try:
    from utils.logging_utils import (
        setup_standalone_logging, 
        silence_noisy_modules_in_console,
        set_verbose_mode
    )
except ImportError:
    # If logging_utils is not available, define placeholder functions
    setup_standalone_logging = None
    silence_noisy_modules_in_console = None
    set_verbose_mode = None

# --- Logger Setup ---
logger = logging.getLogger("GNN_Pipeline")
# --- End Logger Setup ---

# --- Default Pipeline Step Configuration ---
# This dictionary controls which pipeline steps are enabled by default.
# Steps are identified by their script basename.
# Command-line --skip-steps and --only-steps will override these defaults.
PIPELINE_STEP_CONFIGURATION: Dict[str, bool] = {
    "1_gnn.py": True,
    "2_setup.py": True,
    "3_tests.py": False,
    "4_gnn_type_checker.py": True,
    "5_export.py": True,
    "6_visualization.py": True,
    "7_mcp.py": True,
    "8_ontology.py": True,
    "9_render.py": True,
    "10_execute.py": True,
    "11_llm.py": True,
    "12_discopy.py": True,
    "13_discopy_jax_eval.py": True,
    "15_site.py": True, 
    # Add any new [number]_script_name.py here and set to True/False
}
# --- End Default Pipeline Step Configuration ---

# Define types for pipeline summary data
class StepLogData(TypedDict):
    step_number: int
    script_name: str
    status: str
    start_time: Union[str, None]
    end_time: Union[str, None]
    duration_seconds: Union[float, None] # Changed from int | str | None
    details: str
    stdout: str
    stderr: str

class PipelineRunData(TypedDict):
    start_time: str
    arguments: Dict[str, Any]
    steps: List[StepLogData]
    end_time: Union[str, None]
    overall_status: str

def parse_arguments():
    project_root = Path(__file__).resolve().parent.parent
    
    default_output_dir = project_root / "output"
    default_target_dir = project_root / "src" / "gnn" / "examples"
    default_ontology_terms_file = project_root / "src" / "ontology" / "act_inf_ontology_terms.json"
    default_pipeline_summary_file = default_output_dir / "pipeline_execution_summary.json"
    default_discopy_gnn_input_dir = default_target_dir # Default discopy input to general target dir

    parser = argparse.ArgumentParser(
        description="GNN Processing Pipeline: Orchestrates GNN file processing, analysis, and export.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=default_target_dir,
        help=(f"Target directory for GNN source files (e.g., .md GNN specifications).\\n"
              f"Default: {default_target_dir.relative_to(project_root) if default_target_dir.is_relative_to(project_root) else default_target_dir}")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=(f"Main directory to save all pipeline outputs.\\n"
              f"Default: {default_output_dir.relative_to(project_root) if default_output_dir.is_relative_to(project_root) else default_output_dir}")
    )
    parser.add_argument(
        "--recursive", default=True, action=argparse.BooleanOptionalAction,
        help="Recursively process GNN files in the target directory. Enabled by default. Use --no-recursive to disable."
    )
    parser.add_argument(
        "--skip-steps", default="",
        help='Comma-separated list of step numbers or names to skip (e.g., "1,7_mcp" or "1_gnn,7").'
    )
    parser.add_argument(
        "--only-steps", default="",
        help='Comma-separated list of step numbers or names to run exclusively (e.g., "4,6_visualization").'
    )
    parser.add_argument(
        "--verbose", 
        default=True, 
        action=argparse.BooleanOptionalAction, 
        help="Enable verbose output for the pipeline and its steps. On by default (use --no-verbose to disable)."
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Enable strict type checking mode (passed to 4_gnn_type_checker.py)."
    )
    parser.add_argument(
        "--estimate-resources", default=True, action=argparse.BooleanOptionalAction,
        help="Estimate computational resources (passed to 4_gnn_type_checker.py). Enabled by default."
    )
    parser.add_argument(
        "--ontology-terms-file",
        type=Path,
        default=default_ontology_terms_file,
        help=(f"Path to a JSON file defining valid ontological terms (for 8_ontology.py).\\n"
              f"Default: {default_ontology_terms_file.relative_to(project_root) if default_ontology_terms_file.is_relative_to(project_root) else default_ontology_terms_file}")
    )
    parser.add_argument(
        "--llm-tasks", default="all", type=str,
        help='Comma-separated list of LLM tasks for 11_llm.py (e.g., "overview,purpose,ontology"). Default: "all".'
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=60, # Default to 1 minute for the entire LLM script
        help="Timeout in seconds for the LLM processing step (11_llm.py). Default: 360"
    )
    parser.add_argument(
        "--pipeline-summary-file",
        type=Path,
        default=default_pipeline_summary_file,
        help=(f"Path to save the final pipeline summary report (JSON format).\n"
              f"Default: {default_pipeline_summary_file.relative_to(project_root) if default_pipeline_summary_file.is_relative_to(project_root) else default_pipeline_summary_file}")
    )
    parser.add_argument(
        "--site-html-filename",
        type=str,
        default="gnn_pipeline_summary_site.html",
        help=(f"Filename for the generated HTML summary site (for 15_site.py). It will be saved in the main output directory.\\n"
              f"Default: gnn_pipeline_summary_site.html")
    )
    parser.add_argument(
        "--discopy-gnn-input-dir",
        type=Path,
        default=None, # Will be set to default_target_dir or args.target_dir later
        help=(f"Directory containing GNN files for DisCoPy processing (12_discopy.py). \\n"
              f"If not set, uses the main --target-dir. Default if --target-dir is also default: {default_discopy_gnn_input_dir.relative_to(project_root) if default_discopy_gnn_input_dir.is_relative_to(project_root) else default_discopy_gnn_input_dir}")
    )
    parser.add_argument(
        "--discopy-jax-gnn-input-dir",
        type=Path,
        default=None, # Will be set to default_target_dir or args.target_dir later
        help=(f"Directory containing GNN files for DisCoPy JAX evaluation (13_discopy_jax_eval.py). \\n"
              f"If not set, uses the main --target-dir. Default if --target-dir is also default: {default_discopy_gnn_input_dir.relative_to(project_root) if default_discopy_gnn_input_dir.is_relative_to(project_root) else default_discopy_gnn_input_dir}")
    )
    parser.add_argument(
        "--discopy-jax-seed",
        type=int,
        default=0,
        help="Seed for JAX PRNG in 13_discopy_jax_eval.py. Default: 0."
    )
    parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="Recreate virtual environment even if it already exists (for 2_setup.py)."
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Also install development dependencies from requirements-dev.txt (for 2_setup.py)."
    )
    return parser.parse_args()

def get_pipeline_scripts(current_dir: Path) -> list[dict[str, int | str | Path]]:
    potential_scripts_pattern = current_dir / "*_*.py"
    logger.debug(f"ℹ️ Discovering potential pipeline scripts using pattern: {potential_scripts_pattern}")
    all_candidate_files = glob.glob(str(potential_scripts_pattern))
    
    pipeline_scripts_info: list[dict[str, int | str | Path]] = [] # Explicitly type hint
    script_name_regex = re.compile(r"^(\d+)_.*\.py$")

    for script_path_str in all_candidate_files:
        script_basename = os.path.basename(script_path_str)
        match = script_name_regex.match(script_basename)
        if match:
            script_num = int(match.group(1))
            pipeline_scripts_info.append({'num': script_num, 'basename': script_basename, 'path': Path(script_path_str)})
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"ℹ️ Matched script for pipeline: {script_basename} (Number: {script_num})")
    
    pipeline_scripts_info.sort(key=lambda x: (x['num'], x['basename']))
    sorted_script_basenames: list[str] = [str(info['basename']) for info in pipeline_scripts_info] # Explicitly cast to str and type hint

    if logger.isEnabledFor(logging.DEBUG): 
        logger.debug(f"ℹ️ Found and sorted script basenames: {sorted_script_basenames}")
    return pipeline_scripts_info

def get_venv_python(script_dir: Path) -> tuple[Path | None, Path | None]:
    venv_path = script_dir / ".venv"
    venv_python_path = None
    site_packages_path = None

    if venv_path.is_dir():
        potential_python_executables = [
            venv_path / "bin" / "python",
            venv_path / "bin" / "python3",
            venv_path / "Scripts" / "python.exe", # Windows
        ]
        for py_exec in potential_python_executables:
            if py_exec.exists() and py_exec.is_file():
                venv_python_path = py_exec
                logger.debug(f"🐍 Found virtual environment Python: {venv_python_path}")
                break
        
        lib_path = venv_path / "lib"
        if lib_path.is_dir():
            for python_version_dir in lib_path.iterdir():
                if python_version_dir.is_dir() and python_version_dir.name.startswith("python"):
                    current_site_packages = python_version_dir / "site-packages"
                    if current_site_packages.is_dir():
                        site_packages_path = current_site_packages
                        logger.debug(f"Found site-packages at: {site_packages_path}")
                        break
    
    if not venv_python_path:
        logger.warning("⚠️ Virtual environment Python not found. Using system Python. This may lead to issues if dependencies are not globally available.")
        venv_python_path = Path(sys.executable) # Fallback to current interpreter
    
    return venv_python_path, site_packages_path

def run_pipeline(args: argparse.Namespace):
    """
    Run the full GNN processing pipeline based on the provided arguments.
    
    This function:
    1. Discovers all numbered scripts in the src/ directory
    2. Determines which scripts to run based on skip/only flags
    3. Runs each script with appropriate arguments
    4. Generates a structured execution summary
    
    Args:
        args: The parsed command-line arguments
        
    Returns:
        Tuple of (exit_code, pipeline_run_data, all_scripts, overall_status)
    """
    current_dir = Path(__file__).resolve().parent
    all_scripts = get_pipeline_scripts(current_dir)
    
    # Prepare the summary report structure
    _pipeline_run_data_dict: Dict[str, Any] = {
        "start_time": datetime.datetime.now().isoformat(),
        "arguments": vars(args),
        "steps": [],
        "end_time": None,
        "overall_status": "SUCCESS"  # Will be updated if any step fails
    }
    
    # Parse skip/only steps to determine what to run
    scripts_to_run = []
    skip_steps = []
    only_steps = []
    
    if args.skip_steps:
        skip_steps = args.skip_steps.split(',')
    
    if args.only_steps:
        only_steps = args.only_steps.split(',')
        
    # Build list of scripts to run
    for script_info in all_scripts:
        script_basename = script_info['basename']
        script_num_str = str(script_info['num'])
        script_name_no_ext = os.path.splitext(script_basename)[0]
        
        # Check if this script is enabled by default
        is_enabled_by_default = PIPELINE_STEP_CONFIGURATION.get(script_basename, True)
        
        # Skip logic: Skip if explicitly listed in skip_steps by number or name
        should_skip = (
            script_num_str in skip_steps or
            script_basename in skip_steps or
            script_name_no_ext in skip_steps or
            not is_enabled_by_default
        )
        
        # Only logic: Only run if explicitly listed in only_steps by number or name
        # If only_steps is empty, this doesn't apply
        should_only_run = not only_steps or (
            script_num_str in only_steps or
            script_basename in only_steps or
            script_name_no_ext in only_steps
        )
        
        if not should_skip and should_only_run:
            scripts_to_run.append(script_info)
    
    # Check if no scripts are configured to run
    if not scripts_to_run:
        if only_steps:
            logger.warning(f"⚠️ No scripts match the only-steps filter: {args.only_steps}")
        elif skip_steps:
            logger.warning(f"⚠️ All scripts were skipped by skip-steps filter: {args.skip_steps}")
        else:
            logger.warning("⚠️ No scripts are configured to run in PIPELINE_STEP_CONFIGURATION")
        
        # Return with warning status if no scripts run
        _pipeline_run_data_dict["end_time"] = datetime.datetime.now().isoformat()
        _pipeline_run_data_dict["overall_status"] = "SUCCESS_WITH_WARNINGS"
        return 0, cast(PipelineRunData, _pipeline_run_data_dict), all_scripts, "SUCCESS_WITH_WARNINGS"
    
    # Log what we're about to run
    logger.info(f"📋 Will execute {len(scripts_to_run)} pipeline steps:")
    for idx, script_info in enumerate(scripts_to_run, 1):
        script_num = script_info['num']
        script_basename = script_info['basename']
        logger.info(f"  {idx}. {script_num}: {script_basename}")
    
    # Get the Python executable to use for the scripts
    logger.debug("🔍 Determining Python executable for subprocess calls...")
    venv_python, venv_python_no_activation = get_venv_python(current_dir)
    
    if venv_python:
        logger.debug(f"✓ Using virtual environment Python: {venv_python}")
    elif venv_python_no_activation:
        logger.debug(f"⚠️ Using Python without virtualenv activation: {venv_python_no_activation}")
    else:
        logger.debug("⚠️ No specific Python found, will use system default")
    
    # Determine which Python to use based on availability
    python_to_use = venv_python or venv_python_no_activation or "python"
    logger.debug(f"🐍 Selected Python for subprocess calls: {python_to_use}")
    
    # Set up command-line flags to pass to each script
    base_args = []
    
    # Define script-specific argument support
    script_supported_args = {
        # Common arguments supported by all scripts
        "all": ["target-dir", "output-dir", "verbose"],
        
        # Script-specific supported arguments 
        "1_gnn.py": ["recursive"],
        "4_gnn_type_checker.py": ["strict", "estimate-resources"],
        "8_ontology.py": ["ontology-terms-file"],
        "11_llm.py": ["llm-tasks", "llm-timeout"],
        "12_discopy.py": ["discopy-gnn-input-dir"],
        "13_discopy_jax_eval.py": ["discopy-jax-gnn-input-dir", "discopy-jax-seed"],
        "15_site.py": ["site-html-filename"],
        "2_setup.py": ["recreate-venv", "dev"],
    }
    
    # Map of arguments that support negation (--no-X format)
    negatable_args = {
        "recursive": True,
        "verbose": True,
        "estimate-resources": True
    }
    
    # These arguments should be passed to all scripts
    if args.target_dir:
        base_args.extend(["--target-dir", str(args.target_dir)])
    
    if args.output_dir:
        base_args.extend(["--output-dir", str(args.output_dir)])
    
    # Handle verbose flag separately - all scripts support it, but we need to check if they support negation
    if args.verbose:
        base_args.append("--verbose")
    elif hasattr(args, 'verbose') and args.verbose is False:
        # Only add --no-verbose if we're sure it's supported
        base_args.append("--verbose")  # Default to plain --verbose flag for compatibility
    
    # Map of step script names to their additional args
    step_specific_args = {}
    
    # Helper function to check if script supports an argument
    def script_supports_arg(script_name, arg_name, include_negated=False):
        # Check if arg is in common args for all scripts
        if arg_name in script_supported_args.get("all", []):
            return True
        
        # Check if arg is in script-specific args
        return arg_name in script_supported_args.get(script_name, [])
    
    # For step 1 (gnn)
    if script_supports_arg("1_gnn.py", "recursive"):
        if args.recursive:
            step_specific_args["1_gnn.py"] = ["--recursive"]
    
    # For step 4 (gnn_type_checker)
    gnn_type_checker_args = []
    if args.strict and script_supports_arg("4_gnn_type_checker.py", "strict"):
        gnn_type_checker_args.append("--strict")
    
    if hasattr(args, 'estimate_resources') and script_supports_arg("4_gnn_type_checker.py", "estimate-resources"):
        if args.estimate_resources:
            gnn_type_checker_args.append("--estimate-resources")
        else:
            gnn_type_checker_args.append("--no-estimate-resources")
    
    if gnn_type_checker_args:
        step_specific_args["4_gnn_type_checker.py"] = gnn_type_checker_args
    
    # For step 8 (ontology)
    ontology_args = []
    if args.ontology_terms_file and script_supports_arg("8_ontology.py", "ontology-terms-file"):
        ontology_args.extend(["--ontology-terms-file", str(args.ontology_terms_file)])
    
    if ontology_args:
        step_specific_args["8_ontology.py"] = ontology_args
    
    # For step 11 (llm)
    llm_args = []
    if args.llm_tasks and script_supports_arg("11_llm.py", "llm-tasks"):
        llm_args.extend(["--llm-tasks", args.llm_tasks])
    
    if args.llm_timeout and script_supports_arg("11_llm.py", "llm-timeout"):
        llm_args.extend(["--llm-timeout", str(args.llm_timeout)])
    
    if llm_args:
        step_specific_args["11_llm.py"] = llm_args
    
    # For step 12 (discopy)
    discopy_args = []
    if args.discopy_gnn_input_dir and script_supports_arg("12_discopy.py", "discopy-gnn-input-dir"):
        discopy_args.extend(["--discopy-gnn-input-dir", str(args.discopy_gnn_input_dir)])
    
    if discopy_args:
        step_specific_args["12_discopy.py"] = discopy_args
    
    # For step 13 (discopy_jax_eval)
    discopy_jax_args = []
    if args.discopy_jax_gnn_input_dir and script_supports_arg("13_discopy_jax_eval.py", "discopy-jax-gnn-input-dir"):
        discopy_jax_args.extend(["--discopy-jax-gnn-input-dir", str(args.discopy_jax_gnn_input_dir)])
    
    if hasattr(args, 'discopy_jax_seed') and script_supports_arg("13_discopy_jax_eval.py", "discopy-jax-seed"):
        discopy_jax_args.extend(["--discopy-jax-seed", str(args.discopy_jax_seed)])
    
    if discopy_jax_args:
        step_specific_args["13_discopy_jax_eval.py"] = discopy_jax_args
    
    # For step 15 (site)
    site_args = []
    if args.site_html_filename and script_supports_arg("15_site.py", "site-html-filename"):
        site_args.extend(["--site-html-filename", args.site_html_filename])
    
    if site_args:
        step_specific_args["15_site.py"] = site_args
    
    # For step 2 (setup)
    setup_args = []
    if args.recreate_venv and script_supports_arg("2_setup.py", "recreate-venv"):
        setup_args.append("--recreate-venv")
    
    if args.dev and script_supports_arg("2_setup.py", "dev"):
        setup_args.append("--dev")
    
    if setup_args:
        step_specific_args["2_setup.py"] = setup_args
    
    # Execute each script
    overall_status = "SUCCESS"
    
    logger.info("\n  ██████╗  ███╗   ██╗ ███╗   ██╗") 
    logger.info("  ██╔════╝ ████╗  ██║ ████╗  ██║")
    logger.info("  ██║  ███╗██╔██╗ ██║ ██╔██╗ ██║")
    logger.info("  ██║   ██║██║╚██╗██║ ██║╚██╗██║")
    logger.info("  ╚██████╔╝██║ ╚████║ ██║ ╚████║")
    logger.info("   ╚═════╝ ╚═╝  ╚═══╝ ╚═╝  ╚═══╝")
    logger.info("  Running GNN Processing Pipeline...\n")
    
    for idx, script_info in enumerate(scripts_to_run, 1):
        script_num = script_info['num']
        script_basename = script_info['basename']
        script_path = script_info['path']
        script_name_no_ext = os.path.splitext(script_basename)[0]
        
        # Check if this is a critical script (failure halts pipeline)
        # Currently, only setup.py is considered critical.
        is_critical_step = script_basename == "2_setup.py"
        
        # Set longer timeout for setup steps (20 minutes) and standard timeout for others (1 minute)
        # Setup step timeout increased from 10 to 20 minutes to account for slower connections/machines
        step_timeout = 1200 if script_basename == "2_setup.py" else 60
        
        step_header = f"Step {idx}/{len(scripts_to_run)} ({script_num}: {script_basename})"
        logger.info(f"🚀 Starting {step_header}")
        
        # Create the log data structure for this step
        step_log_data: StepLogData = {
            "step_number": script_num,
            "script_name": script_basename,
            "status": "SKIPPED",  # Will be updated after execution
            "start_time": datetime.datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "details": "",
            "stdout": "",
            "stderr": ""
        }
        
        # Build the command with script-specific args
        full_args = base_args.copy()
        
        # Add script-specific arguments only if this script supports them
        if script_basename in step_specific_args:
            script_args = step_specific_args[script_basename]
            # Only add the arguments if they're actually in the list
            if script_args:
                full_args.extend(script_args)
        
        # Log the full command for debugging
        cmd = [str(python_to_use), str(script_path)] + full_args
        logger.debug(f"📋 Executing command: {' '.join(cmd)}")
        
        # For slow steps, provide more detailed progress tracking
        if script_basename == "2_setup.py":
            logger.info(f"⏳ Setting up environment and dependencies (timeout: {step_timeout}s). This may take several minutes...")
            logger.info("   The process will display progress logs while running.")
            # Ensure this gets displayed immediately
            sys.stdout.flush()
        
        try:
            # Create a process to run the script
            start_time = time.time()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Prepare to capture output
            stdout_lines = []
            stderr_lines = []
            
            # Helper to process a stream and log if needed
            def process_stream_line(line, is_stderr, record_to, log_level=None):
                line = line.rstrip('\n')
                if line:
                    record_to.append(line)
                    
                    # Only log to console if we should show this level of detail
                    should_log_to_console = True
                    
                    # Filter out low-level messages for console output
                    # For stdout, we only want to show if the user requested verbose
                    # For stderr, filter in non-verbose mode to only show errors/warnings
                    if not args.verbose:
                        if not is_stderr:
                            # In non-verbose mode, don't show stdout at all
                            should_log_to_console = False
                        else:
                            # For stderr in non-verbose mode, only show ERROR/WARNING/CRITICAL lines
                            if not any(x in line for x in ["ERROR", "WARNING", "CRITICAL", "FATAL"]):
                                should_log_to_console = False
                    
                    # For setup.py, we want to show important messages regardless of verbose mode
                    if script_basename == "2_setup.py":
                        # Important progress indicators
                        if any(x in line for x in ["Installing", "✅", "Collecting", "Successfully", "pip upgraded"]):
                            should_log_to_console = True
                    
                    if should_log_to_console and log_level is not None:
                        logger.log(log_level, f"    [{'STDERR' if is_stderr else 'STDOUT'}] {line}")
                        # Ensure setup.py output is immediately visible
                        if script_basename == "2_setup.py":
                            sys.stdout.flush()
            
            # Track last update time for progress reporting
            last_output_time = time.time()
            last_progress_report = time.time()
            has_reported_progress = False
            
            # Process stdout and stderr streams in real-time
            while True:
                # Check for timeout - but only after allowing some startup time
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time > step_timeout:
                    logger.warning(f"⚠️ {step_header} has been running for {elapsed_time:.1f} seconds, exceeding timeout of {step_timeout} seconds.")
                    logger.warning(f"   Terminating process...")
                    process.terminate()
                    time.sleep(0.5)  # Give it a chance to terminate gracefully
                    if process.poll() is None:
                        process.kill()  # Force kill if still running
                    raise subprocess.TimeoutExpired(cmd, step_timeout)
                
                # For long-running processes, display progress indicators
                if script_basename == "2_setup.py" and current_time - last_progress_report > 30:
                    if process.poll() is None:  # Still running
                        logger.info(f"   ⏳ {script_basename} still running ({elapsed_time:.1f}s)... Please wait.")
                        last_progress_report = current_time
                        has_reported_progress = True
                # For setup.py, provide even more reassurance with specific task messages
                elif script_basename == "2_setup.py" and current_time - last_progress_report > 30:
                    # Rotate through different messages to make it clear the process is still running
                    messages = [
                        "Setting up virtual environment and dependencies...",
                        "Installing Python packages (this may take a while)...",
                        "Resolving dependency conflicts...",
                        "Building wheels for dependencies...",
                        "Processing Python packages...",
                        "Dependency installation ongoing..."
                    ]
                    message_index = int((current_time - start_time) / 30) % len(messages)
                    logger.info(f"   ⏳ {messages[message_index]} (elapsed: {elapsed_time:.1f}s)")
                    last_progress_report = current_time
                    has_reported_progress = True
                
                # Check for stdout data with timeout
                stdout_ready = process.stdout.readable()
                if stdout_ready:
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        # Use the same filtering logic as above
                        should_log = True
                        if not args.verbose:
                            # Default is to show important outputs
                            should_log = True
                            
                            # Filter out specific patterns that are too verbose
                            if "Content snippet for" in stdout_line or "where StateSpaceBlock was not found" in stdout_line:
                                should_log = False
                            elif "Content snippet for" in stdout_line or "where Connections was not found" in stdout_line:
                                should_log = False
                            # Filter additional verbose debug messages from setup
                            elif "Reading metadata" in stdout_line or "removing temporary" in stdout_line:
                                should_log = False
                            elif "Building wheel" in stdout_line or "Created wheel" in stdout_line:
                                should_log = False
                            # Show high-importance setup messages regardless of verbose mode
                            elif script_basename == "2_setup.py" and any(x in stdout_line for x in ["Installing", "Still installing", "✅", "Stage", "took", "elapsed", "🎉"]):
                                should_log = True
                        
                        # Process the line based on our should_log decision
                        if should_log:
                            logger.info(f"    [STDOUT] {stdout_line.strip()}")
                        elif args.verbose:
                            logger.debug(f"    [STDOUT] {stdout_line.strip()}")
                        
                        # Always record the line in our captured output, even if not logged to console
                        stdout_lines.append(stdout_line.strip())
                        last_output_time = time.time()

                # Check for stderr data with timeout
                stderr_ready = process.stderr.readable()
                if stderr_ready:
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        # Use the same filtering logic as above
                        should_log_error = True
                        if "warn(" in stderr_line and "DeprecationWarning" in stderr_line:
                            should_log_error = False
                        elif "pyproject.toml" in stderr_line and "does not comply" in stderr_line:
                            should_log_error = False
                        # If it's a setup.py step and contains an error, always show it
                        elif script_basename == "2_setup.py" and any(x in stderr_line for x in ["ERROR", "Failed", "❌"]):
                            should_log_error = True
                            
                        if should_log_error:
                            logger.warning(f"    [STDERR] {stderr_line.strip()}")
                        elif args.verbose:
                            logger.debug(f"    [STDERR] {stderr_line.strip()}")
                        
                        stderr_lines.append(stderr_line.strip())
                        last_output_time = time.time()
                
                # If process has completed and no more data in pipes, break
                if process.poll() is not None:
                    # One last check for any remaining output
                    while process.stdout.readable():
                        stdout_line = process.stdout.readline()
                        if not stdout_line:
                            break
                        
                        # Use the same filtering logic as above
                        should_log = True
                        if not args.verbose:
                            # Default is to show important outputs
                            should_log = True
                            
                            # Filter out specific patterns that are too verbose
                            if "Content snippet for" in stdout_line or "where StateSpaceBlock was not found" in stdout_line:
                                should_log = False
                            elif "Content snippet for" in stdout_line or "where Connections was not found" in stdout_line:
                                should_log = False
                            # Filter additional verbose debug messages from setup
                            elif "Reading metadata" in stdout_line or "removing temporary" in stdout_line:
                                should_log = False
                            elif "Building wheel" in stdout_line or "Created wheel" in stdout_line:
                                should_log = False
                            # Show high-importance setup messages regardless of verbose mode
                            elif script_basename == "2_setup.py" and any(x in stdout_line for x in ["Installing", "Still installing", "✅", "Stage", "took", "elapsed", "🎉"]):
                                should_log = True
                        
                        # Process the line based on our should_log decision
                        if should_log:
                            logger.info(f"    [STDOUT] {stdout_line.strip()}")
                        elif args.verbose:
                            logger.debug(f"    [STDOUT] {stdout_line.strip()}")
                        
                        # Always record the line in our captured output, even if not logged to console
                        stdout_lines.append(stdout_line.strip())
                    
                    while process.stderr.readable():
                        stderr_line = process.stderr.readline()
                        if not stderr_line:
                            break
                        
                        # Use the same filtering logic as above
                        should_log_error = True
                        if "warn(" in stderr_line and "DeprecationWarning" in stderr_line:
                            should_log_error = False
                        elif "pyproject.toml" in stderr_line and "does not comply" in stderr_line:
                            should_log_error = False
                        # If it's a setup.py step and contains an error, always show it
                        elif script_basename == "2_setup.py" and any(x in stderr_line for x in ["ERROR", "Failed", "❌"]):
                            should_log_error = True
                            
                        if should_log_error:
                            logger.warning(f"    [STDERR] {stderr_line.strip()}")
                        elif args.verbose:
                            logger.debug(f"    [STDERR] {stderr_line.strip()}")
                        
                        stderr_lines.append(stderr_line.strip())
                    
                    break
                
                # If no output for more than 15 seconds but process is still running,
                # provide more frequent reassurance messages for long-running steps
                if (current_time - last_output_time > 15 and 
                    current_time - last_progress_report > 15 and
                    script_basename == "2_setup.py"):
                    logger.info(f"   ⏳ {script_basename} still running ({elapsed_time:.1f}s) but no recent output. Please wait...")
                    last_progress_report = current_time
                    has_reported_progress = True
                    sys.stdout.flush()  # Ensure progress is immediately visible
                # For setup.py, provide even more reassurance with specific task messages
                elif script_basename == "2_setup.py" and current_time - last_progress_report > 30:
                    # Rotate through different messages to make it clear the process is still running
                    messages = [
                        "Setting up virtual environment and dependencies...",
                        "Installing Python packages (this may take a while)...",
                        "Resolving dependency conflicts...",
                        "Building wheels for dependencies...",
                        "Processing Python packages...",
                        "Dependency installation ongoing..."
                    ]
                    message_index = int((current_time - start_time) / 30) % len(messages)
                    logger.info(f"   ⏳ {messages[message_index]} (elapsed: {elapsed_time:.1f}s)")
                    last_progress_report = current_time
                    has_reported_progress = True
                    sys.stdout.flush()  # Ensure progress is immediately visible
                
                # Small sleep to avoid CPU spinning
                time.sleep(0.1)
            
            # Capture any remaining output - this should be handled by the loop above
            # but this is a fallback just in case
            remaining_stdout, remaining_stderr = process.communicate()
            if remaining_stdout:
                for line in remaining_stdout.splitlines():
                    process_stream_line(line, False, stdout_lines, 
                                        logging.DEBUG if args.verbose else None)
            
            if remaining_stderr:
                for line in remaining_stderr.splitlines():
                    process_stream_line(line, True, stderr_lines, 
                                        logging.WARNING)
            
            # Update the log data with captured output
            step_log_data["stdout"] = "\n".join(stdout_lines)
            step_log_data["stderr"] = "\n".join(stderr_lines)
            
            # Get the return code
            return_code = process.returncode
            end_time = time.time()
            duration = end_time - start_time
            
            if return_code == 0:
                step_log_data["status"] = "SUCCESS"
                if has_reported_progress:
                    logger.info(f"✅ {step_header} - COMPLETED successfully in {duration:.1f} seconds.")
                else:
                    logger.info(f"✅ {step_header} - COMPLETED successfully.")
                logger.info("") # Add spacing after step completion
                
                # For non-verbose successful runs, if there's any output, log a summary.
                # For verbose runs, stdout is already streamed at DEBUG.
                if not args.verbose and stdout_lines:
                    # Only log the first 2 lines of stdout summary, and only at DEBUG level
                    summary = stdout_lines[:2]
                    if len(stdout_lines) > 2:
                        summary.append(f"... ({len(stdout_lines)-2} more lines, see log file)")
                    logger.debug(f"   [{script_name_no_ext}-STDOUT] {' | '.join(summary)}")
                
                # If a successful step wrote to stderr, log it at WARNING for visibility
                if stderr_lines:
                    # Only show serious stderr messages in console for successful steps
                    # (This is important for keeping console output clean)
                    error_messages = [line for line in stderr_lines if 
                                     "ERROR" in line or "CRITICAL" in line or 
                                     "FATAL" in line or "WARN" in line]
                    
                    if error_messages:
                        logger.warning(f"   [{script_name_no_ext}] Step completed successfully but had {len(error_messages)} error/warning messages in stderr")
                        for err_msg in error_messages[:3]:  # Show only first 3 errors
                            logger.warning(f"   [{script_name_no_ext}-STDERR] {err_msg}")
                        if len(error_messages) > 3:
                            logger.warning(f"   [{script_name_no_ext}-STDERR] ... ({len(error_messages)-3} more errors/warnings, see log file)")
            else:
                step_log_data["status"] = "FAILED_NONZERO_EXIT"
                step_log_data["details"] = f"Process exited with code {return_code}"
                logger.error(f"❌ {step_header} - FAILED with exit code {return_code} after {duration:.1f} seconds.")
                logger.info("") # Add spacing after step completion
                overall_status = "FAILED"
                if is_critical_step:
                    logger.critical(f"🔥 Critical step {script_name_no_ext} failed. Halting pipeline.")
                    step_log_data["details"] += " Critical step failure, pipeline halted."
                    _pipeline_run_data_dict["steps"].append(step_log_data)
                    break
        
        except subprocess.TimeoutExpired:
            end_time = time.time()
            duration = end_time - start_time
            step_log_data["status"] = "FAILED_TIMEOUT"
            step_log_data["details"] = f"Process timed out after {duration:.1f} seconds (limit: {step_timeout}s)"
            logger.error(f"❌ {step_header} - FAILED due to timeout after {duration:.1f} seconds.")
            logger.info("") # Add spacing after step completion
            overall_status = "FAILED"
            
            if is_critical_step:
                logger.critical(f"🔥 Critical step {script_name_no_ext} timed out. Halting pipeline.")
                step_log_data["details"] += " Critical step timeout, pipeline halted."
                _pipeline_run_data_dict["steps"].append(step_log_data)
                break

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            step_log_data["status"] = "ERROR_UNHANDLED_EXCEPTION"
            step_log_data["details"] = f"Unhandled exception after {duration:.1f} seconds: {str(e)}"
            logger.error(f"❌ Unhandled exception in {step_header}: {e}")
            logger.info("") # Add spacing after step completion
            logger.debug(traceback.format_exc())
            overall_status = "FAILED"
            if is_critical_step:
                logger.critical(f"🔥 Critical step {script_name_no_ext} failed due to unhandled exception. Halting pipeline.")
                step_log_data["details"] += " Critical step failure, pipeline halted."
                _pipeline_run_data_dict["steps"].append(step_log_data)
                break
        
        finally:
            step_log_data["end_time"] = datetime.datetime.now().isoformat()
            if step_log_data["start_time"]:
                duration = datetime.datetime.fromisoformat(step_log_data["end_time"]) - datetime.datetime.fromisoformat(step_log_data["start_time"])
                step_log_data["duration_seconds"] = duration.total_seconds()
            _pipeline_run_data_dict["steps"].append(step_log_data)

    _pipeline_run_data_dict["end_time"] = datetime.datetime.now().isoformat()
    _pipeline_run_data_dict["overall_status"] = overall_status
    # Log a brief summary before returning from run_pipeline
    logger.info(f"🏁 Pipeline processing completed. Overall Status: {overall_status}")

    return (0 if overall_status in ["SUCCESS", "SUCCESS_WITH_WARNINGS"] else 1), cast(PipelineRunData, _pipeline_run_data_dict), all_scripts, overall_status

def main():
    # Configure logging early. If GNN_Pipeline or root logger has no handlers,
    # set up a basic one. The user's traceback suggests handlers are present,
    # so this configuration is for robustness, especially if run in different contexts.
    pipeline_logger = logging.getLogger("GNN_Pipeline") # Use the specific logger

    args = parse_arguments()

    # --- Central Logging Configuration ---
    log_dir = args.output_dir / "logs"
    
    # Setup logging using our new utility if available
    if setup_standalone_logging:
        # Ensure logs directory exists
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error creating log directory: {e}", file=sys.stderr)
            # Fall back to no file logging if directory creation fails
            log_dir = None
        
        # Set up logging with different console/file levels
        console_level = logging.DEBUG if args.verbose else logging.INFO
        file_level = logging.DEBUG  # Always log detailed info to file
        
        # Configure the GNN_Pipeline logger
        setup_standalone_logging(
            level=min(console_level, file_level),
            logger_name="GNN_Pipeline",
            output_dir=log_dir,
            log_filename="pipeline.log",
            console_level=console_level,
            file_level=file_level
        )
        
        # Silence noisy modules in console but keep them in the log file
        if silence_noisy_modules_in_console:
            silence_noisy_modules_in_console()
            
    else:
        # Legacy fallback configuration
        if not pipeline_logger.hasHandlers() and not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO, # Default level
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt=None, # Explicitly use default to get milliseconds
                stream=sys.stdout
            )
            pipeline_logger.info("Initialized basic logging config as no handlers were found for GNN_Pipeline or root.")

        # Quieten noisy dependency loggers (PIL, Matplotlib) unconditionally
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        # Add other noisy libraries here if needed, e.g.:
        # logging.getLogger('some_other_library').setLevel(logging.WARNING)

        # --- Legacy File Handler Setup (after args are parsed) ---
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            # Add a file handler to the GNN_Pipeline logger
            log_file_path = log_dir / "pipeline.log"
            file_handler = logging.FileHandler(log_file_path, mode='w') # 'w' to overwrite each run
            # Use the same format as basicConfig, explicitly ensure milliseconds
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt=None)
            file_handler.setFormatter(file_formatter)
            # The level of the file_handler will be determined by the pipeline_logger's effective level,
            # which is set below based on args.verbose.
            pipeline_logger.addHandler(file_handler)
            # Log this message *after* the handler is added so it goes to the file too.
            pipeline_logger.info(f"File logging configured to: {log_file_path}") 
        except Exception as e:
            # Log this error to console, as file handler might have failed.
            temp_error_logger = logging.getLogger("GNN_Pipeline.FileHandlerSetup")
            # Ensure this specific error message can make it to console/stderr if main logger isn't fully working
            if not temp_error_logger.handlers:
                err_handler = logging.StreamHandler(sys.stderr)
                # Use a more specific format for this temp logger to distinguish its origin
                err_formatter = logging.Formatter('%(asctime)s - %(name)s - [TEMP_SETUP_ERROR] - %(levelname)s - %(message)s')
                err_handler.setFormatter(err_formatter)
                temp_error_logger.addHandler(err_handler)
                temp_error_logger.propagate = False # Don't double log this specific error message
            temp_error_logger.error(f"Failed to configure file logging to {log_dir / 'pipeline.log'}: {e}. Continuing with console logging only.")
        # --- End Legacy File Handler Setup ---

        # Configure logger level based on verbose flag AFTER parsing args
        if args.verbose:
            pipeline_logger.setLevel(logging.DEBUG)
            # Propagate level to handlers if they exist and basicConfig wasn't just called
            handlers_to_update = list(pipeline_logger.handlers) + list(logging.getLogger().handlers)
            for handler in handlers_to_update:
                # Only set level if handler's current level is not effectively DEBUG or lower
                if handler.level == 0 or handler.level > logging.DEBUG: # 0 means NOTSET, effectively inherits.
                    current_level_name = logging.getLevelName(handler.level)
                    logger.debug(f"Updating handler {type(handler).__name__} level from {current_level_name} to DEBUG")
                    handler.setLevel(logging.DEBUG)
            # For Popen streaming, we use GNN_Pipeline's INFO and ERROR, so DEBUG on GNN_Pipeline is fine.
        else:
            pipeline_logger.setLevel(logging.INFO)
            # Propagate level to handlers
            handlers_to_update_info = list(pipeline_logger.handlers) + list(logging.getLogger().handlers)
            for handler in handlers_to_update_info:
                if handler.level == 0 or handler.level > logging.INFO:
                    current_level_name = logging.getLevelName(handler.level)
                    logger.debug(f"Updating handler {type(handler).__name__} level from {current_level_name} to INFO")
                    handler.setLevel(logging.INFO)

    pipeline_logger.debug("Logger level configured based on verbosity.")

    # --- Defensive Path Conversion ---
    # Ensure critical path arguments are pathlib.Path objects.
    # argparse with type=Path should handle this, but this adds robustness.
    path_args_to_check = [
        'output_dir', 'target_dir', 'ontology_terms_file', 'pipeline_summary_file'
    ]

    for arg_name in path_args_to_check:
        if not hasattr(args, arg_name):
            pipeline_logger.debug(f"Argument --{arg_name.replace('_', '-')} not present in args namespace.")
            continue

        arg_value = getattr(args, arg_name)
        
        # Only proceed if arg_value is not None. If it's None, it might be an optional Path not provided.
        if arg_value is not None and not isinstance(arg_value, Path):
            pipeline_logger.warning(
                f"Argument --{arg_name.replace('_', '-')} was unexpectedly a {type(arg_value).__name__} "
                f"(value: '{arg_value}') instead of pathlib.Path. Converting explicitly. "
                "This might indicate an issue with argument parsing configuration or an external override."
            )
            try:
                setattr(args, arg_name, Path(arg_value))
            except TypeError as e:
                pipeline_logger.error(
                    f"Failed to convert argument --{arg_name.replace('_', '-')} (value: '{arg_value}') to Path: {e}. "
                    "This could be due to an unsuitable value for a path."
                )
                # If a critical path like output_dir fails conversion, it's a fatal error for the script's purpose.
                if arg_name in ['output_dir', 'target_dir']:
                    pipeline_logger.critical(f"Critical path argument --{arg_name.replace('_', '-')} could not be converted to Path. Exiting.")
                    sys.exit(1)
        elif arg_value is None and arg_name in ['output_dir', 'target_dir']: # These should always have a default Path value.
             pipeline_logger.critical(
                f"Critical path argument --{arg_name.replace('_', '-')} is None after parsing. "
                "This indicates a problem with default value setup in argparse. Exiting."
             )
             sys.exit(1)
    # --- End Defensive Path Conversion ---

    pipeline_logger.info(f"🚀 Initializing GNN Pipeline with Target: '{args.target_dir}', Output: '{args.output_dir}'")
    
    # Log the arguments being used, showing their types after potential conversion
    if pipeline_logger.isEnabledFor(logging.DEBUG): # Check level before formatting potentially many lines
        log_msgs = ["🛠️ Effective Arguments (after potential defensive conversion):"]
        for arg, value in sorted(vars(args).items()):
            log_msgs.append(f"  --{arg.replace('_', '-')}: {value} (Type: {type(value).__name__})")
        pipeline_logger.debug('\n'.join(log_msgs))


    # Call the main pipeline execution function
    exit_code, pipeline_run_data, all_scripts, overall_status = run_pipeline(args)

    # --- Pipeline Summary Report ---
    logger.info("\n--- Pipeline Execution Summary ---")
    num_total = len(pipeline_run_data["steps"])
    num_success = len([s for s in pipeline_run_data["steps"] if s["status"] == "SUCCESS"])
    num_warn = len([s for s in pipeline_run_data["steps"] if s["status"] == "SUCCESS_WITH_WARNINGS"])
    num_failed = len([s for s in pipeline_run_data["steps"] if "FAILED" in s["status"] or "ERROR" in s["status"]])
    num_skipped = len([s for s in pipeline_run_data["steps"] if s["status"] == "SKIPPED"])

    logger.info(f"📊 Total Steps Attempted/Processed: {num_total - num_skipped} / {len(all_scripts)}")
    logger.info(f"  ✅ Successful: {num_success}")
    logger.info(f"  ⚠️ Success with Warnings: {num_warn}")
    logger.info(f"  ❌ Failed/Error: {num_failed}")
    logger.info(f"  ⏭️ Skipped: {num_skipped}")

    if overall_status == "SUCCESS":
        logger.info("🎉 PIPELINE FINISHED SUCCESSFULLY.")
    elif overall_status == "SUCCESS_WITH_WARNINGS":
        logger.warning("🎉 PIPELINE FINISHED, but with warnings from some steps.")
    else: # FAILED
        logger.error("🛑 PIPELINE FINISHED WITH ERRORS.")

    # Save detailed summary report
    try:
        args.pipeline_summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert Path objects to strings for JSON serialization
        def path_serializer(obj):
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(args.pipeline_summary_file, 'w') as f_summary:
            json.dump(pipeline_run_data, f_summary, indent=4, default=path_serializer)
        logger.info(f"💾 Detailed pipeline execution summary (JSON) saved to: {args.pipeline_summary_file}")
    except Exception as e:
        logger.error(f"❌ Error saving pipeline summary report: {e}")

    sys.exit(exit_code)

if __name__ == "__main__":
    main() 