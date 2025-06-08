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
- 12_discopy.py (Corresponds to discopy_translator_module/ folder, generates DisCoPy diagrams from GNN)
- 13_discopy_jax_eval.py (Corresponds to discopy_translator_module/ folder, generates DisCoPy diagrams from GNN using JAX)
- 14_site.py (Corresponds to site/ folder, generates HTML summary site)


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
                            Filename for the generated HTML summary site (for 14_site.py, saved in output-dir, default: gnn_pipeline_summary_site.html)
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
import datetime
import json
import time
import signal
from typing import TypedDict, List, Union, Dict, Any, cast
import psutil
import resource

# Import the centralized pipeline utilities
from pipeline import (
    PIPELINE_STEP_CONFIGURATION,
    ARG_PROPERTIES,
    SCRIPT_ARG_SUPPORT,
    get_step_timeout,
    is_critical_step,
    get_output_dir_for_script,
    StepExecutionResult,
    get_memory_usage_mb,
    build_command_args,
    execute_pipeline_step
)

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

# Import dependency validation utility
try:
    from utils.dependency_validator import validate_pipeline_dependencies
except ImportError:
    logger.warning("Dependency validator not available")
    validate_pipeline_dependencies = None

# Define types for pipeline summary data
class StepLogData(TypedDict):
    step_number: int
    script_name: str
    status: str
    start_time: Union[str, None]
    end_time: Union[str, None]
    duration_seconds: Union[float, None]
    details: str
    stdout: str
    stderr: str
    # Enhanced metadata
    memory_usage_mb: Union[float, None]
    exit_code: Union[int, None]
    retry_count: int

class PipelineRunData(TypedDict):
    start_time: str
    arguments: Dict[str, Any]
    steps: List[StepLogData]
    end_time: Union[str, None]
    overall_status: str
    # Enhanced summary
    total_duration_seconds: Union[float, None]
    environment_info: Dict[str, Any]
    performance_summary: Dict[str, Any]

def get_system_info() -> Dict[str, Any]:
    """Gather comprehensive system information for pipeline tracking."""
    try:
        return {
            "python_version": sys.version,
            "platform": os.name,
            "cpu_count": os.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_free_gb": round(psutil.disk_usage('.').free / (1024**3), 2),
            "working_directory": str(Path.cwd()),
            "user": os.getenv('USER', 'unknown')
        }
    except Exception as e:
        logger.warning(f"Failed to gather complete system info: {e}")
        return {"error": str(e)}

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
        default=120, # Default to 2 minutes for the entire LLM script
        help="Timeout in seconds for the LLM processing step (11_llm.py). Default: 120"
    )
    parser.add_argument(
        "--pipeline-summary-file",
        type=Path,
        default=default_pipeline_summary_file,
        help=(f"Path to save the final pipeline summary report.\\n"
              f"Default: {default_pipeline_summary_file.relative_to(project_root) if default_pipeline_summary_file.is_relative_to(project_root) else default_pipeline_summary_file}")
    )
    parser.add_argument(
        "--site-html-filename",
        type=str,
        default="gnn_pipeline_summary_site.html",
        help=(f"Filename for the generated HTML summary site (for 14_site.py, saved in --output-dir).\\n"
              f"Default: gnn_pipeline_summary_site.html")
    )
    parser.add_argument(
        "--discopy-gnn-input-dir",
        type=Path,
        default=None, # Will be set to default_target_dir or args.target_dir later
        help=(f"Directory containing GNN files for DisCoPy processing (for 12_discopy.py).\\n"
              f"Default: Uses --target-dir value ({default_discopy_gnn_input_dir.relative_to(project_root) if default_discopy_gnn_input_dir.is_relative_to(project_root) else default_discopy_gnn_input_dir})")
    )
    parser.add_argument(
        "--discopy-jax-gnn-input-dir",
        type=Path,
        default=None, # Will be set to default_target_dir or args.target_dir later
        help=(f"Directory containing GNN files for DisCoPy JAX evaluation (for 13_discopy_jax_eval.py).\\n"
              f"Default: Uses --target-dir value ({default_discopy_gnn_input_dir.relative_to(project_root) if default_discopy_gnn_input_dir.is_relative_to(project_root) else default_discopy_gnn_input_dir})")
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
    parser.add_argument(
        "--skip-dependency-validation",
        action="store_true",
        help="Skip dependency validation and proceed with pipeline execution."
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

def validate_pipeline_dependencies_if_available(args: argparse.Namespace) -> bool:
    """
    Validate dependencies if the validator is available.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        bool: True if validation passed or validator unavailable
    """
    if getattr(args, 'skip_dependency_validation', False):
        logger.info("Dependency validation skipped (--skip-dependency-validation flag)")
        return True
        
    if validate_pipeline_dependencies is None:
        logger.info("Dependency validation skipped (validator not available)")
        return True
        
    logger.info("=== DEPENDENCY VALIDATION ===")
    
    # Determine required steps based on what will run
    required_steps = ["setup"]  # Always need core dependencies
    
    # Check which steps will actually run
    skip_steps = parse_step_list(args.skip_steps) if args.skip_steps else []
    only_steps = parse_step_list(args.only_steps) if args.only_steps else []
    
    # Map step numbers to dependency groups
    step_dependency_map = {
        1: "gnn_processing",    # 1_gnn.py - GNN file processing 
        2: "core",              # 2_setup.py - Setup step
        3: "testing",           # 3_tests.py - Testing framework
        4: "gnn_processing",    # 4_gnn_type_checker.py - GNN validation
        5: "export",            # 5_export.py - Export formats
        6: "visualization",     # 6_visualization.py - Visualization
        7: "core",              # 7_mcp.py - MCP tools
        8: "gnn_processing",    # 8_ontology.py - Ontology processing
        9: "core",              # 9_render.py - Rendering
        10: "core",             # 10_execute.py - Execution
        11: "core",             # 11_llm.py - LLM processing
        12: "core",             # 12_discopy.py - DisCoPy processing
        13: "core",             # 13_discopy_jax_eval.py - JAX evaluation
        14: "core"              # 14_site.py - Site generation
    }
    
    # Determine which dependency groups we need
    required_groups = set(["core"])
    for step_num in range(1, 15):
        # Skip if in skip list
        if step_num in skip_steps or f"{step_num}_" in str(skip_steps):
            continue
        # Skip if only_steps specified and this step not in it
        if only_steps and step_num not in only_steps:
            continue
        
        if step_num in step_dependency_map:
            required_groups.add(step_dependency_map[step_num])
    
    logger.info(f"Validating dependency groups: {sorted(required_groups)}")
    
    # Get the virtual environment Python path for dependency validation
    current_dir = Path(__file__).resolve().parent
    venv_python, _ = get_venv_python(current_dir)
    python_path = str(venv_python) if venv_python else None
    
    if python_path:
        logger.debug(f"Using Python for dependency validation: {python_path}")
    else:
        logger.debug("Using system Python for dependency validation")
    
    # Validate dependencies
    try:
        is_valid = validate_pipeline_dependencies(list(required_groups), python_path=python_path)
        
        if not is_valid:
            logger.critical("Dependency validation failed. Cannot proceed with pipeline execution.")
            logger.critical("Please install the missing dependencies and try again.")
            logger.critical("Alternatively, use --skip-dependency-validation to bypass this check.")
            return False
        
        logger.info("All required dependencies validated successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error during dependency validation: {e}")
        logger.critical("Dependency validation encountered an error. Cannot proceed with pipeline execution.")
        logger.critical("Use --skip-dependency-validation to bypass this check, or fix the validation error.")
        return False


def parse_step_list(step_str: str) -> List[int]:
    """Parse a comma-separated list of step numbers."""
    if not step_str:
        return []
    
    steps = []
    for item in step_str.split(','):
        item = item.strip()
        # Extract number from formats like "1", "1_gnn", etc.
        match = re.match(r'^(\d+)', item)
        if match:
            steps.append(int(match.group(1)))
    return steps

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
        "overall_status": "SUCCESS",  # Will be updated if any step fails
        "total_duration_seconds": None,
        "environment_info": get_system_info(),
        "performance_summary": {
            "peak_memory_mb": 0.0,
            "total_steps": 0,
            "failed_steps": 0,
            "critical_failures": 0
        }
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
    
    # Execute each script
    overall_status = "SUCCESS"
    
    logger.info("\n  ██████╗ ███╗   ██╗ ███╗   ██╗") 
    logger.info("  ██╔════╝ ████╗  ██║ ████╗  ██║")
    logger.info("  ██║  ███╗██╔██╗ ██║ ██╔██╗ ██║")
    logger.info("  ██║   ██║██║╚██╗██║ ██║╚██╗██║")
    logger.info("  ╚██████╔╝██║ ╚████║ ██║ ╚████║")
    logger.info("   ╚═════╝ ╚═╝  ╚═══╝ ╚═╝  ╚═══╝")
    logger.info("  Generalized Notation Notation")
    logger.info("  Active Inference Institute")
    logger.info(f"  Pipeline Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for idx, script_info in enumerate(scripts_to_run, 1):
        # Execute the pipeline step using the centralized execution function
        result = execute_pipeline_step(
            script_info, 
            idx, 
            len(scripts_to_run),
            args,
            str(python_to_use)
        )
        
        # Update performance summary
        _pipeline_run_data_dict["performance_summary"]["total_steps"] += 1
        
        if result.memory_usage_mb:
            _pipeline_run_data_dict["performance_summary"]["peak_memory_mb"] = max(
                _pipeline_run_data_dict["performance_summary"]["peak_memory_mb"], 
                result.memory_usage_mb
            )
        
        if result.status != "SUCCESS":
            overall_status = "FAILED"
            _pipeline_run_data_dict["performance_summary"]["failed_steps"] += 1
            
            # Check if this was a critical step failure
            if is_critical_step(result.script_name):
                _pipeline_run_data_dict["performance_summary"]["critical_failures"] += 1
                logger.critical(f"🔥 Critical step {result.script_name} failed. Halting pipeline.")
                _pipeline_run_data_dict["steps"].append(result.to_dict())
                break
        
        # Add step result to summary
        _pipeline_run_data_dict["steps"].append(result.to_dict())

    _pipeline_run_data_dict["end_time"] = datetime.datetime.now().isoformat()
    _pipeline_run_data_dict["overall_status"] = overall_status
    
    # Calculate total duration
    if _pipeline_run_data_dict["start_time"] and _pipeline_run_data_dict["end_time"]:
        start_dt = datetime.datetime.fromisoformat(_pipeline_run_data_dict["start_time"])
        end_dt = datetime.datetime.fromisoformat(_pipeline_run_data_dict["end_time"])
        _pipeline_run_data_dict["total_duration_seconds"] = (end_dt - start_dt).total_seconds()
    
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

        # --- Legacy File Handler Setup (after args are parsed) ---
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            # Add a file handler to the GNN_Pipeline logger
            log_file_path = log_dir / "pipeline.log"
            file_handler = logging.FileHandler(log_file_path, mode='w') # 'w' to overwrite each run
            # Use the same format as basicConfig, explicitly ensure milliseconds
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt=None)
            file_handler.setFormatter(file_formatter)
            pipeline_logger.addHandler(file_handler)
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

        # Configure logger level based on verbose flag AFTER parsing args
        if args.verbose:
            pipeline_logger.setLevel(logging.DEBUG)
        else:
            pipeline_logger.setLevel(logging.INFO)

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
    
    # Validate dependencies before pipeline execution
    if not validate_pipeline_dependencies_if_available(args):
        pipeline_logger.critical("Pipeline aborted due to dependency validation failures.")
        sys.exit(1)
        
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