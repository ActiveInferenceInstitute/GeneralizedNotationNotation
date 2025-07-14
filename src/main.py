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
- 4_type_checker.py (Corresponds to type_checker/ folder, uses type_checker.py)
- 5_export.py (Corresponds to export/ folder)
- 6_visualization.py (Corresponds to visualization/ folder, uses visualize_gnn.py)
- 7_mcp.py (Corresponds to mcp/ folder)
- 8_ontology.py (Corresponds to ontology/ folder)
- 9_render.py (Corresponds to render/ folder, includes PyMDP, RxInfer, DisCoPy, and JAX rendering)
- 10_execute.py (Corresponds to execute/ folder, includes PyMDP, RxInfer, DisCoPy, and JAX execution)
- 11_llm.py (Corresponds to llm/ folder)
- 12_site.py (Corresponds to site/ folder, generates HTML summary site)
- 13_sapf.py (Corresponds to sapf/ folder, generates SAPF (Sound As Pure Form) audio representations and sonifications of GNN models)

Configuration:
The pipeline uses a YAML configuration file located at input/config.yaml to configure
all aspects of the pipeline execution. The input directory structure is:

input/
├── config.yaml              # Pipeline configuration
└── gnn_files/               # GNN files to process
    ├── model1.md
    ├── model2.md
    └── ...

Usage:
    python main.py [options]
    
Options:
    --config-file FILE       Path to configuration file (default: input/config.yaml)
    --target-dir DIR         Target directory for GNN files (overrides config)
    --output-dir DIR         Directory to save outputs (overrides config)
    --recursive / --no-recursive    Recursively process directories (overrides config)
    --skip-steps LIST        Comma-separated list of steps to skip (overrides config)
    --only-steps LIST        Comma-separated list of steps to run (overrides config)
    --verbose                Enable verbose output (overrides config)
    --strict                 Enable strict type checking mode (for 4_type_checker)
    --estimate-resources / --no-estimate-resources 
                             Estimate computational resources (for 4_type_checker) (default: --estimate-resources)
    --ontology-terms-file    Path to the ontology terms file (overrides config)
    --llm-tasks LIST         Comma-separated list of LLM tasks to run for 11_llm.py 
                             (e.g., "summarize,explain_structure")
    --llm-timeout            Timeout in seconds for the LLM processing step (11_llm.py)
    --pipeline-summary-file FILE
                             Path to save the final pipeline summary report (overrides config)
    --site-html-filename NAME
                             Filename for the generated HTML summary site (for 12_site.py, saved in output-dir, default: gnn_pipeline_summary_site.html)
    --duration               Audio duration in seconds for SAPF generation (for 13_sapf.py, default: 30.0)
    --recreate-venv          Recreate virtual environment even if it already exists (for 2_setup.py)
    --dev                    Also install development dependencies from requirements-dev.txt (for 2_setup.py)

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
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import resource

# Import the centralized pipeline utilities
from pipeline import (
    get_pipeline_config,
    STEP_METADATA,
    get_output_dir_for_script
)
from pipeline.execution import execute_pipeline_step

# Import the streamlined utilities
from utils import (
    setup_main_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    PipelineArguments,
    PipelineLogger,
    performance_tracker,
    validate_pipeline_dependencies,
    UTILS_AVAILABLE,
    load_config,
    GNNPipelineConfig
)
from utils.argument_utils import build_enhanced_step_command_args

# --- Logger Setup ---
logger = logging.getLogger("GNN_Pipeline")

# Log psutil availability status after logger is available
if not PSUTIL_AVAILABLE:
    logger.warning("psutil not available - system memory and disk information will be limited")
# --- End Logger Setup ---

# Dependency validation is imported above with other utilities

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
        base_info = {
            "python_version": sys.version,
            "platform": os.name,
            "cpu_count": os.cpu_count(),
            "working_directory": str(Path.cwd()),
            "user": os.getenv('USER', 'unknown')
        }
        
        # Add psutil-dependent info if available
        if PSUTIL_AVAILABLE:
            base_info.update({
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "disk_free_gb": round(psutil.disk_usage('.').free / (1024**3), 2)
            })
        else:
            base_info.update({
                "memory_total_gb": "unavailable (psutil not installed)",
                "disk_free_gb": "unavailable (psutil not installed)"
            })
        
        return base_info
    except Exception as e:
        logger.warning(f"Failed to gather complete system info: {e}")
        return {"error": str(e)}

def parse_arguments() -> PipelineArguments:
    """Parse command line arguments and load configuration."""
    # Create argument parser for command line options
    parser = argparse.ArgumentParser(
        description="GNN Processing Pipeline with YAML configuration support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Add configuration file option
    parser.add_argument(
        '--config-file',
        type=Path,
        default=Path("input/config.yaml"),
        help='Path to configuration file (default: input/config.yaml)'
    )
    
    # Add all other options that can override config
    parser.add_argument('--target-dir', type=Path, help='Target directory for GNN files (overrides config)')
    parser.add_argument('--output-dir', type=Path, help='Directory to save outputs (overrides config)')
    parser.add_argument('--recursive', action=argparse.BooleanOptionalAction, help='Recursively process directories (overrides config)')
    parser.add_argument('--skip-steps', help='Comma-separated list of steps to skip (overrides config)')
    parser.add_argument('--only-steps', help='Comma-separated list of steps to run (overrides config)')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='Enable verbose output (overrides config)')
    parser.add_argument('--strict', action='store_true', help='Enable strict type checking mode')
    parser.add_argument('--estimate-resources', action=argparse.BooleanOptionalAction, help='Estimate computational resources')
    parser.add_argument('--ontology-terms-file', type=Path, help='Path to ontology terms file (overrides config)')
    parser.add_argument('--llm-tasks', help='Comma-separated list of LLM tasks')
    parser.add_argument('--llm-timeout', type=int, help='Timeout for LLM processing in seconds')
    parser.add_argument('--pipeline-summary-file', type=Path, help='Path to save pipeline summary')
    parser.add_argument('--site-html-filename', help='Filename for generated HTML site')
    parser.add_argument('--duration', type=float, help='Audio duration in seconds for SAPF generation')
    parser.add_argument('--recreate-venv', action='store_true', help='Recreate virtual environment')
    parser.add_argument('--dev', action='store_true', help='Install development dependencies')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Load configuration from YAML file
    try:
        # Resolve config file path relative to project root
        config_path = args.config_file
        if not config_path.is_absolute():
            # If we're in src/, go up one level to project root
            if Path.cwd().name == "src":
                config_path = Path("../") / config_path
            else:
                config_path = Path(".") / config_path
        
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")
    except Exception as e:
        logger.warning(f"Failed to load configuration from {args.config_file}: {e}")
        logger.info("Using default configuration")
        config = GNNPipelineConfig()
    
    # Convert config to PipelineArguments
    pipeline_args = PipelineArguments()
    
    # Set values from config
    config_dict = config.to_pipeline_arguments()
    for key, value in config_dict.items():
        if hasattr(pipeline_args, key):
            setattr(pipeline_args, key, value)
    
    # Override with command line arguments if provided
    if args.target_dir is not None:
        pipeline_args.target_dir = args.target_dir
    if args.output_dir is not None:
        pipeline_args.output_dir = args.output_dir
    if args.recursive is not None:
        pipeline_args.recursive = args.recursive
    if args.skip_steps is not None:
        pipeline_args.skip_steps = args.skip_steps
    if args.only_steps is not None:
        pipeline_args.only_steps = args.only_steps
    if args.verbose is not None:
        pipeline_args.verbose = args.verbose
    if args.strict:
        pipeline_args.strict = True
    if args.estimate_resources is not None:
        pipeline_args.estimate_resources = args.estimate_resources
    if args.ontology_terms_file is not None:
        pipeline_args.ontology_terms_file = args.ontology_terms_file
    if args.llm_tasks is not None:
        pipeline_args.llm_tasks = args.llm_tasks
    if args.llm_timeout is not None:
        pipeline_args.llm_timeout = args.llm_timeout
    if args.pipeline_summary_file is not None:
        pipeline_args.pipeline_summary_file = args.pipeline_summary_file
    if args.site_html_filename is not None:
        pipeline_args.site_html_filename = args.site_html_filename
    if args.duration is not None:
        pipeline_args.duration = args.duration
    if args.recreate_venv:
        pipeline_args.recreate_venv = True
    if args.dev:
        pipeline_args.dev = True
    
    # Resolve relative paths relative to input directory
    input_dir = Path("input")
    # If target_dir is relative, make it relative to input directory
    if not pipeline_args.target_dir.is_absolute():
        pipeline_args.target_dir = input_dir / pipeline_args.target_dir
    
    return pipeline_args

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
    """
    Find the virtual environment Python executable and site-packages path.
    
    Args:
        script_dir: The directory where the script is located (typically src/)
        
    Returns:
        Tuple of (venv_python_path, site_packages_path)
    """
    venv_python_path = None
    site_packages_path = None
    
    # Try multiple common virtual environment locations
    venv_candidates = [
        script_dir / ".venv",  # Standard .venv in script directory
        script_dir.parent / "venv",  # venv in parent directory (our case)
        script_dir.parent / ".venv",  # .venv in parent directory
    ]
    
    for venv_path in venv_candidates:
        logger.debug(f"🔍 Checking for virtual environment at: {venv_path}")
        
        if venv_path.is_dir():
            logger.debug(f"✓ Found virtual environment directory: {venv_path}")
            
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
            
            # Find site-packages path
            lib_path = venv_path / "lib"
            if lib_path.is_dir():
                for python_version_dir in lib_path.iterdir():
                    if python_version_dir.is_dir() and python_version_dir.name.startswith("python"):
                        current_site_packages = python_version_dir / "site-packages"
                        if current_site_packages.is_dir():
                            site_packages_path = current_site_packages
                            logger.debug(f"📦 Found site-packages at: {site_packages_path}")
                            break
            
            # If we found a Python executable, break out of the loop
            if venv_python_path:
                logger.info(f"✅ Using virtual environment: {venv_path}")
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
        4: "gnn_processing",    # 4_type_checker.py - GNN validation
        5: "export",            # 5_export.py - Export formats
        6: "visualization",     # 6_visualization.py - Visualization
        7: "core",              # 7_mcp.py - MCP tools
        8: "gnn_processing",    # 8_ontology.py - Ontology processing
        9: "core",              # 9_render.py - Rendering
        10: "core",             # 10_execute.py - Execution
        11: "core",             # 11_llm.py - LLM processing
        12: "core",             # 12_site.py - Site generation
        13: "core"              # 13_sapf.py - SAPF audio generation
    }
    
    # Determine which dependency groups we need
    required_groups = set(["core"])
    for step_num in range(1, 14):  # Updated to include steps 1-13
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

def run_pipeline(args: PipelineArguments):
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
        
        # Skip logic: Skip if explicitly listed in skip_steps by number or name
        # Note: We removed the "not is_enabled_by_default" check to run ALL discovered steps
        # Only critical failures (required=True steps) will halt the pipeline
        should_skip = (
            script_num_str in skip_steps or
            script_basename in skip_steps or
            script_name_no_ext in skip_steps
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
            logger.warning("⚠️ No scripts are configured to run in pipeline configuration")
        
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
    venv_python, venv_site_packages = get_venv_python(current_dir)
    
    if venv_python and venv_python != Path(sys.executable):
        logger.info(f"✅ Using virtual environment Python: {venv_python}")
    elif venv_python:
        logger.debug(f"✓ Using current Python interpreter: {venv_python}")
    else:
        logger.warning("⚠️ No specific Python found, will use system default")
    
    # Determine which Python to use based on availability
    python_to_use = venv_python or Path(sys.executable)
    logger.debug(f"🐍 Selected Python for subprocess calls: {python_to_use}")
    
    # Verify the selected Python executable exists and is executable
    if not python_to_use.exists():
        logger.error(f"❌ Selected Python executable does not exist: {python_to_use}")
        logger.critical("Cannot proceed without a valid Python executable.")
        sys.exit(1)
    
    # Ensure target_dir is correct relative to cwd (src/)
    if not args.target_dir.is_absolute():
        # If running from src/, prepend ../
        src_dir = Path.cwd()
        expected_path = src_dir.parent / args.target_dir
        if expected_path.exists():
            args.target_dir = os.path.relpath(str(expected_path), str(src_dir))
        else:
            args.target_dir = str(args.target_dir)

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
        # Build the command line for the step using the enhanced argument builder
        step_cmd = build_enhanced_step_command_args(
            script_info['basename'], args, str(python_to_use), script_info['path']
        )
        logger.debug(f"📋 Executing command: {' '.join(map(str, step_cmd))}")
        try:
            result = subprocess.run(
                step_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            step_status = "SUCCESS" if result.returncode == 0 else "FAILED"
            step_log = {
                "step_number": idx,
                "script_name": script_info['basename'],
                "status": step_status,
                "start_time": None,
                "end_time": None,
                "duration_seconds": None,
                "details": "",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "memory_usage_mb": None,
                "exit_code": result.returncode,
                "retry_count": 0
            }
            if step_status == "FAILED":
                overall_status = "FAILED"
                _pipeline_run_data_dict["performance_summary"]["failed_steps"] += 1
                config = get_pipeline_config()
                step_config = config.get_step_config(script_info['basename'])
                is_critical = step_config.required if step_config else True
                if is_critical:
                    _pipeline_run_data_dict["performance_summary"]["critical_failures"] += 1
                    logger.critical(f"🔥 Critical step {script_info['basename']} failed. Halting pipeline.")
                    _pipeline_run_data_dict["steps"].append(step_log)
                    break
            _pipeline_run_data_dict["steps"].append(step_log)
        except Exception as e:
            logger.error(f"Exception running step {script_info['basename']}: {e}")
            step_log = {
                "step_number": idx,
                "script_name": script_info['basename'],
                "status": "ERROR",
                "start_time": None,
                "end_time": None,
                "duration_seconds": None,
                "details": str(e),
                "stdout": "",
                "stderr": str(e),
                "memory_usage_mb": None,
                "exit_code": 1,
                "retry_count": 0
            }
            _pipeline_run_data_dict["steps"].append(step_log)
            overall_status = "FAILED"
            break

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
    """Main entry point for the GNN Processing Pipeline."""
    args = parse_arguments()
    
    # Set up streamlined logging
    log_dir = args.output_dir / "logs"
    pipeline_logger = setup_main_logging(log_dir, args.verbose)
    
    # Set correlation context for main pipeline
    PipelineLogger.set_correlation_context("main")

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