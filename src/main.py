#!/usr/bin/env python3
"""
GNN Processing Pipeline

This script orchestrates the full processing pipeline for GNN files and related artifacts.
It dynamically discovers and runs numbered scripts in the src/ directory, where each script
corresponds to a top-level folder and a specific processing stage.

Pipeline Steps (Dynamically Discovered and Ordered):
- 0_template.py (Corresponds to template/ folder, pipeline template and initialization)
- 1_setup.py (Corresponds to setup/ folder, environment setup and dependencies)
- 2_tests.py (Corresponds to tests/ folder, comprehensive test suite execution)
- 3_gnn.py (Corresponds to gnn/ folder, GNN file discovery and processing)
- 4_model_registry.py (Corresponds to model_registry/ folder, model registry management)
- 5_type_checker.py (Corresponds to type_checker/ folder, GNN syntax validation)
- 6_validation.py (Corresponds to validation/ folder, advanced validation)
- 7_export.py (Corresponds to export/ folder, multi-format export)
- 8_visualization.py (Corresponds to visualization/ folder, graph visualization)
- 9_advanced_viz.py (Corresponds to advanced_visualization/ folder, advanced plots)
- 10_ontology.py (Corresponds to ontology/ folder, Active Inference Ontology)
- 11_render.py (Corresponds to render/ folder, code generation for simulation environments)
- 12_execute.py (Corresponds to execute/ folder, simulation execution)
- 13_llm.py (Corresponds to llm/ folder, LLM-enhanced analysis)
- 14_ml_integration.py (Corresponds to ml_integration/ folder, ML integration)
- 15_audio.py (Corresponds to audio/ folder, audio generation)
- 16_analysis.py (Corresponds to analysis/ folder, advanced analysis)
- 17_integration.py (Corresponds to integration/ folder, system integration)
- 18_security.py (Corresponds to security/ folder, security validation)
- 19_research.py (Corresponds to research/ folder, research tools)
- 20_website.py (Corresponds to website/ folder, HTML website generation)
- 21_report.py (Corresponds to report/ folder, comprehensive reports)

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
    --website-html-filename NAME
                             Filename for the generated HTML summary website (for 13_website.py, saved in output-dir, default: gnn_pipeline_summary_website.html)
    --duration               Audio duration in seconds for audio generation (for 12_audio.py, default: 30.0)
    --audio-backend          Audio backend to use (for 12_audio.py, options: auto, sapf, pedalboard, default: auto)
    --recreate-venv          Recreate virtual environment even if it already exists (for 1_setup.py)
    --dev                    Also install development dependencies from requirements-dev.txt (for 1_setup.py)

"""

import sys
import os
from pathlib import Path
import logging
import datetime
import subprocess
from typing import TypedDict, List, Union, Dict, Any
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import pipeline utilities
from pipeline.discovery import get_pipeline_scripts
from pipeline.execution import prepare_scripts_to_run, execute_pipeline_step, summarize_execution, generate_and_print_summary
from utils.logging_utils import setup_main_logging, PipelineLogger
from utils.dependency_validator import validate_pipeline_dependencies_if_available
from utils.argument_utils import parse_arguments, PipelineArguments, validate_and_convert_paths
from utils.venv_utils import get_venv_python
from utils.system_utils import get_system_info

# Fix import path issues by ensuring src directory is in Python path
current_file = Path(__file__).resolve()
current_dir = current_file.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Check if virtual environment exists and create one if needed
current_file = Path(__file__).resolve()
current_dir = current_file.parent
project_root = current_dir.parent  # This should be the project root (one level up from src)

# Simple check for virtual environment
venv_path = project_root / ".venv"
if not venv_path.exists():
    print("🔧 No virtual environment detected. Creating one automatically...")
    try:
        import subprocess
        import sys
        
        # Create virtual environment
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        
        # Install dependencies
        print("📦 Installing dependencies...")
        if sys.platform == "win32":
            pip_path = venv_path / "Scripts" / "pip.exe"
        else:
            pip_path = venv_path / "bin" / "pip"
        
        subprocess.run([str(pip_path), "install", "-r", str(project_root / "requirements.txt")], check=True)
        print("✅ Virtual environment created successfully!")
        
        # Restart the script with the virtual environment
        print("🔄 Restarting with virtual environment...")
        if sys.platform == "win32":
            python_path = venv_path / "Scripts" / "python.exe"
        else:
            python_path = venv_path / "bin" / "python"
        
        # Get the original command line arguments
        args = sys.argv[1:] if len(sys.argv) > 1 else []
        
        # Restart with the virtual environment Python
        os.execv(str(python_path), [str(python_path), str(current_file)] + args)
        
    except Exception as e:
        print(f"❌ Error creating virtual environment: {e}")
        print("Please run 'python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt' manually.")
        sys.exit(1)

# Import the centralized pipeline utilities
try:
    from pipeline import (
        get_pipeline_config,
        get_output_dir_for_script
    )
    from pipeline.execution import prepare_scripts_to_run, execute_pipeline_step, summarize_execution, generate_and_print_summary
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)

# Import the streamlined utilities
try:
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
        validate_pipeline_dependencies_if_available,
        UTILS_AVAILABLE,
        load_config,
        GNNPipelineConfig,
        get_venv_python,
        get_system_info,
        parse_arguments,
        validate_and_convert_paths
    )
    from utils.argument_utils import build_enhanced_step_command_args
except ImportError as e:
    print(f"Error importing utility modules: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)

# Move get_pipeline_scripts to pipeline/discovery.py and import it
from pipeline.discovery import get_pipeline_scripts
from pipeline.execution import prepare_scripts_to_run, execute_pipeline_step, summarize_execution

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

def main():
    """Main entry point for the GNN Processing Pipeline."""
    args = parse_arguments()
    
    # After args = parse_arguments()
    project_root = current_dir.parent  # This should be the project root (one level up from src)
    
    # Resolve paths to absolute
    args.target_dir = (project_root / args.target_dir).resolve()
    args.output_dir = (project_root / args.output_dir).resolve()
    
    if args.ontology_terms_file:
        args.ontology_terms_file = (project_root / args.ontology_terms_file).resolve()
    if args.pipeline_summary_file:
        args.pipeline_summary_file = (project_root / args.pipeline_summary_file).resolve()
    else:
        args.pipeline_summary_file = args.output_dir / "pipeline_execution_summary.json"

    # Set up streamlined logging
    log_dir = args.output_dir / "logs"
    pipeline_logger = setup_main_logging(log_dir, args.verbose)
    
    # Set correlation context for main pipeline
    PipelineLogger.set_correlation_context("main")

    validate_and_convert_paths(args, pipeline_logger)

    pipeline_logger.info(f"🚀 Initializing GNN Pipeline with Target: '{args.target_dir}', Output: '{args.output_dir}'")
    
    # Log the arguments being used, showing their types after potential conversion
    if pipeline_logger.isEnabledFor(logging.DEBUG): # Check level before formatting potentially many lines
        log_msgs = ["🛠️ Effective Arguments (after potential defensive conversion):"]
        for arg, value in sorted(vars(args).items()):
            log_msgs.append(f"  --{arg.replace('_', '-')}: {value} (Type: {type(value).__name__})")
        pipeline_logger.debug('\n'.join(log_msgs))
    
    # Check if virtual environment exists and create one if needed
    venv_python, venv_site_packages = get_venv_python(current_dir)
    if not venv_python:
        pipeline_logger.info("🔧 No virtual environment detected. Creating one automatically...")
        try:
            # Import setup module for automatic virtual environment creation
            from setup.setup import perform_full_setup
            pipeline_logger.info("📦 Creating virtual environment and installing dependencies...")
            perform_full_setup(verbose=args.verbose, recreate_venv=False, dev=args.dev)
            pipeline_logger.info("✅ Virtual environment created successfully!")
            
            # Re-check for virtual environment
            venv_python, venv_site_packages = get_venv_python(current_dir)
            if not venv_python:
                pipeline_logger.critical("Failed to create virtual environment automatically.")
                sys.exit(1)
        except ImportError as e:
            pipeline_logger.critical(f"Could not import setup modules: {e}")
            pipeline_logger.critical("Please run 'python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt' manually.")
            sys.exit(1)
        except Exception as e:
            pipeline_logger.critical(f"Error creating virtual environment: {e}")
            sys.exit(1)
    
    # Validate dependencies before pipeline execution
    if not validate_pipeline_dependencies_if_available(args):
        pipeline_logger.critical("Pipeline aborted due to dependency validation failures.")
        sys.exit(1)
        
    # Call the main pipeline execution function
    exit_code, pipeline_run_data, all_scripts, overall_status = run_pipeline(args, pipeline_logger)

    generate_and_print_summary(pipeline_run_data, all_scripts, args, pipeline_logger, overall_status)

    sys.exit(exit_code)

def run_pipeline(args: PipelineArguments, logger: logging.Logger):
    current_dir = Path(__file__).resolve().parent
    all_scripts = get_pipeline_scripts(current_dir)
    
    pipeline_run_data = {
        "start_time": datetime.datetime.now().isoformat(),
        "arguments": vars(args),
        "steps": [],
        "end_time": None,
        "overall_status": "SUCCESS",
        "total_duration_seconds": None,
        "environment_info": get_system_info(),
        "performance_summary": {
            "peak_memory_mb": 0.0,
            "total_steps": 0,
            "failed_steps": 0,
            "critical_failures": 0,
            "successful_steps": 0,
            "warnings": 0
        }
    }
    
    scripts_to_run = prepare_scripts_to_run(all_scripts, args)
    
    if not scripts_to_run:
        pipeline_run_data["end_time"] = datetime.datetime.now().isoformat()
        pipeline_run_data["overall_status"] = "SUCCESS_WITH_WARNINGS"
        return 0, pipeline_run_data, all_scripts, "SUCCESS_WITH_WARNINGS"
    
    venv_python, venv_site_packages = get_venv_python(current_dir)
    python_to_use = venv_python or Path(sys.executable)
    
    overall_status = "SUCCESS"
    for idx, script_info in enumerate(scripts_to_run, 1):
        step_result = execute_pipeline_step(script_info['basename'], idx, len(scripts_to_run), python_to_use, args.target_dir, args.output_dir, args, logger)
        pipeline_run_data["steps"].append(step_result)
        if step_result["status"] == "FAILED":
            overall_status = "FAILED"
            break
    
    summarize_execution(pipeline_run_data, scripts_to_run, overall_status)
    
    exit_code = 0 if overall_status == "SUCCESS" else 1
    return exit_code, pipeline_run_data, all_scripts, overall_status

if __name__ == "__main__":
    main() 