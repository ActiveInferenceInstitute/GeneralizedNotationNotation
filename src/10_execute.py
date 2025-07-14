#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 10: Execute

This script executes rendered GNN simulators.

Usage:
    python 10_execute.py [options]
    (Typically called by main.py)
"""

import sys
import subprocess
import json
import time
import logging
from pathlib import Path
import argparse

# Import centralized utilities and configuration
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    performance_tracker,
    UTILS_AVAILABLE
)

from pipeline import (
    STEP_METADATA,
    get_output_dir_for_script
)

# Initialize logger for this step
logger = setup_step_logging("10_execute", verbose=False)

# Import execution functionality
try:
    from execute.pymdp.pymdp_runner import run_pymdp_scripts
    PYMDP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PyMDP execution not available: {e}")
    PYMDP_AVAILABLE = False
    run_pymdp_scripts = None

try:
    from execute.rxinfer.rxinfer_runner import run_rxinfer_scripts
    RXINFER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RxInfer execution not available: {e}")
    RXINFER_AVAILABLE = False
    run_rxinfer_scripts = None

try:
    from execute.discopy.discopy_executor import DisCoPyExecutor, run_discopy_analysis
    DISCOPY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DisCoPy execution not available: {e}")
    DISCOPY_AVAILABLE = False
    DisCoPyExecutor = None
    run_discopy_analysis = None

try:
    from execute.activeinference_jl.activeinference_runner import run_activeinference_analysis
    ACTIVEINFERENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ActiveInference.jl execution not available: {e}")
    ACTIVEINFERENCE_AVAILABLE = False
    run_activeinference_analysis = None

try:
    from execute.jax.jax_runner import run_jax_scripts
    JAX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"JAX execution not available: {e}")
    JAX_AVAILABLE = False
    run_jax_scripts = None

logger.debug(f"Execution modules availability - PyMDP: {PYMDP_AVAILABLE}, RxInfer: {RXINFER_AVAILABLE}, DisCoPy: {DISCOPY_AVAILABLE}, ActiveInference.jl: {ACTIVEINFERENCE_AVAILABLE}, JAX: {JAX_AVAILABLE}")

def execute_rendered_simulators(target_dir: Path, output_dir: Path, recursive: bool = False, verbose: bool = False):
    """Execute rendered simulator scripts."""
    log_step_start(logger, "Executing rendered simulator scripts")
    
    # Use centralized output directory configuration
    execution_output_dir = get_output_dir_for_script("10_execute.py", output_dir)
    execution_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        execution_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_directory": str(target_dir),
            "pymdp_executions": [],
            "rxinfer_executions": [],
            "discopy_executions": [],
            "activeinference_executions": [],
            "jax_executions": [],
            "total_successes": 0,
            "total_failures": 0
        }
        
        # Execute PyMDP scripts if available
        if PYMDP_AVAILABLE and run_pymdp_scripts:
            try:
                with performance_tracker.track_operation("execute_pymdp_scripts"):
                    pymdp_success = run_pymdp_scripts(
                        pipeline_output_dir=target_dir,
                        recursive_search=recursive,
                        verbose=verbose
                    )
                    if pymdp_success:
                        execution_results["total_successes"] += 1
                        execution_results["pymdp_executions"].append({"status": "SUCCESS", "message": "PyMDP scripts executed successfully"})
                    else:
                        execution_results["total_failures"] += 1
                        execution_results["pymdp_executions"].append({"status": "FAILED", "message": "PyMDP script execution failed"})
                log_step_success(logger, "PyMDP script execution completed")
            except Exception as e:
                execution_results["total_failures"] += 1
                execution_results["pymdp_executions"].append({"status": "ERROR", "message": str(e)})
                log_step_warning(logger, f"PyMDP script execution failed: {e}")
        
        # Execute RxInfer scripts if available
        if RXINFER_AVAILABLE and run_rxinfer_scripts:
            try:
                with performance_tracker.track_operation("execute_rxinfer_scripts"):
                    rxinfer_success = run_rxinfer_scripts(
                        pipeline_output_dir=target_dir,
                        recursive_search=recursive,
                        verbose=verbose
                    )
                    if rxinfer_success:
                        execution_results["total_successes"] += 1
                        execution_results["rxinfer_executions"].append({"status": "SUCCESS", "message": "RxInfer scripts executed successfully"})
                    else:
                        execution_results["total_failures"] += 1
                        execution_results["rxinfer_executions"].append({"status": "FAILED", "message": "RxInfer script execution failed"})
                log_step_success(logger, "RxInfer script execution completed")
            except Exception as e:
                execution_results["total_failures"] += 1
                execution_results["rxinfer_executions"].append({"status": "ERROR", "message": str(e)})
                log_step_warning(logger, f"RxInfer script execution failed: {e}")
        
        # Execute DisCoPy analysis if available
        if DISCOPY_AVAILABLE and run_discopy_analysis:
            try:
                with performance_tracker.track_operation("execute_discopy_analysis"):
                    discopy_success = run_discopy_analysis(
                        pipeline_output_dir=target_dir,
                        recursive_search=recursive,
                        verbose=verbose
                    )
                    if discopy_success:
                        execution_results["total_successes"] += 1
                        execution_results["discopy_executions"].append({"status": "SUCCESS", "message": "DisCoPy analysis completed successfully"})
                    else:
                        execution_results["total_failures"] += 1
                        execution_results["discopy_executions"].append({"status": "FAILED", "message": "DisCoPy analysis failed"})
                log_step_success(logger, "DisCoPy analysis completed")
            except Exception as e:
                execution_results["total_failures"] += 1
                execution_results["discopy_executions"].append({"status": "ERROR", "message": str(e)})
                log_step_warning(logger, f"DisCoPy analysis failed: {e}")
        
        # Execute ActiveInference.jl analysis if available
        if ACTIVEINFERENCE_AVAILABLE and run_activeinference_analysis:
            try:
                with performance_tracker.track_operation("execute_activeinference_analysis"):
                    activeinference_success = run_activeinference_analysis(
                        pipeline_output_dir=target_dir,
                        recursive_search=recursive,
                        verbose=verbose,
                        analysis_type="comprehensive"
                    )
                    if activeinference_success:
                        execution_results["total_successes"] += 1
                        execution_results["activeinference_executions"].append({"status": "SUCCESS", "message": "ActiveInference.jl analysis completed successfully"})
                    else:
                        execution_results["total_failures"] += 1
                        execution_results["activeinference_executions"].append({"status": "FAILED", "message": "ActiveInference.jl analysis failed"})
                log_step_success(logger, "ActiveInference.jl analysis completed")
            except Exception as e:
                execution_results["total_failures"] += 1
                execution_results["activeinference_executions"].append({"status": "ERROR", "message": str(e)})
                log_step_warning(logger, f"ActiveInference.jl analysis failed: {e}")
        
        # Execute JAX scripts if available
        if JAX_AVAILABLE and run_jax_scripts:
            try:
                with performance_tracker.track_operation("execute_jax_scripts"):
                    jax_success = run_jax_scripts(
                        pipeline_output_dir=target_dir,
                        recursive_search=recursive,
                        verbose=verbose
                    )
                    if jax_success:
                        execution_results["total_successes"] += 1
                        execution_results["jax_executions"].append({"status": "SUCCESS", "message": "JAX scripts executed successfully"})
                    else:
                        execution_results["total_failures"] += 1
                        execution_results["jax_executions"].append({"status": "FAILED", "message": "JAX script execution failed"})
                log_step_success(logger, "JAX script execution completed")
            except Exception as e:
                execution_results["total_failures"] += 1
                execution_results["jax_executions"].append({"status": "ERROR", "message": str(e)})
                log_step_warning(logger, f"JAX script execution failed: {e}")
        
        # Save execution summary
        summary_file = execution_output_dir / "execution_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(execution_results, f, indent=2)
        
        # Generate markdown report
        report_file = execution_output_dir / "execution_report.md"
        with open(report_file, 'w') as f:
            f.write("# Execution Results Report\n\n")
            f.write(f"**Generated:** {execution_results['timestamp']}\n")
            f.write(f"**Target Directory:** {execution_results['target_directory']}\n")
            f.write(f"**Total Successes:** {execution_results['total_successes']}\n")
            f.write(f"**Total Failures:** {execution_results['total_failures']}\n\n")
            
            if execution_results["pymdp_executions"]:
                f.write("## PyMDP Executions\n\n")
                for exec_info in execution_results["pymdp_executions"]:
                    f.write(f"- **{exec_info.get('script', 'Unknown')}**: {exec_info.get('status', 'Unknown')}\n")
                f.write("\n")
            
            if execution_results["rxinfer_executions"]:
                f.write("## RxInfer Executions\n\n")
                for exec_info in execution_results["rxinfer_executions"]:
                    f.write(f"- **{exec_info.get('script', 'Unknown')}**: {exec_info.get('status', 'Unknown')}\n")
                f.write("\n")
            
            if execution_results["discopy_executions"]:
                f.write("## DisCoPy Analyses\n\n")
                for exec_info in execution_results["discopy_executions"]:
                    f.write(f"- **{exec_info.get('script', 'Unknown')}** ({exec_info.get('type', 'analysis')}): {exec_info.get('status', 'Unknown')}\n")
                f.write("\n")
            
            if execution_results["activeinference_executions"]:
                f.write("## ActiveInference.jl Analyses\n\n")
                for exec_info in execution_results["activeinference_executions"]:
                    f.write(f"- **{exec_info.get('script', 'Unknown')}**: {exec_info.get('status', 'Unknown')}\n")
                f.write("\n")
            
            if execution_results["jax_executions"]:
                f.write("## JAX Executions\n\n")
                for exec_info in execution_results["jax_executions"]:
                    f.write(f"- **{exec_info.get('script', 'Unknown')}**: {exec_info.get('status', 'Unknown')}\n")
                f.write("\n")
        
        # Log results summary
        total_executions = (len(execution_results["pymdp_executions"]) + 
                          len(execution_results["rxinfer_executions"]) + 
                          len(execution_results["discopy_executions"]) +
                          len(execution_results["activeinference_executions"]) +
                          len(execution_results["jax_executions"]))
        if total_executions > 0:
            success_rate = execution_results["total_successes"] / total_executions * 100
            log_step_success(logger, f"Execution completed. Success rate: {success_rate:.1f}% ({execution_results['total_successes']}/{total_executions})")
            return execution_results["total_failures"] == 0
        else:
            log_step_warning(logger, "No simulator scripts or outputs found to execute/analyze")
            return True
        
    except Exception as e:
        log_step_error(logger, f"Execution failed: {e}")
        return False

def main(parsed_args: argparse.Namespace):
    """Main function for execution operations."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("10_execute.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Execute rendered simulators')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Execute rendered simulators
    success = execute_rendered_simulators(
        target_dir=Path(parsed_args.target_dir),
        output_dir=Path(parsed_args.output_dir),
        recursive=getattr(parsed_args, 'recursive', False),
        verbose=getattr(parsed_args, 'verbose', False)
    )
    
    if success:
        log_step_success(logger, "Execution completed successfully")
        return 0
    else:
        log_step_error(logger, "Execution failed")
        return 1

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("10_execute")
    else:
        # Fallback argument parsing
        parser = argparse.ArgumentParser(description="Execute rendered simulators")
        parser.add_argument("--target-dir", type=Path, required=True,
                          help="Target directory containing GNN files")
        parser.add_argument("--output-dir", type=Path, required=True,
                          help="Output directory for generated artifacts")
        parser.add_argument("--recursive", action="store_true",
                          help="Search recursively in subdirectories")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose output")
        parsed_args = parser.parse_args()
    
    exit_code = main(parsed_args)
    sys.exit(exit_code) 