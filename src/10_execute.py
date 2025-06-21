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
    from execute.pymdp_runner import PyMDPRunner
    from execute.rxinfer_runner import RxInferRunner
    logger.debug("Successfully imported execution modules")
except ImportError as e:
    log_step_error(logger, f"Could not import execution modules: {e}")
    PyMDPRunner = None
    RxInferRunner = None

def execute_rendered_simulators(target_dir: Path, output_dir: Path, recursive: bool = False):
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
            "total_successes": 0,
            "total_failures": 0
        }
        
        # Execute PyMDP scripts if runner available
        if PyMDPRunner:
            try:
                with performance_tracker.track_operation("execute_pymdp_scripts"):
                    pymdp_runner = PyMDPRunner()
                    pymdp_results = pymdp_runner.execute_directory(
                        target_dir=target_dir / "gnn_rendered_simulators" / "pymdp",
                        output_dir=execution_output_dir / "pymdp_results"
                    )
                    execution_results["pymdp_executions"] = pymdp_results.get("executions", [])
                    execution_results["total_successes"] += pymdp_results.get("successes", 0)
                    execution_results["total_failures"] += pymdp_results.get("failures", 0)
                log_step_success(logger, "PyMDP script execution completed")
            except Exception as e:
                log_step_warning(logger, f"PyMDP script execution failed: {e}")
        
        # Execute RxInfer scripts if runner available
        if RxInferRunner:
            try:
                with performance_tracker.track_operation("execute_rxinfer_scripts"):
                    rxinfer_runner = RxInferRunner()
                    rxinfer_results = rxinfer_runner.execute_directory(
                        target_dir=target_dir / "gnn_rendered_simulators" / "rxinfer",
                        output_dir=execution_output_dir / "rxinfer_results"
                    )
                    execution_results["rxinfer_executions"] = rxinfer_results.get("executions", [])
                    execution_results["total_successes"] += rxinfer_results.get("successes", 0)
                    execution_results["total_failures"] += rxinfer_results.get("failures", 0)
                log_step_success(logger, "RxInfer script execution completed")
            except Exception as e:
                log_step_warning(logger, f"RxInfer script execution failed: {e}")
        
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
        
        # Log results summary
        total_executions = len(execution_results["pymdp_executions"]) + len(execution_results["rxinfer_executions"])
        if total_executions > 0:
            success_rate = execution_results["total_successes"] / total_executions * 100
            log_step_success(logger, f"Execution completed. Success rate: {success_rate:.1f}% ({execution_results['total_successes']}/{total_executions})")
            return execution_results["total_failures"] == 0
        else:
            log_step_warning(logger, "No simulator scripts found to execute")
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
        recursive=getattr(parsed_args, 'recursive', False)
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