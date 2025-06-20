"""
GNN Processing Pipeline - Step 10: Execute Rendered Simulators

This script orchestrates the execution of rendered GNN simulators,
focusing on PyMDP scripts and RxInfer.jl configurations generated by Step 9.

Usage:
    python 10_execute.py [options]
    (Typically called by main.py)
    
Options:
    Same as main.py (passes arguments through, especially output_dir and verbose)
"""

import sys
from pathlib import Path
import argparse
import subprocess
import json
import datetime

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
logger = setup_step_logging("10_execute", verbose=False)

# Attempt to import the runner functions from the execute module
try:
    from execute import pymdp_runner 
    try:
        from execute import rxinfer_runner
    except ImportError:
        log_step_warning(logger, "RxInfer runner module not found. RxInfer execution will be disabled.")
        rxinfer_runner = None
except ImportError as e:
    log_step_error(logger, f"Could not import pymdp_runner from execute module: {e}")
    logger.error("Ensure src/execute/pymdp_runner.py exists and src/ is discoverable.")
    pymdp_runner = None
    rxinfer_runner = None

def check_julia_availability():
    """Check if Julia is available in the system."""
    try:
        result = subprocess.run(
            ["julia", "--version"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        if result.returncode == 0:
            logger.info(f"Julia is available: {result.stdout.strip()}")
            return True
        else:
            logger.warning("Julia command returned non-zero exit code")
            return False
    except FileNotFoundError:
        logger.warning("Julia executable not found in PATH")
        return False
    except Exception as e:
        logger.warning(f"Error checking Julia availability: {e}")
        return False

def main(args: argparse.Namespace) -> int:
    """
    Main function for the GNN model execution step (Step 10).

    This function serves as the entry point for executing rendered GNN simulators.
    It orchestrates the execution of PyMDP scripts and RxInfer.jl configurations
    generated by Step 9.

    Args:
        args: Parsed command-line arguments with output_dir, recursive, and verbose.

    Returns:
        int: 0 for success, 1 for failure.
    """
    # Update logger verbosity based on args
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)
    
    log_step_start(logger, "Starting Step 10: Execute Rendered Simulators")
    
    # Create execution results output directory
    execution_output_dir = Path(args.output_dir) / "execution_results"
    if not validate_output_directory(Path(args.output_dir), "execution_results"):
        log_step_error(logger, "Failed to create execution results directory")
        return 1
    
    # Track overall success
    overall_success = True
    execution_summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "pymdp_results": {},
        "rxinfer_results": {},
        "overall_status": "SUCCESS"
    }

    try:
        # Execute PyMDP rendered scripts if available
        if pymdp_runner and hasattr(pymdp_runner, 'run_pymdp_scripts'):
            logger.info("Executing PyMDP rendered scripts...")
            
            try:
                pymdp_success = pymdp_runner.run_pymdp_scripts(
                    pipeline_output_dir=args.output_dir,
                    recursive_search=args.recursive,
                    verbose=args.verbose
                )
                
                execution_summary["pymdp_results"] = {
                    "status": "SUCCESS" if pymdp_success else "FAILED",
                    "executed": True
                }
                
                if pymdp_success:
                    logger.info("PyMDP scripts executed successfully")
                else:
                    log_step_error(logger, "Some PyMDP scripts failed during execution")
                    overall_success = False
                    
            except Exception as e:
                log_step_error(logger, f"PyMDP execution failed: {e}")
                execution_summary["pymdp_results"] = {"status": "ERROR", "error": str(e)}
                overall_success = False
        else:
            log_step_warning(logger, "PyMDP runner not available. Skipping PyMDP execution.")
            execution_summary["pymdp_results"] = {"status": "SKIPPED", "reason": "Runner not available"}

        # Execute RxInfer.jl rendered scripts if available
        if rxinfer_runner and hasattr(rxinfer_runner, 'run_rxinfer_scripts'):
            logger.info("Executing RxInfer.jl rendered scripts...")
            
            try:
                rxinfer_success = rxinfer_runner.run_rxinfer_scripts(
                    pipeline_output_dir=args.output_dir,
                    recursive_search=args.recursive,
                    verbose=args.verbose
                )
                
                execution_summary["rxinfer_results"] = {
                    "status": "SUCCESS" if rxinfer_success else "FAILED",
                    "executed": True
                }
                
                if rxinfer_success:
                    logger.info("RxInfer.jl scripts executed successfully")
                else:
                    log_step_error(logger, "Some RxInfer.jl scripts failed during execution")
                    overall_success = False
                    
            except Exception as e:
                log_step_error(logger, f"RxInfer execution failed: {e}")
                execution_summary["rxinfer_results"] = {"status": "ERROR", "error": str(e)}
                overall_success = False
        else:
            julia_available = check_julia_availability()
            reason = "Julia not available" if not julia_available else "Runner not available"
            log_step_warning(logger, f"RxInfer execution skipped: {reason}")
            execution_summary["rxinfer_results"] = {"status": "SKIPPED", "reason": reason}

        # Save execution summary
        if overall_success:
            execution_summary["overall_status"] = "SUCCESS"
        else:
            execution_summary["overall_status"] = "PARTIAL_SUCCESS" if any(
                result.get("status") == "SUCCESS" for result in 
                [execution_summary["pymdp_results"], execution_summary["rxinfer_results"]]
            ) else "FAILED"
        
        summary_file = execution_output_dir / "execution_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(execution_summary, f, indent=2)
        
        if overall_success:
            log_step_success(logger, "All available simulators executed successfully")
            return 0
        else:
            log_step_warning(logger, "Some simulator executions failed - check execution_summary.json")
            return 1
            
    except Exception as e:
        log_step_error(logger, f"Critical error during execution step: {e}")
        execution_summary["overall_status"] = "CRITICAL_ERROR"
        execution_summary["error"] = str(e)
        
        try:
            summary_file = execution_output_dir / "execution_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(execution_summary, f, indent=2)
        except Exception:
            pass  # Don't fail further if we can't save the summary
            
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execute rendered GNN simulators")
    
    # Define defaults for standalone execution
    script_file_path = Path(__file__).resolve()
    project_root = script_file_path.parent.parent
    default_output_dir = project_root / "output"

    parser.add_argument("--output-dir", type=Path, default=default_output_dir,
                       help="Main pipeline output directory")
    parser.add_argument("--recursive", action='store_true', 
                       help="Recursively search for scripts to execute")
    parser.add_argument("--verbose", action='store_true',
                       help="Enable verbose output")
    
    parsed_args = parser.parse_args()

    # Update logger for standalone execution
    if parsed_args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled for standalone execution")

    exit_code = main(parsed_args)
    sys.exit(exit_code) 