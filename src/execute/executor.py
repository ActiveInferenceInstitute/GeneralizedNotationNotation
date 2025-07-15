"""
GNN Executor Module

This module provides the main execution functionality for GNN models,
including script execution, simulation management, and result collection.
"""

import os
import subprocess
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time
import sys

# Import execution functionality
try:
    from .pymdp.pymdp_runner import run_pymdp_scripts
    PYMDP_AVAILABLE = True
except ImportError as e:
    PYMDP_AVAILABLE = False
    run_pymdp_scripts = None

try:
    from .rxinfer.rxinfer_runner import run_rxinfer_scripts
    RXINFER_AVAILABLE = True
except ImportError as e:
    RXINFER_AVAILABLE = False
    run_rxinfer_scripts = None

try:
    from .discopy.discopy_executor import run_discopy_analysis
    DISCOPY_AVAILABLE = True
except ImportError as e:
    DISCOPY_AVAILABLE = False
    run_discopy_analysis = None

try:
    from .activeinference_jl.activeinference_runner import run_activeinference_analysis
    ACTIVEINFERENCE_AVAILABLE = True
except ImportError as e:
    ACTIVEINFERENCE_AVAILABLE = False
    run_activeinference_analysis = None

try:
    from .jax.jax_runner import run_jax_scripts
    JAX_AVAILABLE = True
except ImportError as e:
    JAX_AVAILABLE = False
    run_jax_scripts = None

from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error, performance_tracker

logger = logging.getLogger(__name__)


class GNNExecutor:
    """
    Main executor for GNN model simulations and scripts.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the GNN executor.
        
        Args:
            output_dir: Directory for execution outputs
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to a subdirectory within the current working directory
            self.output_dir = Path.cwd() / "output" / "execution_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.execution_log = []
    
    def execute_gnn_model(self, model_path: str, execution_type: str = "pymdp", 
                         options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a GNN model with the specified execution type.
        
        Args:
            model_path: Path to the GNN model or rendered script
            execution_type: Type of execution (pymdp, rxinfer, discopy, etc.)
            options: Additional execution options
        
        Returns:
            Dictionary with execution results
        """
        try:
            start_time = time.time()
            
            if execution_type == "pymdp":
                result = self._execute_pymdp_script(model_path, options)
            elif execution_type == "rxinfer":
                result = self._execute_rxinfer_config(model_path, options)
            elif execution_type == "discopy":
                result = self._execute_discopy_diagram(model_path, options)
            elif execution_type == "jax":
                result = self._execute_jax_script(model_path, options)
            else:
                result = {
                    "success": False,
                    "error": f"Unsupported execution type: {execution_type}"
                }
            
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            result["execution_type"] = execution_type
            result["model_path"] = model_path
            
            # Log execution
            self.execution_log.append(result)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_type": execution_type,
                "model_path": model_path
            }
    
    def run_simulation(self, simulation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a simulation based on configuration.
        
        Args:
            simulation_config: Configuration dictionary for the simulation
        
        Returns:
            Dictionary with simulation results
        """
        try:
            model_path = simulation_config.get("model_path")
            execution_type = simulation_config.get("execution_type", "pymdp")
            options = simulation_config.get("options", {})
            
            if not model_path:
                return {
                    "success": False,
                    "error": "No model path specified in simulation config"
                }
            
            return self.execute_gnn_model(model_path, execution_type, options)
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def generate_execution_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate an execution report from the execution log.
        
        Args:
            output_file: Path for the output report file
        
        Returns:
            Path to the generated report
        """
        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"execution_report_{timestamp}.json"
        
        report_data = {
            "execution_summary": {
                "total_executions": len(self.execution_log),
                "successful_executions": sum(1 for r in self.execution_log if r.get("success", False)),
                "failed_executions": sum(1 for r in self.execution_log if not r.get("success", False)),
                "total_execution_time": sum(r.get("execution_time", 0) for r in self.execution_log)
            },
            "execution_details": self.execution_log
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return str(output_file)
    
    def _execute_pymdp_script(self, script_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a PyMDP script."""
        try:
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True, timeout=300)
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Execution timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _execute_rxinfer_config(self, config_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute an RxInfer.jl configuration."""
        try:
            # This would typically involve calling Julia
            result = subprocess.run(["julia", config_path], 
                                  capture_output=True, text=True, timeout=300)
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Execution timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _execute_discopy_diagram(self, diagram_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a DisCoPy diagram."""
        try:
            result = subprocess.run([sys.executable, diagram_path], 
                                  capture_output=True, text=True, timeout=300)
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Execution timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _execute_jax_script(self, script_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a JAX script."""
        try:
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True, timeout=300)
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Execution timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


def execute_gnn_model(model_path: str, execution_type: str = "pymdp", 
                     options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to execute a GNN model.
    
    Args:
        model_path: Path to the GNN model or rendered script
        execution_type: Type of execution
        options: Additional execution options
    
    Returns:
        Dictionary with execution results
    """
    executor = GNNExecutor()
    return executor.execute_gnn_model(model_path, execution_type, options)


def run_simulation(simulation_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to run a simulation.
    
    Args:
        simulation_config: Configuration dictionary for the simulation
    
    Returns:
        Dictionary with simulation results
    """
    executor = GNNExecutor()
    return executor.run_simulation(simulation_config)


def generate_execution_report(execution_log: List[Dict[str, Any]], 
                            output_file: Optional[str] = None) -> str:
    """
    Convenience function to generate an execution report.
    
    Args:
        execution_log: List of execution results
        output_file: Path for the output report file
    
    Returns:
        Path to the generated report
    """
    executor = GNNExecutor()
    executor.execution_log = execution_log
    return executor.generate_execution_report(output_file) 


def execute_rendered_simulators(
    target_dir: Path, 
    output_dir: Path, 
    logger: logging.Logger, 
    recursive: bool = False, 
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Execute rendered simulator scripts.
    
    Args:
        target_dir: Directory containing rendered simulator scripts
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional execution options
        
    Returns:
        True if execution succeeded, False otherwise
    """
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