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

from utils import log_step_start, log_step_success, log_step_warning, log_step_error, performance_tracker

try:
    from pipeline import get_output_dir_for_script
except ImportError:
    # Fallback for test environment
    from ..pipeline import get_output_dir_for_script

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
    Execute rendered simulator scripts with enhanced error handling and dependency checking.
    Framework outputs are organized in separate subdirectories.
    
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
    log_step_start(logger, "Executing rendered simulator scripts with framework-specific organization")
    
    # Use centralized output directory configuration
    execution_output_dir = get_output_dir_for_script("10_execute.py", output_dir)
    execution_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create framework-specific output directories
    framework_dirs = {
        "pymdp": execution_output_dir / "pymdp",
        "rxinfer": execution_output_dir / "rxinfer", 
        "discopy": execution_output_dir / "discopy",
        "activeinference_jl": execution_output_dir / "activeinference_jl",
        "jax": execution_output_dir / "jax"
    }
    
    for framework, framework_dir in framework_dirs.items():
        framework_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created framework directory: {framework_dir}")
    
    try:
        execution_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_directory": str(target_dir),
            "framework_execution_dirs": {k: str(v) for k, v in framework_dirs.items()},
            "pymdp_executions": [],
            "rxinfer_executions": [],
            "discopy_executions": [],
            "activeinference_executions": [],
            "jax_executions": [],
            "total_successes": 0,
            "total_failures": 0,
            "dependency_issues": [],
            "syntax_errors": [],
            "execution_details": {}
        }
        
        # Pre-execution validation and dependency checking
        logger.info("🔍 Pre-execution validation and dependency checking...")
        
        # Check Python dependencies
        python_deps = ["numpy", "pymdp", "flax", "jax", "optax"]
        missing_python_deps = []
        for dep in python_deps:
            try:
                __import__(dep)
                logger.debug(f"✅ Python dependency available: {dep}")
            except ImportError:
                missing_python_deps.append(dep)
                logger.warning(f"⚠️ Python dependency missing: {dep}")
        
        if missing_python_deps:
            execution_results["dependency_issues"].extend([
                f"Missing Python dependencies: {', '.join(missing_python_deps)}"
            ])
        
        # Check Julia availability
        try:
            result = subprocess.run(["julia", "--version"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                logger.info(f"✅ Julia available: {result.stdout.strip()}")
            else:
                logger.warning("⚠️ Julia not available or not working properly")
                execution_results["dependency_issues"].append("Julia not available")
        except FileNotFoundError:
            logger.warning("⚠️ Julia not found in PATH")
            execution_results["dependency_issues"].append("Julia not found in PATH")
        
        # Execute PyMDP scripts if available
        if PYMDP_AVAILABLE and run_pymdp_scripts:
            try:
                with performance_tracker.track_operation("execute_pymdp_scripts"):
                    logger.info("🚀 Executing PyMDP scripts...")
                    
                    # Look for rendered simulators in the output directory, not target_dir
                    rendered_simulators_dir = execution_output_dir.parent / "gnn_rendered_simulators"
                    pymdp_dir = rendered_simulators_dir / "pymdp"
                    
                    # Pre-validate PyMDP scripts for syntax errors
                    if pymdp_dir.exists():
                        pymdp_scripts = list(pymdp_dir.glob("*.py"))
                        for script in pymdp_scripts:
                            try:
                                with open(script, 'r') as f:
                                    compile(f.read(), script.name, 'exec')
                                logger.debug(f"✅ PyMDP script syntax valid: {script.name}")
                            except SyntaxError as e:
                                logger.warning(f"⚠️ PyMDP script syntax error in {script.name}: {e}")
                                execution_results["syntax_errors"].append(f"PyMDP: {script.name} - {e}")
                    
                    # Pass the output directory where rendered simulators should be located
                    pymdp_success = run_pymdp_scripts(
                        pipeline_output_dir=execution_output_dir.parent,
                        execution_output_dir=framework_dirs["pymdp"],
                        recursive_search=recursive,
                        verbose=verbose
                    )
                    if pymdp_success:
                        execution_results["total_successes"] += 1
                        execution_results["pymdp_executions"].append({
                            "status": "SUCCESS", 
                            "message": "PyMDP scripts executed successfully",
                            "output_dir": str(framework_dirs["pymdp"]),
                            "scripts_processed": len(list(pymdp_dir.glob("*.py"))) if pymdp_dir.exists() else 0
                        })
                    else:
                        execution_results["total_failures"] += 1
                        execution_results["pymdp_executions"].append({
                            "status": "FAILED", 
                            "message": "PyMDP script execution failed",
                            "output_dir": str(framework_dirs["pymdp"]),
                            "scripts_processed": len(list(pymdp_dir.glob("*.py"))) if pymdp_dir.exists() else 0
                        })
                log_step_success(logger, "PyMDP script execution completed")
            except Exception as e:
                execution_results["total_failures"] += 1
                execution_results["pymdp_executions"].append({
                    "status": "ERROR", 
                    "message": str(e),
                    "output_dir": str(framework_dirs["pymdp"])
                })
                log_step_warning(logger, f"PyMDP script execution failed: {e}")
        
        # Execute RxInfer scripts if available
        if RXINFER_AVAILABLE and run_rxinfer_scripts:
            try:
                with performance_tracker.track_operation("execute_rxinfer_scripts"):
                    logger.info("🚀 Executing RxInfer scripts...")
                    rxinfer_success = run_rxinfer_scripts(
                        pipeline_output_dir=execution_output_dir.parent,
                        execution_output_dir=framework_dirs["rxinfer"],
                        recursive_search=recursive,
                        verbose=verbose
                    )
                    if rxinfer_success:
                        execution_results["total_successes"] += 1
                        execution_results["rxinfer_executions"].append({
                            "status": "SUCCESS", 
                            "message": "RxInfer scripts executed successfully",
                            "output_dir": str(framework_dirs["rxinfer"])
                        })
                    else:
                        execution_results["total_failures"] += 1
                        execution_results["rxinfer_executions"].append({
                            "status": "FAILED", 
                            "message": "RxInfer script execution failed",
                            "output_dir": str(framework_dirs["rxinfer"])
                        })
                log_step_success(logger, "RxInfer script execution completed")
            except Exception as e:
                execution_results["total_failures"] += 1
                execution_results["rxinfer_executions"].append({
                    "status": "ERROR", 
                    "message": str(e),
                    "output_dir": str(framework_dirs["rxinfer"])
                })
                log_step_warning(logger, f"RxInfer script execution failed: {e}")
        
        # Execute DisCoPy analysis if available
        if DISCOPY_AVAILABLE and run_discopy_analysis:
            try:
                with performance_tracker.track_operation("execute_discopy_analysis"):
                    logger.info("🚀 Executing DisCoPy analysis...")
                    discopy_success = run_discopy_analysis(
                        pipeline_output_dir=execution_output_dir.parent,
                        execution_output_dir=framework_dirs["discopy"],
                        recursive_search=recursive,
                        verbose=verbose
                    )
                    if discopy_success:
                        execution_results["total_successes"] += 1
                        execution_results["discopy_executions"].append({
                            "status": "SUCCESS", 
                            "message": "DisCoPy analysis completed successfully",
                            "output_dir": str(framework_dirs["discopy"])
                        })
                    else:
                        execution_results["total_failures"] += 1
                        execution_results["discopy_executions"].append({
                            "status": "FAILED", 
                            "message": "DisCoPy analysis failed",
                            "output_dir": str(framework_dirs["discopy"])
                        })
                log_step_success(logger, "DisCoPy analysis completed")
            except Exception as e:
                execution_results["total_failures"] += 1
                execution_results["discopy_executions"].append({
                    "status": "ERROR", 
                    "message": str(e),
                    "output_dir": str(framework_dirs["discopy"])
                })
                log_step_warning(logger, f"DisCoPy analysis failed: {e}")
        
        # Execute ActiveInference.jl analysis if available
        if ACTIVEINFERENCE_AVAILABLE and run_activeinference_analysis:
            try:
                with performance_tracker.track_operation("execute_activeinference_analysis"):
                    logger.info("🚀 Executing ActiveInference.jl analysis...")
                    activeinference_success = run_activeinference_analysis(
                        pipeline_output_dir=execution_output_dir.parent,
                        execution_output_dir=framework_dirs["activeinference_jl"],
                        recursive_search=recursive,
                        verbose=verbose,
                        analysis_type="comprehensive"
                    )
                    if activeinference_success:
                        execution_results["total_successes"] += 1
                        execution_results["activeinference_executions"].append({
                            "status": "SUCCESS", 
                            "message": "ActiveInference.jl analysis completed successfully",
                            "output_dir": str(framework_dirs["activeinference_jl"])
                        })
                    else:
                        execution_results["total_failures"] += 1
                        execution_results["activeinference_executions"].append({
                            "status": "FAILED", 
                            "message": "ActiveInference.jl analysis failed",
                            "output_dir": str(framework_dirs["activeinference_jl"])
                        })
                log_step_success(logger, "ActiveInference.jl analysis completed")
            except Exception as e:
                execution_results["total_failures"] += 1
                execution_results["activeinference_executions"].append({
                    "status": "ERROR", 
                    "message": str(e),
                    "output_dir": str(framework_dirs["activeinference_jl"])
                })
                log_step_warning(logger, f"ActiveInference.jl analysis failed: {e}")
        
        # Execute JAX scripts if available
        if JAX_AVAILABLE and run_jax_scripts:
            try:
                with performance_tracker.track_operation("execute_jax_scripts"):
                    logger.info("🚀 Executing JAX scripts...")
                    jax_success = run_jax_scripts(
                        pipeline_output_dir=execution_output_dir.parent,
                        execution_output_dir=framework_dirs["jax"],
                        recursive_search=recursive,
                        verbose=verbose
                    )
                    if jax_success:
                        execution_results["total_successes"] += 1
                        execution_results["jax_executions"].append({
                            "status": "SUCCESS", 
                            "message": "JAX scripts executed successfully",
                            "output_dir": str(framework_dirs["jax"])
                        })
                    else:
                        execution_results["total_failures"] += 1
                        execution_results["jax_executions"].append({
                            "status": "FAILED", 
                            "message": "JAX script execution failed",
                            "output_dir": str(framework_dirs["jax"])
                        })
                log_step_success(logger, "JAX script execution completed")
            except Exception as e:
                execution_results["total_failures"] += 1
                execution_results["jax_executions"].append({
                    "status": "ERROR", 
                    "message": str(e),
                    "output_dir": str(framework_dirs["jax"])
                })
                log_step_warning(logger, f"JAX script execution failed: {e}")
        
        # Save execution summary with enhanced details
        summary_file = execution_output_dir / "execution_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(execution_results, f, indent=2)
        
        # Generate enhanced markdown report
        report_file = execution_output_dir / "execution_report.md"
        with open(report_file, 'w') as f:
            f.write("# Enhanced Execution Results Report\n\n")
            f.write(f"**Generated:** {execution_results['timestamp']}\n")
            f.write(f"**Target Directory:** {execution_results['target_directory']}\n")
            f.write(f"**Total Successes:** {execution_results['total_successes']}\n")
            f.write(f"**Total Failures:** {execution_results['total_failures']}\n\n")
            
            # Framework-specific output directories
            f.write("## Framework-Specific Output Directories\n\n")
            for framework, framework_dir in execution_results["framework_execution_dirs"].items():
                f.write(f"- **{framework.upper()}**: {framework_dir}\n")
            f.write("\n")
            
            # Dependency issues section
            if execution_results["dependency_issues"]:
                f.write("## Dependency Issues\n\n")
                for issue in execution_results["dependency_issues"]:
                    f.write(f"- ⚠️ {issue}\n")
                f.write("\n")
            
            # Syntax errors section
            if execution_results["syntax_errors"]:
                f.write("## Syntax Errors\n\n")
                for error in execution_results["syntax_errors"]:
                    f.write(f"- ❌ {error}\n")
                f.write("\n")
            
            if execution_results["pymdp_executions"]:
                f.write("## PyMDP Executions\n\n")
                for exec_info in execution_results["pymdp_executions"]:
                    status_icon = "✅" if exec_info.get('status') == 'SUCCESS' else "❌"
                    f.write(f"- {status_icon} **{exec_info.get('script', 'PyMDP Scripts')}**: {exec_info.get('status', 'Unknown')}\n")
                    f.write(f"  - {exec_info.get('message', 'No message')}\n")
                    f.write(f"  - Output Directory: {exec_info.get('output_dir', 'N/A')}\n")
                    if 'scripts_processed' in exec_info:
                        f.write(f"  - Scripts processed: {exec_info['scripts_processed']}\n")
                f.write("\n")
            
            if execution_results["rxinfer_executions"]:
                f.write("## RxInfer Executions\n\n")
                for exec_info in execution_results["rxinfer_executions"]:
                    status_icon = "✅" if exec_info.get('status') == 'SUCCESS' else "❌"
                    f.write(f"- {status_icon} **{exec_info.get('script', 'RxInfer Scripts')}**: {exec_info.get('status', 'Unknown')}\n")
                    f.write(f"  - {exec_info.get('message', 'No message')}\n")
                    f.write(f"  - Output Directory: {exec_info.get('output_dir', 'N/A')}\n")
                f.write("\n")
            
            if execution_results["discopy_executions"]:
                f.write("## DisCoPy Analyses\n\n")
                for exec_info in execution_results["discopy_executions"]:
                    status_icon = "✅" if exec_info.get('status') == 'SUCCESS' else "❌"
                    f.write(f"- {status_icon} **{exec_info.get('script', 'DisCoPy Analysis')}** ({exec_info.get('type', 'analysis')}): {exec_info.get('status', 'Unknown')}\n")
                    f.write(f"  - {exec_info.get('message', 'No message')}\n")
                    f.write(f"  - Output Directory: {exec_info.get('output_dir', 'N/A')}\n")
                f.write("\n")
            
            if execution_results["activeinference_executions"]:
                f.write("## ActiveInference.jl Analyses\n\n")
                for exec_info in execution_results["activeinference_executions"]:
                    status_icon = "✅" if exec_info.get('status') == 'SUCCESS' else "❌"
                    f.write(f"- {status_icon} **{exec_info.get('script', 'ActiveInference.jl Scripts')}**: {exec_info.get('status', 'Unknown')}\n")
                    f.write(f"  - {exec_info.get('message', 'No message')}\n")
                    f.write(f"  - Output Directory: {exec_info.get('output_dir', 'N/A')}\n")
                f.write("\n")
            
            if execution_results["jax_executions"]:
                f.write("## JAX Executions\n\n")
                for exec_info in execution_results["jax_executions"]:
                    status_icon = "✅" if exec_info.get('status') == 'SUCCESS' else "❌"
                    f.write(f"- {status_icon} **{exec_info.get('script', 'JAX Scripts')}**: {exec_info.get('status', 'Unknown')}\n")
                    f.write(f"  - {exec_info.get('message', 'No message')}\n")
                    f.write(f"  - Output Directory: {exec_info.get('output_dir', 'N/A')}\n")
                f.write("\n")
            
            # Recommendations section
            f.write("## Recommendations\n\n")
            if execution_results["dependency_issues"]:
                f.write("### Install Missing Dependencies\n\n")
                for issue in execution_results["dependency_issues"]:
                    if "Python dependencies" in issue:
                        f.write("- Install missing Python packages: `pip install <package_name>`\n")
                    elif "Julia" in issue:
                        f.write("- Install Julia from https://julialang.org/downloads/\n")
                f.write("\n")
            
            if execution_results["syntax_errors"]:
                f.write("### Fix Syntax Errors\n\n")
                f.write("- Review and fix syntax errors in rendered scripts\n")
                f.write("- Check for stray characters or malformed code\n")
                f.write("- Re-run the rendering step (9_render.py) to regenerate scripts\n\n")
        
        # Log results summary
        total_executions = (len(execution_results["pymdp_executions"]) + 
                          len(execution_results["rxinfer_executions"]) + 
                          len(execution_results["discopy_executions"]) +
                          len(execution_results["activeinference_executions"]) +
                          len(execution_results["jax_executions"]))
        
        if total_executions > 0:
            success_rate = execution_results["total_successes"] / total_executions * 100
            log_step_success(logger, f"Execution completed with framework-specific organization. Success rate: {success_rate:.1f}% ({execution_results['total_successes']}/{total_executions})")
            
            # Log specific issues
            if execution_results["dependency_issues"]:
                logger.warning(f"⚠️ Dependency issues found: {len(execution_results['dependency_issues'])}")
            if execution_results["syntax_errors"]:
                logger.warning(f"⚠️ Syntax errors found: {len(execution_results['syntax_errors'])}")
            
            return execution_results["total_failures"] == 0
        else:
            log_step_warning(logger, "No simulator scripts or outputs found to execute/analyze")
            return True
        
    except Exception as e:
        log_step_error(logger, f"Execution failed: {e}")
        return False 