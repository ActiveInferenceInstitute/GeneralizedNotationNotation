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
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "execution_results"
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