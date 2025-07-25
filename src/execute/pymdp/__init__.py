#!/usr/bin/env python3
"""
PyMDP Execution Module

This module provides PyMDP simulation execution capabilities for the GNN pipeline.
It includes utilities for running PyMDP simulations configured from GNN specifications
with enhanced safety patterns and comprehensive error handling.
"""

import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .pymdp_simulation import PyMDPSimulation
from .pymdp_utils import (
    convert_numpy_for_json,
    safe_json_dump,
    safe_pickle_dump,
    clean_trace_for_serialization,
    save_simulation_results,
    generate_simulation_summary,
    create_output_directory_with_timestamp,
    format_duration
)
from .pymdp_visualizer import PyMDPVisualizer, create_visualizer, save_all_visualizations

__all__ = [
    'PyMDPSimulation',
    'PyMDPVisualizer',
    'execute_pymdp_simulation_from_gnn',
    'execute_pymdp_simulation',
    'validate_pymdp_environment',
    'get_pymdp_health_status',
    'create_enhanced_pymdp_context'
]

logger = logging.getLogger(__name__)


def validate_pymdp_environment() -> Dict[str, Any]:
    """
    Validate PyMDP execution environment and dependencies.
    
    Returns:
        Dictionary with validation results and health status
    """
    validation_results = {
        "pymdp_available": False,
        "dependencies": {},
        "errors": [],
        "warnings": [],
        "recommendations": []
    }
    
    # Check PyMDP availability
    try:
        import pymdp
        validation_results["pymdp_available"] = True
        validation_results["dependencies"]["pymdp"] = {
            "available": True,
            "version": getattr(pymdp, "__version__", "unknown")
        }
        logger.info(f"PyMDP available: version {validation_results['dependencies']['pymdp']['version']}")
    except ImportError as e:
        validation_results["pymdp_available"] = False
        validation_results["dependencies"]["pymdp"] = {
            "available": False,
            "error": str(e)
        }
        validation_results["errors"].append("PyMDP not available - install with: pip install pymdp")
        logger.error("PyMDP not available for execution")
    
    # Check numpy availability
    try:
        import numpy as np
        validation_results["dependencies"]["numpy"] = {
            "available": True,
            "version": np.__version__
        }
    except ImportError:
        validation_results["dependencies"]["numpy"] = {
            "available": False,
            "error": "NumPy not available"
        }
        validation_results["errors"].append("NumPy not available - install with: pip install numpy")
    
    # Check matplotlib for visualizations
    try:
        import matplotlib
        validation_results["dependencies"]["matplotlib"] = {
            "available": True,
            "version": matplotlib.__version__
        }
    except ImportError:
        validation_results["dependencies"]["matplotlib"] = {
            "available": False,
            "error": "Matplotlib not available"
        }
        validation_results["warnings"].append("Matplotlib not available - visualizations will be disabled")
        validation_results["recommendations"].append("Install matplotlib for visualization support: pip install matplotlib")
    
    return validation_results


def get_pymdp_health_status() -> Dict[str, Any]:
    """Get comprehensive health status for PyMDP execution."""
    health_status = validate_pymdp_environment()
    
    # Add runtime health checks
    health_status["runtime_checks"] = {
        "timestamp": datetime.now().isoformat(),
        "can_execute": health_status["pymdp_available"] and health_status["dependencies"]["numpy"]["available"],
        "can_visualize": health_status["dependencies"]["matplotlib"]["available"],
        "critical_errors": len(health_status["errors"]),
        "warnings": len(health_status["warnings"])
    }
    
    return health_status


def create_enhanced_pymdp_context(
    gnn_spec: Dict[str, Any],
    output_dir: Path,
    correlation_id: str = "",
    config_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create enhanced execution context for PyMDP simulation.
    
    Args:
        gnn_spec: GNN specification dictionary
        output_dir: Output directory for results
        correlation_id: Correlation ID for tracking
        config_overrides: Configuration overrides
        
    Returns:
        Enhanced execution context dictionary
    """
    context = {
        "correlation_id": correlation_id,
        "gnn_spec": gnn_spec,
        "output_dir": output_dir,
        "start_time": datetime.now(),
        "config": {
            "num_episodes": 20,
            "max_steps_per_episode": 30,
            "planning_horizon": 3,
            "verbose_output": True,
            "save_visualizations": True,
            "random_seed": 42,
            "use_gnn_matrices": True,
            "enable_logging": True,
            "safety_checks": True
        },
        "health_status": get_pymdp_health_status(),
        "execution_metadata": {
            "model_name": gnn_spec.get("model_name", "unknown"),
            "has_gnn_matrices": False,
            "matrix_sources": [],
            "parameter_count": 0
        }
    }
    
    # Apply configuration overrides
    if config_overrides:
        context["config"].update(config_overrides)
    
    # Analyze GNN specification
    if "parsed_model_file" in gnn_spec:
        try:
            import json
            with open(gnn_spec["parsed_model_file"], 'r') as f:
                parsed_gnn = json.load(f)
            
            # Check for matrices in InitialParameterization
            if "InitialParameterization" in parsed_gnn:
                matrices = parsed_gnn["InitialParameterization"]
                available_matrices = []
                for matrix_name in ["A", "B", "C", "D", "E"]:
                    if matrix_name in matrices:
                        available_matrices.append(matrix_name)
                
                context["execution_metadata"]["has_gnn_matrices"] = len(available_matrices) > 0
                context["execution_metadata"]["matrix_sources"] = available_matrices
                
                if available_matrices:
                    logger.info(f"[{correlation_id}] GNN matrices available: {available_matrices}")
                else:
                    logger.warning(f"[{correlation_id}] No GNN matrices found in InitialParameterization")
        except Exception as e:
            logger.warning(f"[{correlation_id}] Failed to analyze GNN specification: {e}")
    
    return context


def execute_pymdp_simulation_from_gnn(
    gnn_spec: Dict[str, Any],
    output_dir: Path,
    config_overrides: Optional[Dict[str, Any]] = None,
    correlation_id: str = ""
) -> Tuple[bool, Dict[str, Any]]:
    """
    Execute PyMDP simulation from GNN specification with enhanced safety patterns.
    
    Args:
        gnn_spec: GNN specification dictionary
        output_dir: Output directory for results
        config_overrides: Configuration overrides
        correlation_id: Correlation ID for tracking
        
    Returns:
        Tuple of (success, results) where results contains execution details
    """
    if not correlation_id:
        import uuid
        correlation_id = str(uuid.uuid4())[:8]
    
    logger.info(f"[{correlation_id}] Starting enhanced PyMDP simulation from GNN")
    
    # Create enhanced execution context
    context = create_enhanced_pymdp_context(gnn_spec, output_dir, correlation_id, config_overrides)
    
    # Check health status
    if not context["health_status"]["runtime_checks"]["can_execute"]:
        error_msg = "PyMDP execution environment not ready"
        logger.error(f"[{correlation_id}] {error_msg}")
        return False, {
            "error": error_msg,
            "health_status": context["health_status"],
            "correlation_id": correlation_id
        }
    
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize PyMDP simulation
        simulation = PyMDPSimulation(
            output_dir=output_dir,
            verbose=context["config"]["verbose_output"],
            correlation_id=correlation_id
        )
        
        # Set GNN matrices if available
        if context["execution_metadata"]["has_gnn_matrices"]:
            logger.info(f"[{correlation_id}] Using GNN matrices for authentic simulation")
            simulation.set_gnn_matrices(gnn_spec)
        else:
            logger.info(f"[{correlation_id}] Using default matrices (no GNN matrices available)")
        
        # Configure simulation parameters
        simulation.configure_simulation(
            num_episodes=context["config"]["num_episodes"],
            max_steps_per_episode=context["config"]["max_steps_per_episode"],
            planning_horizon=context["config"]["planning_horizon"],
            random_seed=context["config"]["random_seed"]
        )
        
        # Run simulation with error handling
        simulation_results = simulation.run_simulation()
        
        # Save results with enhanced metadata
        results = {
            "success": True,
            "correlation_id": correlation_id,
            "execution_time": (datetime.now() - context["start_time"]).total_seconds(),
            "gnn_matrices_used": context["execution_metadata"]["matrix_sources"],
            "simulation_results": simulation_results,
            "health_status": context["health_status"],
            "config_used": context["config"]
        }
        
        # Add simulation metrics if available
        if simulation_results and "episodes" in simulation_results:
            episodes = simulation_results["episodes"]
            results["total_episodes"] = len(episodes)
            results["success_rate"] = sum(1 for ep in episodes if ep.get("success", False)) / len(episodes)
            results["avg_episode_length"] = sum(ep.get("length", 0) for ep in episodes) / len(episodes)
        
        # Generate visualizations if enabled
        if context["config"]["save_visualizations"] and context["health_status"]["runtime_checks"]["can_visualize"]:
            try:
                visualizer = create_visualizer(simulation_results, output_dir)
                save_all_visualizations(visualizer, output_dir)
                results["visualizations_created"] = True
                logger.info(f"[{correlation_id}] Visualizations saved successfully")
            except Exception as e:
                logger.warning(f"[{correlation_id}] Failed to create visualizations: {e}")
                results["visualizations_created"] = False
                results["visualization_error"] = str(e)
        
        # Save comprehensive results
        save_simulation_results(results, output_dir / "enhanced_simulation_results.json")
        
        logger.info(f"[{correlation_id}] PyMDP simulation completed successfully")
        return True, results
        
    except Exception as e:
        error_msg = f"PyMDP simulation failed: {str(e)}"
        logger.error(f"[{correlation_id}] {error_msg}")
        logger.debug(f"[{correlation_id}] Traceback: {traceback.format_exc()}")
        
        return False, {
            "error": error_msg,
            "traceback": traceback.format_exc(),
            "correlation_id": correlation_id,
            "execution_time": (datetime.now() - context["start_time"]).total_seconds(),
            "health_status": context["health_status"]
        }


def execute_pymdp_simulation(
    gnn_spec: Dict[str, Any],
    output_dir: Path,
    correlation_id: str = ""
) -> Tuple[bool, Dict[str, Any]]:
    """
    Execute basic PyMDP simulation (fallback function for backward compatibility).
    
    Args:
        gnn_spec: GNN specification dictionary
        output_dir: Output directory for results
        correlation_id: Correlation ID for tracking
        
    Returns:
        Tuple of (success, results)
    """
    logger.info(f"[{correlation_id}] Executing basic PyMDP simulation")
    
    # Use basic configuration for compatibility
    basic_config = {
        "num_episodes": 10,
        "max_steps_per_episode": 20,
        "planning_horizon": 2,
        "verbose_output": False,
        "save_visualizations": False,
        "use_gnn_matrices": False
    }
    
    return execute_pymdp_simulation_from_gnn(
        gnn_spec=gnn_spec,
        output_dir=output_dir,
        config_overrides=basic_config,
        correlation_id=correlation_id
    )


# Backward compatibility aliases
def execute_pymdp_scripts(*args, **kwargs):
    """Backward compatibility function."""
    logger.warning("execute_pymdp_scripts is deprecated, use execute_pymdp_simulation_from_gnn instead")
    return execute_pymdp_simulation_from_gnn(*args, **kwargs) 