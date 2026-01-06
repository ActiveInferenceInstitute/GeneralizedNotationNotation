#!/usr/bin/env python3
"""
PyMDP Executor module for GNN Processing Pipeline.

This module provides PyMDP simulation execution capabilities.
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

logger = logging.getLogger(__name__)

def execute_pymdp_simulation_from_gnn(
    gnn_spec: Dict[str, Any],
    output_dir: Path,
    config_overrides: Optional[Dict[str, Any]] = None,
    correlation_id: str = ""
) -> Tuple[bool, Dict[str, Any]]:
    """
    Execute PyMDP simulation from GNN specification.
    
    Args:
        gnn_spec: GNN specification dictionary
        output_dir: Output directory for results
        config_overrides: Optional configuration overrides
        correlation_id: Correlation ID for tracking
        
    Returns:
        Tuple of (success, results)
    """
    try:
        logger.info(f"Starting PyMDP simulation from GNN spec (correlation_id: {correlation_id})")
        
        # Create enhanced context
        context = create_enhanced_pymdp_context(
            gnn_spec, output_dir, correlation_id, config_overrides
        )
        
        # Execute simulation
        success, results = execute_pymdp_simulation(gnn_spec, output_dir, correlation_id)
        
        if success:
            logger.info(f"PyMDP simulation completed successfully (correlation_id: {correlation_id})")
        else:
            logger.error(f"PyMDP simulation failed (correlation_id: {correlation_id})")
        
        return success, results
        
    except Exception as e:
        logger.error(f"PyMDP simulation execution failed: {e}")
        return False, {"error": str(e), "traceback": traceback.format_exc()}

def execute_pymdp_simulation(
    gnn_spec: Dict[str, Any],
    output_dir: Path,
    correlation_id: str = ""
) -> Tuple[bool, Dict[str, Any]]:
    """
    Execute PyMDP simulation using simple tutorial-based approach.
    
    Args:
        gnn_spec: GNN specification dictionary
        output_dir: Output directory for results
        correlation_id: Correlation ID for tracking
        
    Returns:
        Tuple of (success, results)
    """
    try:
        logger.info(f"Executing PyMDP simulation (correlation_id: {correlation_id})")
        
        # Use simple simulation based on official PyMDP tutorial
        from .simple_simulation import run_simple_pymdp_simulation
        
        success, results = run_simple_pymdp_simulation(gnn_spec, output_dir)
        
        if success:
            logger.info(f"PyMDP simulation completed successfully")
            results["correlation_id"] = correlation_id
            return True, results
        else:
            logger.error(f"PyMDP simulation failed: {results.get('error', 'Unknown')}")
            results["correlation_id"] = correlation_id
            return False, results
            
    except Exception as e:
        logger.error(f"PyMDP simulation execution failed: {e}")
        import traceback
        return False, {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "correlation_id": correlation_id
        }
