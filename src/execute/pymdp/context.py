#!/usr/bin/env python3
"""
PyMDP Context module for GNN Processing Pipeline.

This module provides PyMDP context creation capabilities.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def create_enhanced_pymdp_context(
    gnn_spec: Dict[str, Any],
    output_dir: Path,
    correlation_id: str = "",
    config_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create enhanced PyMDP execution context.
    
    Args:
        gnn_spec: GNN specification dictionary
        output_dir: Output directory for results
        correlation_id: Correlation ID for tracking
        config_overrides: Optional configuration overrides
        
    Returns:
        Dictionary with enhanced context
    """
    try:
        logger.info(f"Creating enhanced PyMDP context (correlation_id: {correlation_id})")
        
        # Base context
        context = {
            "gnn_spec": gnn_spec,
            "output_dir": output_dir,
            "correlation_id": correlation_id,
            "timestamp": datetime.now().isoformat(),
            "config": {}
        }
        
        # Extract configuration from GNN spec
        if "config" in gnn_spec:
            context["config"].update(gnn_spec["config"])
        
        # Apply overrides
        if config_overrides:
            context["config"].update(config_overrides)
        
        # Set default configuration
        default_config = {
            "simulation_steps": 100,
            "visualization_enabled": True,
            "save_results": True,
            "log_level": "INFO"
        }
        
        for key, value in default_config.items():
            if key not in context["config"]:
                context["config"][key] = value
        
        # Validate context
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        
        # Add environment information
        try:
            import pymdp
            context["environment"] = {
                "pymdp_version": getattr(pymdp, "__version__", "unknown"),
                "pymdp_available": True
            }
        except ImportError:
            context["environment"] = {
                "pymdp_version": "not_available",
                "pymdp_available": False
            }
        
        logger.info(f"Enhanced PyMDP context created (correlation_id: {correlation_id})")
        return context
        
    except Exception as e:
        logger.error(f"Failed to create enhanced PyMDP context: {e}")
        return {
            "error": str(e),
            "correlation_id": correlation_id,
            "timestamp": datetime.now().isoformat()
        }
