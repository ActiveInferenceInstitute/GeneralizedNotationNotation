#!/usr/bin/env python3
"""
PyMDP Validator module for GNN Processing Pipeline.

This module provides PyMDP environment validation capabilities.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

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
        validation_results["errors"].append(
            "PyMDP framework not available - this is expected if PyMDP is not installed. "
            "To enable PyMDP: install via pip (pip install pymdp) or uv (uv pip install pymdp). "
            "Alternative frameworks still available: RxInfer.jl, ActiveInference.jl, JAX, DisCoPy."
        )
        logger.error(
            "PyMDP not available for execution - using fallback analysis. "
            "Install PyMDP if you want to run PyMDP-based simulations: pip install pymdp"
        )
    
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
        validation_results["errors"].append("NumPy not available - install with: uv pip install numpy")
    
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
    
    # Check scipy for advanced computations
    try:
        import scipy
        validation_results["dependencies"]["scipy"] = {
            "available": True,
            "version": scipy.__version__
        }
    except ImportError:
        validation_results["dependencies"]["scipy"] = {
            "available": False,
            "error": "SciPy not available"
        }
        validation_results["warnings"].append("SciPy not available - some advanced features may be limited")
    
    # Overall health assessment
    if validation_results["pymdp_available"]:
        validation_results["overall_health"] = "healthy"
        validation_results["recommendations"].append("Environment ready for PyMDP simulations")
    else:
        validation_results["overall_health"] = "unhealthy"
        validation_results["recommendations"].append("Install PyMDP to enable simulations")
    
    return validation_results

def get_pymdp_health_status() -> Dict[str, Any]:
    """
    Get PyMDP health status and environment information.
    
    Returns:
        Dictionary with health status information
    """
    validation_results = validate_pymdp_environment()
    
    health_status = {
        "status": "unknown",
        "pymdp_available": validation_results["pymdp_available"],
        "dependencies_healthy": len(validation_results["errors"]) == 0,
        "warnings_count": len(validation_results["warnings"]),
        "errors_count": len(validation_results["errors"]),
        "recommendations": validation_results["recommendations"]
    }
    
    if validation_results["pymdp_available"] and len(validation_results["errors"]) == 0:
        health_status["status"] = "healthy"
    elif validation_results["pymdp_available"] and len(validation_results["warnings"]) > 0:
        health_status["status"] = "warning"
    else:
        health_status["status"] = "unhealthy"
    
    return health_status
