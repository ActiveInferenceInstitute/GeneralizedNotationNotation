#!/usr/bin/env python3
"""
PyMDP Validator module for GNN Processing Pipeline.

This module provides PyMDP environment validation capabilities.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .package_detector import (
    detect_pymdp_installation,
    get_pymdp_installation_instructions,
    is_correct_pymdp_package
)

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
    
    # Check PyMDP availability using package detector
    detection = detect_pymdp_installation()
    
    if detection.get("correct_package"):
        validation_results["pymdp_available"] = True
        validation_results["dependencies"]["pymdp"] = {
            "available": True,
            "version": detection.get("version", "unknown"),
            "package_name": "inferactively-pymdp",
            "correct_variant": True
        }
        logger.info(f"PyMDP (inferactively-pymdp) available: version {detection.get('version', 'unknown')}")
    elif detection.get("wrong_package"):
        validation_results["pymdp_available"] = False
        validation_results["dependencies"]["pymdp"] = {
            "available": False,
            "error": "Wrong PyMDP package variant installed",
            "wrong_variant": True,
            "has_mdp_solver": detection.get("has_mdp_solver", False)
        }
        instructions = get_pymdp_installation_instructions()
        validation_results["errors"].append(instructions)
        validation_results["errors"].append(
            "Alternative frameworks still available: RxInfer.jl, ActiveInference.jl, JAX, DisCoPy."
        )
        logger.error(
            "Wrong PyMDP package detected - install inferactively-pymdp instead. "
            f"Instructions: {instructions}"
        )
    elif detection.get("installed"):
        # Package installed but variant unclear
        validation_results["pymdp_available"] = False
        validation_results["dependencies"]["pymdp"] = {
            "available": False,
            "error": "PyMDP package variant unclear",
            "version": detection.get("version", "unknown")
        }
        instructions = get_pymdp_installation_instructions()
        validation_results["warnings"].append(instructions)
        logger.warning(f"PyMDP package status unclear: {instructions}")
    else:
        # Not installed
        validation_results["pymdp_available"] = False
        validation_results["dependencies"]["pymdp"] = {
            "available": False,
            "error": detection.get("error", "PyMDP not installed")
        }
        instructions = get_pymdp_installation_instructions()
        validation_results["errors"].append(instructions)
        validation_results["errors"].append(
            "Alternative frameworks still available: RxInfer.jl, ActiveInference.jl, JAX, DisCoPy."
        )
        logger.info(
            "PyMDP not available for execution - using fallback analysis. "
            f"Install with: {instructions}"
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
        pymdp_dep = validation_results["dependencies"].get("pymdp", {})
        if pymdp_dep.get("wrong_variant"):
            validation_results["overall_health"] = "unhealthy"
            validation_results["recommendations"].append(
                "Uninstall wrong PyMDP package and install inferactively-pymdp"
            )
        else:
            validation_results["overall_health"] = "unhealthy"
            validation_results["recommendations"].append(
                "Install inferactively-pymdp to enable PyMDP simulations"
            )
    
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
