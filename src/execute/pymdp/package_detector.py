#!/usr/bin/env python3
"""
PyMDP Package Detector module for GNN Processing Pipeline.

This module provides centralized PyMDP package detection and validation capabilities.
It distinguishes between the correct package (inferactively-pymdp) and wrong variants.
"""

import logging
import subprocess
import sys
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def detect_pymdp_installation() -> Dict[str, Any]:
    """
    Detect PyMDP package installation and variant.
    
    Returns:
        Dictionary with detection results:
        - installed: bool - Whether any pymdp package is installed
        - correct_package: bool - Whether correct package (inferactively-pymdp) is installed
        - wrong_package: bool - Whether wrong package variant is installed
        - package_name: str - Name of installed package
        - has_agent: bool - Whether Agent class is available
        - has_mdp_solver: bool - Whether MDP/MDPSolver is present (wrong variant)
        - version: Optional[str] - Package version if available
        - error: Optional[str] - Error message if detection failed
    """
    result = {
        "installed": False,
        "correct_package": False,
        "wrong_package": False,
        "package_name": None,
        "has_agent": False,
        "has_mdp_solver": False,
        "version": None,
        "error": None
    }
    
    try:
        import pymdp
        result["installed"] = True
        result["package_name"] = getattr(pymdp, "__name__", "pymdp")
        result["version"] = getattr(pymdp, "__version__", None)
        
        # Check what's available in the package
        available_attrs = dir(pymdp)
        
        # Check for correct package indicators
        # Modern inferactively-pymdp has Agent in pymdp.agent submodule
        if "Agent" in available_attrs or hasattr(pymdp, "Agent"):
            result["has_agent"] = True
            result["correct_package"] = True
            logger.info("Detected correct PyMDP package (inferactively-pymdp) with Agent class")
        elif "agent" in available_attrs:
            # Check if pymdp.agent submodule has Agent class
            try:
                from pymdp.agent import Agent
                result["has_agent"] = True
                result["correct_package"] = True
                logger.info("Detected correct PyMDP package (inferactively-pymdp) with Agent in agent submodule")
            except ImportError:
                pass
        
        # Check for wrong package indicators
        if "MDP" in available_attrs or "MDPSolver" in available_attrs:
            result["has_mdp_solver"] = True
            if not result["has_agent"]:
                result["wrong_package"] = True
                logger.warning("Detected wrong PyMDP package variant (pymdp with MDP/MDPSolver)")
        
        # If neither Agent nor MDP found, it's unclear
        if not result["has_agent"] and not result["has_mdp_solver"]:
            logger.warning("PyMDP package found but cannot determine variant")
            
    except ImportError:
        result["installed"] = False
        result["error"] = "PyMDP package not installed"
        logger.debug("PyMDP package not found")
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error detecting PyMDP package: {e}")
    
    return result


def is_correct_pymdp_package() -> bool:
    """
    Check if the correct PyMDP package is installed.
    
    Returns:
        True if correct package (inferactively-pymdp) is installed, False otherwise
    """
    detection = detect_pymdp_installation()
    return detection.get("correct_package", False)


def get_pymdp_installation_instructions() -> str:
    """
    Get installation instructions for PyMDP.
    
    Returns:
        String with installation instructions
    """
    detection = detect_pymdp_installation()
    
    if detection.get("correct_package"):
        return "PyMDP (inferactively-pymdp) is correctly installed."
    
    if detection.get("wrong_package"):
        return (
            "Wrong PyMDP package detected. The installed 'pymdp' package contains "
            "MDP/MDPSolver but not the Active Inference Agent class.\n"
            "Install the correct package with:\n"
            "  uv pip install inferactively-pymdp\n"
            "Or using the setup module:\n"
            "  python src/1_setup.py --install_optional --optional_groups pymdp"
        )
    
    if not detection.get("installed"):
        return (
            "PyMDP is not installed.\n"
            "Install with:\n"
            "  uv pip install inferactively-pymdp\n"
            "Or using the setup module:\n"
            "  python src/1_setup.py --install_optional --optional_groups pymdp"
        )
    
    return (
        "PyMDP package status unclear.\n"
        "Try installing the correct package:\n"
        "  uv pip install inferactively-pymdp"
    )


def attempt_pymdp_auto_install(use_uv: bool = True) -> Tuple[bool, str]:
    """
    Attempt to automatically install the correct PyMDP package.
    
    Args:
        use_uv: Whether to use UV package manager (preferred)
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    package_name = "inferactively-pymdp"
    
    try:
        if use_uv:
            logger.info(f"Attempting to install {package_name} using UV...")
            result = subprocess.run(
                [sys.executable, "-m", "uv", "pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully installed {package_name} using UV")
                return True, f"Successfully installed {package_name}"
            else:
                logger.warning(f"UV installation failed, trying pip...")
                # Fall through to pip
                use_uv = False
        
        if not use_uv:
            logger.info(f"Attempting to install {package_name} using pip...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully installed {package_name} using pip")
                return True, f"Successfully installed {package_name}"
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                logger.error(f"Failed to install {package_name}: {error_msg}")
                return False, f"Installation failed: {error_msg}"
                
    except subprocess.TimeoutExpired:
        logger.error(f"Installation of {package_name} timed out")
        return False, "Installation timed out after 120 seconds"
    except Exception as e:
        logger.error(f"Error during {package_name} installation: {e}")
        return False, f"Installation error: {str(e)}"


def validate_pymdp_for_execution() -> Dict[str, Any]:
    """
    Validate that PyMDP is ready for execution.
    
    Returns:
        Dictionary with validation results:
        - ready: bool - Whether PyMDP is ready for execution
        - detection: Dict - Full detection results
        - instructions: str - Installation instructions if not ready
        - can_auto_install: bool - Whether auto-installation is possible
    """
    detection = detect_pymdp_installation()
    
    result = {
        "ready": False,
        "detection": detection,
        "instructions": "",
        "can_auto_install": True
    }
    
    if detection.get("correct_package"):
        result["ready"] = True
        result["instructions"] = "PyMDP is ready for execution"
    elif detection.get("wrong_package"):
        result["instructions"] = get_pymdp_installation_instructions()
        result["can_auto_install"] = True
    elif not detection.get("installed"):
        result["instructions"] = get_pymdp_installation_instructions()
        result["can_auto_install"] = True
    else:
        result["instructions"] = get_pymdp_installation_instructions()
        result["can_auto_install"] = False
    
    return result

