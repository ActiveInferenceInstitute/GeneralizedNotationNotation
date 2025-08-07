#!/usr/bin/env python3
"""
Setup Validator module for GNN Processing Pipeline.

This module provides setup validation capabilities.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def validate_system() -> Dict[str, Any]:
    """
    Validate the system requirements for GNN with UV support.
    
    Returns:
        Dictionary with system validation results
    """
    try:
        from .setup import check_system_requirements, check_uv_availability
        return {
            "success": check_system_requirements() and check_uv_availability(),
            "message": "System validation completed with UV support"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

def get_environment_info() -> Dict[str, Any]:
    """
    Get comprehensive environment information.
    
    Returns:
        Dictionary with environment information
    """
    try:
        from .setup import get_uv_setup_info, get_installed_package_versions
        
        uv_info = get_uv_setup_info()
        package_versions = get_installed_package_versions()
        
        return {
            "uv_info": uv_info,
            "package_versions": package_versions,
            "status": "healthy"
        }
        
    except Exception as e:
        logger.error(f"Failed to get environment info: {e}")
        return {
            "error": str(e),
            "status": "error"
        }

def get_uv_status() -> Dict[str, Any]:
    """
    Get UV status and configuration.
    
    Returns:
        Dictionary with UV status information
    """
    try:
        from .setup import check_uv_availability, get_uv_setup_info
        
        uv_available = check_uv_availability()
        uv_info = get_uv_setup_info()
        
        return {
            "uv_available": uv_available,
            "uv_info": uv_info,
            "status": "healthy" if uv_available else "unavailable"
        }
        
    except Exception as e:
        logger.error(f"Failed to get UV status: {e}")
        return {
            "uv_available": False,
            "error": str(e),
            "status": "error"
        }
