"""
Utils Module MCP Integration

This module exposes utility functions and system information tools through the Model Context Protocol.
It provides tools for system diagnostics, file operations, logging management, and general utilities.

Key Features:
- System information and diagnostics
- File and directory operations
- Logging management and configuration
- Performance tracking and monitoring
- Environment and dependency validation
- Utility function access
"""

import logging
import os
import sys
import platform
import psutil
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

def get_system_info(mcp_instance_ref) -> Dict[str, Any]:
    """
    Get comprehensive system information and diagnostics.
    
    Returns:
        Dictionary containing system information.
    """
    try:
        # System information
        system_info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "python_path": sys.path
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent,
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2)
        }
        
        # CPU information
        cpu_info = {
            "count": psutil.cpu_count(),
            "count_logical": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(interval=1),
            "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
        
        # Disk information
        disk = psutil.disk_usage('/')
        disk_info = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent,
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2)
        }
        
        return {
            "success": True,
            "system": system_info,
            "memory": memory_info,
            "cpu": cpu_info,
            "disk": disk_info,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_environment_info(mcp_instance_ref) -> Dict[str, Any]:
    """
    Get environment information including Python environment and dependencies.
    
    Returns:
        Dictionary containing environment information.
    """
    try:
        # Environment variables
        env_vars = {}
        for key, value in os.environ.items():
            if not any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token']):
                env_vars[key] = value
        
        # Python packages
        try:
            import pkg_resources
            installed_packages = []
            for dist in pkg_resources.working_set:
                installed_packages.append({
                    "name": dist.project_name,
                    "version": dist.version,
                    "location": dist.location
                })
        except ImportError:
            installed_packages = []
        
        # Working directory and paths
        current_dir = Path.cwd()
        home_dir = Path.home()
        
        return {
            "success": True,
            "environment_variables": env_vars,
            "installed_packages": installed_packages,
            "working_directory": str(current_dir),
            "home_directory": str(home_dir),
            "python_path": sys.path,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting environment info: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_file_info(mcp_instance_ref, file_path: str) -> Dict[str, Any]:
    """
    Get detailed information about a file or directory.
    
    Args:
        file_path: Path to the file or directory.
        
    Returns:
        Dictionary containing file information.
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            return {
                "success": False,
                "error": f"Path does not exist: {file_path}"
            }
        
        stat = path.stat()
        
        file_info = {
            "path": str(path),
            "name": path.name,
            "parent": str(path.parent),
            "exists": True,
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "is_symlink": path.is_symlink(),
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024**2), 2),
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "accessed": stat.st_atime,
            "permissions": oct(stat.st_mode)[-3:],
            "owner": stat.st_uid,
            "group": stat.st_gid
        }
        
        # Additional info for directories
        if path.is_dir():
            try:
                contents = list(path.iterdir())
                file_info["contents_count"] = len(contents)
                file_info["contents"] = [item.name for item in contents[:10]]  # First 10 items
                if len(contents) > 10:
                    file_info["contents"].append(f"... and {len(contents) - 10} more")
            except PermissionError:
                file_info["contents_error"] = "Permission denied"
        
        return {
            "success": True,
            "file_info": file_info
        }
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_logging_info(mcp_instance_ref) -> Dict[str, Any]:
    """
    Get current logging configuration and status.
    
    Returns:
        Dictionary containing logging information.
    """
    try:
        # Get root logger info
        root_logger = logging.getLogger()
        
        # Get all loggers
        loggers = {}
        for name in logging.root.manager.loggerDict:
            logger_obj = logging.getLogger(name)
            loggers[name] = {
                "level": logging.getLevelName(logger_obj.level),
                "handlers": len(logger_obj.handlers),
                "propagate": logger_obj.propagate
            }
        
        # Get current logging configuration
        config = {
            "root_level": logging.getLevelName(root_logger.level),
            "root_handlers": len(root_logger.handlers),
            "loggers_count": len(loggers),
            "loggers": loggers
        }
        
        return {
            "success": True,
            "logging_config": config
        }
    except Exception as e:
        logger.error(f"Error getting logging info: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def validate_dependencies(mcp_instance_ref) -> Dict[str, Any]:
    """
    Validate system dependencies and requirements.
    
    Returns:
        Dictionary containing dependency validation results.
    """
    try:
        from .validate_pipeline_dependencies import validate_pipeline_dependencies
        
        # Validate pipeline dependencies
        pipeline_validation = validate_pipeline_dependencies()
        
        # Check for common Python packages
        required_packages = [
            "numpy", "matplotlib", "networkx", "pandas", "requests",
            "pathlib", "json", "logging", "typing", "psutil"
        ]
        
        package_status = {}
        for package in required_packages:
            try:
                __import__(package)
                package_status[package] = "available"
            except ImportError:
                package_status[package] = "missing"
        
        return {
            "success": True,
            "pipeline_validation": pipeline_validation,
            "package_status": package_status,
            "missing_packages": [pkg for pkg, status in package_status.items() if status == "missing"]
        }
    except Exception as e:
        logger.error(f"Error validating dependencies: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def register_tools(mcp_instance):
    """
    Register utility tools with the MCP server.
    
    Args:
        mcp_instance: The MCP instance to register tools with.
    """
    logger.info("Registering utils MCP tools")
    
    # Create wrapper functions
    def get_system_info_wrapper():
        return get_system_info(mcp_instance)
    
    def get_environment_info_wrapper():
        return get_environment_info(mcp_instance)
    
    def get_file_info_wrapper(file_path: str):
        return get_file_info(mcp_instance, file_path)
    
    def get_logging_info_wrapper():
        return get_logging_info(mcp_instance)
    
    def validate_dependencies_wrapper():
        return validate_dependencies(mcp_instance)
    
    # Register tools
    mcp_instance.register_tool(
        name="get_system_info",
        function=get_system_info_wrapper,
        schema={},
        description="Get comprehensive system information including CPU, memory, disk, and platform details."
    )
    
    mcp_instance.register_tool(
        name="get_environment_info",
        function=get_environment_info_wrapper,
        schema={},
        description="Get environment information including Python packages, environment variables, and paths."
    )
    
    mcp_instance.register_tool(
        name="get_file_info",
        function=get_file_info_wrapper,
        schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file or directory to analyze"
                }
            },
            "required": ["file_path"]
        },
        description="Get detailed information about a file or directory including size, permissions, and contents."
    )
    
    mcp_instance.register_tool(
        name="get_logging_info",
        function=get_logging_info_wrapper,
        schema={},
        description="Get current logging configuration and status for all loggers."
    )
    
    mcp_instance.register_tool(
        name="validate_dependencies",
        function=validate_dependencies_wrapper,
        schema={},
        description="Validate system dependencies and required packages for the GNN pipeline."
    )
    
    logger.info("Successfully registered utils MCP tools") 