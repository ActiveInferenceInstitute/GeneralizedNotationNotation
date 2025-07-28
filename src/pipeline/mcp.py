"""
Pipeline Module MCP Integration

This module exposes pipeline management and configuration tools through the Model Context Protocol.
It provides tools for pipeline discovery, execution, monitoring, and configuration management.

Key Features:
- Pipeline step discovery and metadata
- Pipeline execution and monitoring
- Configuration management and validation
- Pipeline status and performance tracking
- Step dependency analysis and validation
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

def get_pipeline_steps(mcp_instance_ref) -> Dict[str, Any]:
    """
    Get information about all available pipeline steps.
    
    Returns:
        Dictionary containing pipeline step information.
    """
    try:
        from .pipeline_config import STEP_METADATA, get_pipeline_config
        
        steps_info = {}
        for step_name, metadata in STEP_METADATA.items():
            steps_info[step_name] = {
                "name": step_name,
                "script": metadata.get("script", ""),
                "module": metadata.get("module", ""),
                "description": metadata.get("description", ""),
                "required": metadata.get("required", False),
                "category": metadata.get("category", "General"),
                "dependencies": metadata.get("dependencies", []),
                "output_dir": metadata.get("output_dir", ""),
                "version": metadata.get("version", "1.0.0")
            }
        
        return {
            "success": True,
            "total_steps": len(steps_info),
            "steps": steps_info,
            "pipeline_config": get_pipeline_config()
        }
    except Exception as e:
        logger.error(f"Error getting pipeline steps: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_pipeline_status(mcp_instance_ref) -> Dict[str, Any]:
    """
    Get current pipeline execution status and statistics.
    
    Returns:
        Dictionary containing pipeline status information.
    """
    try:
        from .pipeline_config import get_pipeline_config
        
        config = get_pipeline_config()
        output_dir = Path(config.get("output_dir", "output"))
        
        # Check for pipeline execution summary
        summary_file = output_dir / "pipeline_execution_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {
                "last_execution": None,
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0
            }
        
        # Check for recent logs
        logs_dir = output_dir / "logs"
        recent_logs = []
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.log"))
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for log_file in log_files[:5]:  # Last 5 log files
                recent_logs.append({
                    "file": log_file.name,
                    "size": log_file.stat().st_size,
                    "modified": time.ctime(log_file.stat().st_mtime)
                })
        
        return {
            "success": True,
            "pipeline_config": config,
            "execution_summary": summary,
            "recent_logs": recent_logs,
            "output_directory": str(output_dir),
            "output_directory_exists": output_dir.exists()
        }
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def validate_pipeline_dependencies(mcp_instance_ref) -> Dict[str, Any]:
    """
    Validate pipeline step dependencies and identify any issues.
    
    Returns:
        Dictionary containing dependency validation results.
    """
    try:
        from .pipeline_config import STEP_METADATA
        from ..utils.validate_pipeline_dependencies import validate_pipeline_dependencies as validate_deps
        
        validation_result = validate_deps()
        
        # Additional analysis
        dependency_graph = {}
        missing_deps = []
        circular_deps = []
        
        for step_name, metadata in STEP_METADATA.items():
            deps = metadata.get("dependencies", [])
            dependency_graph[step_name] = deps
            
            # Check for missing dependencies
            for dep in deps:
                if dep not in STEP_METADATA:
                    missing_deps.append({
                        "step": step_name,
                        "missing_dependency": dep
                    })
        
        return {
            "success": True,
            "validation_result": validation_result,
            "dependency_graph": dependency_graph,
            "missing_dependencies": missing_deps,
            "circular_dependencies": circular_deps,
            "total_steps": len(STEP_METADATA),
            "total_dependencies": sum(len(metadata.get("dependencies", [])) for metadata in STEP_METADATA.values())
        }
    except Exception as e:
        logger.error(f"Error validating pipeline dependencies: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_pipeline_config_info(mcp_instance_ref) -> Dict[str, Any]:
    """
    Get detailed pipeline configuration information.
    
    Returns:
        Dictionary containing pipeline configuration details.
    """
    try:
        from .pipeline_config import get_pipeline_config
        
        config = get_pipeline_config()
        
        return {
            "success": True,
            "configuration": config,
            "configuration_keys": list(config.keys()),
            "output_directory": config.get("output_dir", "output"),
            "log_level": config.get("log_level", "INFO"),
            "parallel_execution": config.get("parallel_execution", False),
            "max_workers": config.get("max_workers", 1)
        }
    except Exception as e:
        logger.error(f"Error getting pipeline config info: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def register_tools(mcp_instance):
    """
    Register pipeline management tools with the MCP server.
    
    Args:
        mcp_instance: The MCP instance to register tools with.
    """
    logger.info("Registering pipeline MCP tools")
    
    # Create wrapper functions
    def get_steps_wrapper():
        return get_pipeline_steps(mcp_instance)
    
    def get_status_wrapper():
        return get_pipeline_status(mcp_instance)
    
    def validate_deps_wrapper():
        return validate_pipeline_dependencies(mcp_instance)
    
    def get_config_wrapper():
        return get_pipeline_config_info(mcp_instance)
    
    # Register tools
    mcp_instance.register_tool(
        name="get_pipeline_steps",
        function=get_steps_wrapper,
        schema={},
        description="Get information about all available pipeline steps, their metadata, and dependencies."
    )
    
    mcp_instance.register_tool(
        name="get_pipeline_status",
        function=get_status_wrapper,
        schema={},
        description="Get current pipeline execution status, recent logs, and execution statistics."
    )
    
    mcp_instance.register_tool(
        name="validate_pipeline_dependencies",
        function=validate_deps_wrapper,
        schema={},
        description="Validate pipeline step dependencies and identify missing or circular dependencies."
    )
    
    mcp_instance.register_tool(
        name="get_pipeline_config_info",
        function=get_config_wrapper,
        schema={},
        description="Get detailed pipeline configuration information and settings."
    )
    
    logger.info("Successfully registered pipeline MCP tools") 