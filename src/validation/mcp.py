"""
Validation MCP Integration

This module provides Model Context Protocol integration for the validation step.
It registers tools that can be used by MCP-enabled applications to validate GNN models.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

def register_tools(registry):
    """
    Register all validation tools with the MCP registry.
    
    Args:
        registry: The MCP tool registry
    """
    try:
        # Register validate_model tool
        registry.register_tool(
            name="validation.validate_model",
            description="Validate a GNN model for semantic correctness, performance, and consistency",
            function=validate_model,
            parameters=[
                {
                    "name": "model_path",
                    "description": "Path to the GNN model file",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "validation_level",
                    "description": "Validation level (basic, standard, strict, research)",
                    "type": "string",
                    "required": False,
                    "default": "standard"
                },
                {
                    "name": "profile_performance",
                    "description": "Whether to profile performance",
                    "type": "boolean",
                    "required": False,
                    "default": True
                },
                {
                    "name": "check_consistency",
                    "description": "Whether to check consistency",
                    "type": "boolean",
                    "required": False,
                    "default": True
                }
            ],
            returns={
                "type": "object",
                "description": "Validation result with errors, warnings, and metrics"
            },
            examples=[
                {
                    "description": "Validate a model with standard validation",
                    "code": 'validation.validate_model("input/gnn_files/model.md")'
                },
                {
                    "description": "Validate a model with strict validation",
                    "code": 'validation.validate_model("input/gnn_files/model.md", validation_level="strict")'
                }
            ]
        )
        
        # Register validate_semantic tool
        registry.register_tool(
            name="validation.validate_semantic",
            description="Validate the semantic aspects of a GNN model",
            function=validate_semantic,
            parameters=[
                {
                    "name": "model_path",
                    "description": "Path to the GNN model file",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "validation_level",
                    "description": "Validation level (basic, standard, strict, research)",
                    "type": "string",
                    "required": False,
                    "default": "standard"
                }
            ],
            returns={
                "type": "object",
                "description": "Semantic validation result with errors and warnings"
            },
            examples=[
                {
                    "description": "Validate the semantics of a model",
                    "code": 'validation.validate_semantic("input/gnn_files/model.md")'
                }
            ]
        )
        
        # Register profile_performance tool
        registry.register_tool(
            name="validation.profile_performance",
            description="Profile the performance characteristics of a GNN model",
            function=profile_performance,
            parameters=[
                {
                    "name": "model_path",
                    "description": "Path to the GNN model file",
                    "type": "string",
                    "required": True
                }
            ],
            returns={
                "type": "object",
                "description": "Performance profile with metrics and warnings"
            },
            examples=[
                {
                    "description": "Profile the performance of a model",
                    "code": 'validation.profile_performance("input/gnn_files/model.md")'
                }
            ]
        )
        
        # Register check_consistency tool
        registry.register_tool(
            name="validation.check_consistency",
            description="Check the consistency of a GNN model",
            function=check_consistency,
            parameters=[
                {
                    "name": "model_path",
                    "description": "Path to the GNN model file",
                    "type": "string",
                    "required": True
                }
            ],
            returns={
                "type": "object",
                "description": "Consistency check result with warnings"
            },
            examples=[
                {
                    "description": "Check the consistency of a model",
                    "code": 'validation.check_consistency("input/gnn_files/model.md")'
                }
            ]
        )
        
        logger.info("Successfully registered validation MCP tools")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register validation MCP tools: {e}")
        return False

def validate_model(model_path: str, validation_level: str = "standard", profile_performance: bool = True, check_consistency: bool = True) -> Dict[str, Any]:
    """
    Validate a GNN model for semantic correctness, performance, and consistency.
    
    Args:
        model_path: Path to the GNN model file
        validation_level: Validation level (basic, standard, strict, research)
        profile_performance: Whether to profile performance
        check_consistency: Whether to check consistency
        
    Returns:
        Validation result with errors, warnings, and metrics
    """
    try:
        from .semantic_validator import SemanticValidator
        from .performance_profiler import PerformanceProfiler
        from .consistency_checker import ConsistencyChecker
        
        # Convert string path to Path object
        model_path = Path(model_path)
        
        # Read model content
        with open(model_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = {
            "file_path": str(model_path),
            "file_name": model_path.name,
            "validation_level": validation_level,
            "semantic_validation": {},
            "performance_profile": {},
            "consistency_check": {},
            "warnings": [],
            "errors": [],
            "status": "unknown"
        }
        
        # Semantic validation
        semantic_validator = SemanticValidator(validation_level)
        semantic_result = semantic_validator.validate(content)
        result["semantic_validation"] = semantic_result
        
        if not semantic_result.get("is_valid", False):
            result["errors"].extend(semantic_result.get("errors", []))
        
        # Performance profiling
        if profile_performance:
            performance_profiler = PerformanceProfiler()
            profile_result = performance_profiler.profile(content)
            result["performance_profile"] = profile_result
            
            if profile_result.get("warnings", []):
                result["warnings"].extend(profile_result.get("warnings", []))
        
        # Consistency checking
        if check_consistency:
            consistency_checker = ConsistencyChecker()
            consistency_result = consistency_checker.check(content)
            result["consistency_check"] = consistency_result
            
            if not consistency_result.get("is_consistent", False):
                result["warnings"].extend(consistency_result.get("warnings", []))
        
        # Determine overall status
        if result["errors"]:
            result["status"] = "failed"
        elif result["warnings"]:
            result["status"] = "warnings"
        else:
            result["status"] = "passed"
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to validate model {model_path}: {e}")
        return {
            "status": "error",
            "file_path": str(model_path),
            "error": str(e)
        }

def validate_semantic(model_path: str, validation_level: str = "standard") -> Dict[str, Any]:
    """
    Validate the semantic aspects of a GNN model.
    
    Args:
        model_path: Path to the GNN model file
        validation_level: Validation level (basic, standard, strict, research)
        
    Returns:
        Semantic validation result with errors and warnings
    """
    try:
        from .semantic_validator import SemanticValidator
        
        # Convert string path to Path object
        model_path = Path(model_path)
        
        # Read model content
        with open(model_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Semantic validation
        semantic_validator = SemanticValidator(validation_level)
        semantic_result = semantic_validator.validate(content)
        
        return {
            "file_path": str(model_path),
            "file_name": model_path.name,
            "validation_level": validation_level,
            "is_valid": semantic_result.get("is_valid", False),
            "errors": semantic_result.get("errors", []),
            "warnings": semantic_result.get("warnings", [])
        }
        
    except Exception as e:
        logger.error(f"Failed to validate semantics of model {model_path}: {e}")
        return {
            "status": "error",
            "file_path": str(model_path),
            "error": str(e)
        }

def profile_performance(model_path: str) -> Dict[str, Any]:
    """
    Profile the performance characteristics of a GNN model.
    
    Args:
        model_path: Path to the GNN model file
        
    Returns:
        Performance profile with metrics and warnings
    """
    try:
        from .performance_profiler import PerformanceProfiler
        
        # Convert string path to Path object
        model_path = Path(model_path)
        
        # Read model content
        with open(model_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Performance profiling
        performance_profiler = PerformanceProfiler()
        profile_result = performance_profiler.profile(content)
        
        return {
            "file_path": str(model_path),
            "file_name": model_path.name,
            "metrics": profile_result.get("metrics", {}),
            "warnings": profile_result.get("warnings", [])
        }
        
    except Exception as e:
        logger.error(f"Failed to profile performance of model {model_path}: {e}")
        return {
            "status": "error",
            "file_path": str(model_path),
            "error": str(e)
        }

def check_consistency(model_path: str) -> Dict[str, Any]:
    """
    Check the consistency of a GNN model.
    
    Args:
        model_path: Path to the GNN model file
        
    Returns:
        Consistency check result with warnings
    """
    try:
        from .consistency_checker import ConsistencyChecker
        
        # Convert string path to Path object
        model_path = Path(model_path)
        
        # Read model content
        with open(model_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Consistency checking
        consistency_checker = ConsistencyChecker()
        consistency_result = consistency_checker.check(content)
        
        return {
            "file_path": str(model_path),
            "file_name": model_path.name,
            "is_consistent": consistency_result.get("is_consistent", False),
            "warnings": consistency_result.get("warnings", []),
            "checks": consistency_result.get("checks", {})
        }
        
    except Exception as e:
        logger.error(f"Failed to check consistency of model {model_path}: {e}")
        return {
            "status": "error",
            "file_path": str(model_path),
            "error": str(e)
        } 