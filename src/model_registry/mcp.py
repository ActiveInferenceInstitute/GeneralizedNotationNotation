"""
Model Registry MCP Integration

This module provides Model Context Protocol integration for the model registry.
It registers tools that can be used by MCP-enabled applications to interact with the model registry.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

def register_tools(registry):
    """
    Register all model registry tools with the MCP registry.
    
    Args:
        registry: The MCP tool registry
    """
    try:
        # Register register_model tool
        registry.register_tool(
            name="model_registry.register_model",
            description="Register a model in the model registry",
            function=register_model,
            parameters=[
                {
                    "name": "model_path",
                    "description": "Path to the model file",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "registry_path",
                    "description": "Path to the registry file",
                    "type": "string",
                    "required": False,
                    "default": "output/model_registry/model_registry.json"
                }
            ],
            returns={
                "type": "object",
                "description": "Registration result with model ID and status"
            },
            examples=[
                {
                    "description": "Register a model",
                    "code": 'model_registry.register_model("input/gnn_files/model.md")'
                }
            ]
        )
        
        # Register get_model tool
        registry.register_tool(
            name="model_registry.get_model",
            description="Get a model from the registry by ID",
            function=get_model,
            parameters=[
                {
                    "name": "model_id",
                    "description": "Model ID",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "registry_path",
                    "description": "Path to the registry file",
                    "type": "string",
                    "required": False,
                    "default": "output/model_registry/model_registry.json"
                }
            ],
            returns={
                "type": "object",
                "description": "Model entry with metadata and versions"
            },
            examples=[
                {
                    "description": "Get a model by ID",
                    "code": 'model_registry.get_model("my_model")'
                }
            ]
        )
        
        # Register search_models tool
        registry.register_tool(
            name="model_registry.search_models",
            description="Search models in the registry by name, description, or tags",
            function=search_models,
            parameters=[
                {
                    "name": "query",
                    "description": "Search query",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "registry_path",
                    "description": "Path to the registry file",
                    "type": "string",
                    "required": False,
                    "default": "output/model_registry/model_registry.json"
                }
            ],
            returns={
                "type": "array",
                "description": "List of matching model entries"
            },
            examples=[
                {
                    "description": "Search models",
                    "code": 'model_registry.search_models("POMDP")'
                }
            ]
        )
        
        # Register list_models tool
        registry.register_tool(
            name="model_registry.list_models",
            description="List all models in the registry",
            function=list_models,
            parameters=[
                {
                    "name": "registry_path",
                    "description": "Path to the registry file",
                    "type": "string",
                    "required": False,
                    "default": "output/model_registry/model_registry.json"
                }
            ],
            returns={
                "type": "array",
                "description": "List of all model entries"
            },
            examples=[
                {
                    "description": "List all models",
                    "code": 'model_registry.list_models()'
                }
            ]
        )
        
        logger.info("Successfully registered model registry MCP tools")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register model registry MCP tools: {e}")
        return False

def register_model(model_path: str, registry_path: str = "output/model_registry/model_registry.json") -> Dict[str, Any]:
    """
    Register a model in the registry.
    
    Args:
        model_path: Path to the model file
        registry_path: Path to the registry file
        
    Returns:
        Registration result with model ID and status
    """
    try:
        from .registry import ModelRegistry
        
        # Convert string paths to Path objects
        model_path = Path(model_path)
        registry_path = Path(registry_path)
        
        # Ensure registry directory exists
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize registry
        registry = ModelRegistry(registry_path)
        
        # Register model
        success = registry.register_model(model_path)
        
        if success:
            # Save registry
            registry.save()
            
            # Get model ID
            model_id = model_path.stem
            
            # Get model entry
            model = registry.get_model(model_id)
            
            if model:
                return {
                    "status": "success",
                    "model_id": model_id,
                    "model_name": model.name,
                    "current_version": model.current_version,
                    "registry_path": str(registry_path)
                }
        
        return {
            "status": "error",
            "model_path": str(model_path),
            "error": "Failed to register model"
        }
        
    except Exception as e:
        logger.error(f"Failed to register model {model_path}: {e}")
        return {
            "status": "error",
            "model_path": str(model_path),
            "error": str(e)
        }

def get_model(model_id: str, registry_path: str = "output/model_registry/model_registry.json") -> Dict[str, Any]:
    """
    Get a model from the registry by ID.
    
    Args:
        model_id: Model ID
        registry_path: Path to the registry file
        
    Returns:
        Model entry with metadata and versions
    """
    try:
        from .registry import ModelRegistry
        
        # Convert string path to Path object
        registry_path = Path(registry_path)
        
        # Initialize registry
        registry = ModelRegistry(registry_path)
        
        # Get model
        model = registry.get_model(model_id)
        
        if model:
            return {
                "status": "success",
                "model_id": model.model_id,
                "model_name": model.name,
                "description": model.description,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
                "current_version": model.current_version,
                "versions": [v for v in model.versions.keys()],
                "tags": model.tags,
                "metadata": model.metadata
            }
        
        return {
            "status": "error",
            "model_id": model_id,
            "error": "Model not found"
        }
        
    except Exception as e:
        logger.error(f"Failed to get model {model_id}: {e}")
        return {
            "status": "error",
            "model_id": model_id,
            "error": str(e)
        }

def search_models(query: str, registry_path: str = "output/model_registry/model_registry.json") -> List[Dict[str, Any]]:
    """
    Search models in the registry by name, description, or tags.
    
    Args:
        query: Search query
        registry_path: Path to the registry file
        
    Returns:
        List of matching model entries
    """
    try:
        from .registry import ModelRegistry
        
        # Convert string path to Path object
        registry_path = Path(registry_path)
        
        # Initialize registry
        registry = ModelRegistry(registry_path)
        
        # Search models
        models = registry.search_models(query)
        
        # Convert to dictionaries
        return [
            {
                "model_id": model.model_id,
                "model_name": model.name,
                "description": model.description,
                "current_version": model.current_version,
                "tags": model.tags
            }
            for model in models
        ]
        
    except Exception as e:
        logger.error(f"Failed to search models: {e}")
        return [
            {
                "status": "error",
                "error": str(e)
            }
        ]

def list_models(registry_path: str = "output/model_registry/model_registry.json") -> List[Dict[str, Any]]:
    """
    List all models in the registry.
    
    Args:
        registry_path: Path to the registry file
        
    Returns:
        List of all model entries
    """
    try:
        from .registry import ModelRegistry
        
        # Convert string path to Path object
        registry_path = Path(registry_path)
        
        # Initialize registry
        registry = ModelRegistry(registry_path)
        
        # List models
        models = registry.list_models()
        
        # Convert to dictionaries
        return [
            {
                "model_id": model.model_id,
                "model_name": model.name,
                "description": model.description,
                "current_version": model.current_version,
                "tags": model.tags
            }
            for model in models
        ]
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return [
            {
                "status": "error",
                "error": str(e)
            }
        ] 