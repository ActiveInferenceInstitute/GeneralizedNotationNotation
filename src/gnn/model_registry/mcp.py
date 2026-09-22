"""
Model Registry MCP Integration

This module provides Model Context Protocol integration for the model registry.
It registers tools that can be used by MCP-enabled applications to interact with the model registry.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

# Set up logging
logger = logging.getLogger(__name__)


def register_tools(registry: Any) -> bool:
    """
    Register all model registry tools with the MCP registry.

    Args:
        registry: The MCP tool registry
    """
    try:
        # Register register_model tool
        registry.register_tool(
            "model_registry.register_model",
            register_model,
            {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to the model file",
                    },
                    "registry_path": {
                        "type": "string",
                        "description": "Path to the registry file",
                        "default": "output/model_registry/model_registry.json",
                    },
                },
                "required": ["model_path"],
            },
            "Register a model in the model registry",
            module=__package__,
            category="model_registry",
            examples=[
                {
                    "description": "Register a model",
                    "code": 'model_registry.register_model("input/gnn_files/model.md")',
                }
            ],
        )

        # Register get_model tool
        registry.register_tool(
            "model_registry.get_model",
            get_model,
            {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model ID",
                    },
                    "registry_path": {
                        "type": "string",
                        "description": "Path to the registry file",
                        "default": "output/model_registry/model_registry.json",
                    },
                },
                "required": ["model_id"],
            },
            "Get a model from the registry by ID",
            module=__package__,
            category="model_registry",
            examples=[
                {
                    "description": "Get a model by ID",
                    "code": 'model_registry.get_model("my_model")',
                }
            ],
        )

        # Register search_models tool
        registry.register_tool(
            "model_registry.search_models",
            search_models,
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "registry_path": {
                        "type": "string",
                        "description": "Path to the registry file",
                        "default": "output/model_registry/model_registry.json",
                    },
                },
                "required": ["query"],
            },
            "Search models in the registry by name, description, or tags",
            module=__package__,
            category="model_registry",
            examples=[
                {
                    "description": "Search models",
                    "code": 'model_registry.search_models("POMDP")',
                }
            ],
        )

        # Register list_models tool
        registry.register_tool(
            "model_registry.list_models",
            list_models,
            {
                "type": "object",
                "properties": {
                    "registry_path": {
                        "type": "string",
                        "description": "Path to the registry file",
                        "default": "output/model_registry/model_registry.json",
                    },
                },
            },
            "List all models in the registry",
            module=__package__,
            category="model_registry",
            examples=[
                {
                    "description": "List all models",
                    "code": "model_registry.list_models()",
                }
            ],
        )

        logger.info("Successfully registered model registry MCP tools")
        return True

    except Exception as e:
        logger.error(f"Failed to register model registry MCP tools: {e}")
        return False


def register_model(
    model_path: str, registry_path: str = "output/model_registry/model_registry.json"
) -> Dict[str, Any]:
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

        model_file = Path(model_path)
        registry_file = Path(registry_path)

        # Ensure registry directory exists
        registry_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize registry
        registry = ModelRegistry(registry_file)

        # Register model
        success = registry.register_model(model_file)

        if success:
            # Save registry
            registry.save()

            # Get model ID
            model_id = model_file.stem

            # Get model entry
            model = registry.get_model(model_id)

            if model:
                return {
                    "status": "success",
                    "model_id": model_id,
                    "model_name": model.name,
                    "current_version": model.current_version,
                    "registry_path": str(registry_path),
                }

        return {
            "status": "error",
            "model_path": str(model_path),
            "error": "Failed to register model",
        }

    except Exception as e:
        logger.error(f"Failed to register model {model_path}: {e}")
        return {"status": "error", "model_path": str(model_path), "error": str(e)}


def get_model(
    model_id: str, registry_path: str = "output/model_registry/model_registry.json"
) -> Dict[str, Any]:
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

        registry_file = Path(registry_path)

        # Initialize registry
        registry = ModelRegistry(registry_file)

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
                "versions": list(model.versions.keys()),
                "tags": model.tags,
                "metadata": model.metadata,
            }

        return {"status": "error", "model_id": model_id, "error": "Model not found"}

    except Exception as e:
        logger.error(f"Failed to get model {model_id}: {e}")
        return {"status": "error", "model_id": model_id, "error": str(e)}


def search_models(
    query: str, registry_path: str = "output/model_registry/model_registry.json"
) -> List[Dict[str, Any]]:
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

        registry_file = Path(registry_path)

        # Initialize registry
        registry = ModelRegistry(registry_file)

        # Search models
        models = registry.search_models(query)

        # Convert to dictionaries
        return [
            {
                "model_id": model.model_id,
                "model_name": model.name,
                "description": model.description,
                "current_version": model.current_version,
                "tags": model.tags,
            }
            for model in models
        ]

    except Exception as e:
        logger.error(f"Failed to search models: {e}")
        return [{"status": "error", "error": str(e)}]


def list_models(
    registry_path: str = "output/model_registry/model_registry.json",
) -> List[Dict[str, Any]]:
    """
    List all models in the registry.

    Args:
        registry_path: Path to the registry file

    Returns:
        List of all model entries
    """
    try:
        from .registry import ModelRegistry

        registry_file = Path(registry_path)

        # Initialize registry
        registry = ModelRegistry(registry_file)

        # List models
        models = registry.list_models()

        # Convert to dictionaries
        return [
            {
                "model_id": model.model_id,
                "model_name": model.name,
                "description": model.description,
                "current_version": model.current_version,
                "tags": model.tags,
            }
            for model in models
        ]

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return [{"status": "error", "error": str(e)}]
