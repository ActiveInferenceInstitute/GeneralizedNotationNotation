#!/usr/bin/env python3
"""
GNN Pipeline REST API Module.

Provides a FastAPI-based REST interface for triggering pipeline steps,
polling job status, and invoking individual tools without running the full pipeline.

Optional module — requires the [api] extra:
    uv sync --extra api

Usage:
    python -m gnn.api.server
    # or via pipeline:
    python src/gnn/main.py --only-steps 21  # MCP step also registers API tools
"""

from pathlib import Path
from typing import Any

from gnn import __version__

MODULE_NAME = "api"
MODULE_VERSION = "3.3.0"
MODULE_DESCRIPTION = "FastAPI-based REST interface for the GNN processing pipeline"

# API is optional — check for fastapi at import time
try:
    import fastapi  # noqa: F401

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

FEATURES: dict[str, Any] = {
    "rest_api": FASTAPI_AVAILABLE,
    "job_management": FASTAPI_AVAILABLE,
    "async_execution": FASTAPI_AVAILABLE,
    "mcp_tool_registration": True,
}

__all__: list[str] = ["MODULE_NAME", "MODULE_VERSION", "FASTAPI_AVAILABLE", "FEATURES"]


def get_module_info() -> dict[str, Any]:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "api",
        "version": __version__,
        "description": "REST API (FastAPI) for pipeline-as-a-service",
        "features": FEATURES,
    }
