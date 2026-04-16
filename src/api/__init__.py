#!/usr/bin/env python3
"""
GNN Pipeline REST API Module.

Provides a FastAPI-based REST interface for triggering pipeline steps,
polling job status, and invoking individual tools without running the full pipeline.

Optional module — requires the [api] extra:
    uv sync --extra api

Usage:
    python -m api.server
    # or via pipeline:
    python src/main.py --only-steps 21  # MCP step also registers API tools
"""
__version__ = "1.6.0"


from pathlib import Path

MODULE_NAME = "api"
MODULE_VERSION = "1.0.0"
MODULE_DESCRIPTION = "FastAPI-based REST interface for the GNN processing pipeline"

# API is optional — check for fastapi at import time
try:
    import fastapi  # noqa: F401
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

FEATURES = {
    "rest_api": FASTAPI_AVAILABLE,
    "job_management": FASTAPI_AVAILABLE,
    "async_execution": FASTAPI_AVAILABLE,
    "mcp_tool_registration": True,
}

__all__ = ["MODULE_NAME", "MODULE_VERSION", "FASTAPI_AVAILABLE", "FEATURES"]


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "api",
        "version": __version__,
        "description": "REST API (FastAPI) for pipeline-as-a-service",
        "features": FEATURES,
    }
