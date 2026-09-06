#!/usr/bin/env python3
"""
MCP API module for GNN pipeline job management.

This module integrates the GNN pipeline API capabilities with the MCP
(Multi-Agent Communication Protocol) framework, allowing AI assistants
to interact with the GNN pipeline for submitting, monitoring, and
managing processing jobs.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .path_utils import PathValidationError, resolve_repo_path
from .processor import PIPELINE_STEPS, cancel_job, create_job, get_job, list_jobs

logger = logging.getLogger(__name__)

# Basic module metadata
__version__ = "1.0.0"
__description__ = "API module MCP integration for GNN pipeline job management."
__dependencies__: list[Any] = []


def gnn_submit_job_mcp(
    target_dir: str,
    steps: List[int] | None = None,
    skip_steps: List[int] | None = None,
    verbose: bool = False,
    strict: bool = False,
) -> Dict[str, Any]:
    """Submit a GNN pipeline processing job via MCP."""
    try:
        target_path = resolve_repo_path(
            target_dir,
            purpose="Target directory",
            must_exist=True,
        )

        job_id = create_job(
            target_dir=str(target_path),
            steps=steps,
            skip_steps=skip_steps,
            verbose=verbose,
            strict=strict,
        )

        # We need to trigger async execution somehow, but we are in a sync wrapper.
        # Since we use an external process invocation in create_job_async,
        # we can use subprocess directly here to initiate it optionally, or
        # rely on the API server running. We will return the job_id and instructions.

        # Alternatively, we just return the job_id. The user can start the server.
        return {
            "status": "success",
            "job_id": job_id,
            "message": "Job created. Note: async execution requires the API server to be running.",
        }
    except (PathValidationError, ValueError) as e:
        logger.warning("Rejected MCP job submission: %s", e)
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error("Failed to submit job via MCP: %s", e, exc_info=True)
        return {"status": "error", "message": str(e)}


def gnn_get_job_status_mcp(job_id: str) -> Dict[str, Any]:
    """Retrieve the status of a GNN pipeline job via MCP."""
    try:
        job = get_job(job_id)
        if job:
            return {"status": "success", "job": job}
        return {"status": "error", "message": f"Job {job_id} not found."}
    except Exception as e:
        logger.error(f"Failed to get job status via MCP: {e}")
        return {"status": "error", "message": str(e)}


def gnn_cancel_job_mcp(job_id: str) -> Dict[str, Any]:
    """Cancel a GNN pipeline job via MCP."""
    try:
        success = cancel_job(job_id)
        if success:
            return {
                "status": "success",
                "message": f"Job {job_id} cancelled successfully.",
            }
        return {
            "status": "error",
            "message": f"Failed to cancel job {job_id}. It may not exist or is already terminal.",
        }
    except Exception as e:
        logger.error(f"Failed to cancel job via MCP: {e}")
        return {"status": "error", "message": str(e)}


def gnn_list_jobs_mcp(limit: int = 50) -> Dict[str, Any]:
    """List recent GNN pipeline jobs via MCP."""
    try:
        jobs = list_jobs(limit=limit)
        return {"status": "success", "jobs": jobs, "total": len(jobs)}
    except Exception as e:
        logger.error(f"Failed to list jobs via MCP: {e}")
        return {"status": "error", "message": str(e)}


def gnn_get_pipeline_tools_mcp() -> Dict[str, Any]:
    """List available pipeline steps via MCP."""
    try:
        tools = [
            {"step_number": step, "name": name, "description": desc}
            for step, (name, desc) in PIPELINE_STEPS.items()
        ]
        return {"status": "success", "tools": tools}
    except Exception as e:
        logger.error(f"Failed to list pipeline tools via MCP: {e}")
        return {"status": "error", "message": str(e)}


_MCP_TOOL_DEFINITIONS: tuple[Dict[str, Any], ...] = (
    {
        "name": "gnn_submit_job",
        "handler": gnn_submit_job_mcp,
        "input_schema": {
            "type": "object",
            "properties": {
                "target_dir": {
                    "type": "string",
                    "description": "Target directory containing GNN files",
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Specific steps to run (optional)",
                },
                "skip_steps": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Steps to skip (optional)",
                },
                "verbose": {"type": "boolean", "default": False},
                "strict": {"type": "boolean", "default": False},
            },
            "required": ["target_dir"],
        },
        "description": "Submit a GNN pipeline processing job.",
    },
    {
        "name": "gnn_get_job_status",
        "handler": gnn_get_job_status_mcp,
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The ID of the job to query",
                }
            },
            "required": ["job_id"],
        },
        "description": "Retrieve the status of a GNN pipeline job.",
    },
    {
        "name": "gnn_cancel_job",
        "handler": gnn_cancel_job_mcp,
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The ID of the job to cancel",
                }
            },
            "required": ["job_id"],
        },
        "description": "Cancel a GNN pipeline job.",
    },
    {
        "name": "gnn_list_jobs",
        "handler": gnn_list_jobs_mcp,
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 50,
                    "description": "Maximum number of jobs to return",
                }
            },
        },
        "description": "List recent GNN pipeline jobs.",
    },
    {
        "name": "gnn_get_pipeline_tools",
        "handler": gnn_get_pipeline_tools_mcp,
        "input_schema": {"type": "object", "properties": {}},
        "description": "List available pipeline steps.",
    },
)


def register_tools(mcp_instance: Any) -> None:
    """Register the same API tools advertised by the serialized manifest."""
    for tool in _MCP_TOOL_DEFINITIONS:
        mcp_instance.register_tool(
            tool["name"],
            tool["handler"],
            tool["input_schema"],
            tool["description"],
            module=__package__,
            category="api",
        )

    logger.info("api module MCP tools registered (%d tools).", len(MCP_TOOLS))


MCP_TOOLS: list[Dict[str, Any]] = [
    {
        "name": tool["name"],
        "description": tool["description"],
        "inputSchema": tool["input_schema"],
    }
    for tool in _MCP_TOOL_DEFINITIONS
]


def register_mcp_tools() -> Dict[str, Any]:
    """Return MCP tool registration manifest."""
    return {
        "module": "api",
        "tools": MCP_TOOLS,
        "endpoint": "http://localhost:8000/api/v1",
        "version": "1.0.0",
    }


def save_mcp_manifest(output_dir: Path) -> bool:
    """Save MCP tool manifest to output directory."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = register_mcp_tools()
        manifest_path = output_dir / "api_mcp_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"MCP manifest saved to {manifest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save MCP manifest: {e}")
        return False
