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
from typing import Dict, Any

from src.api.processor import create_job, get_job, cancel_job, list_jobs, PIPELINE_STEPS

logger = logging.getLogger(__name__)

# Basic module metadata
__version__ = "1.0.0"
__description__ = "API module MCP integration for GNN pipeline job management."
__dependencies__ = []


def gnn_submit_job_mcp(target_dir: str, steps: list = None, skip_steps: list = None, verbose: bool = False, strict: bool = False) -> Dict[str, Any]:
    """Submit a GNN pipeline processing job via MCP."""
    try:
        job_id = create_job(target_dir=target_dir, steps=steps, skip_steps=skip_steps, verbose=verbose, strict=strict)

        # We need to trigger async execution somehow, but we are in a sync wrapper.
        # Since we use an external process invocation in create_job_async,
        # we can use subprocess directly here to initiate it optionally, or
        # rely on the API server running. We will return the job_id and instructions.

        # Alternatively, we just return the job_id. The user can start the server.
        return {"status": "success", "job_id": job_id, "message": "Job created. Note: async execution requires the API server to be running."}
    except Exception as e:
        logger.error(f"Failed to submit job via MCP: {e}")
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
            return {"status": "success", "message": f"Job {job_id} cancelled successfully."}
        return {"status": "error", "message": f"Failed to cancel job {job_id}. It may not exist or is already terminal."}
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
        tools = [{"step_number": step, "name": name, "description": desc} for step, (name, desc) in PIPELINE_STEPS.items()]
        return {"status": "success", "tools": tools}
    except Exception as e:
        logger.error(f"Failed to list pipeline tools via MCP: {e}")
        return {"status": "error", "message": str(e)}


def register_tools(mcp_instance) -> None:
    """Register API domain tools with the MCP server."""

    mcp_instance.register_tool(
        "gnn_submit_job",
        gnn_submit_job_mcp,
        {
            "type": "object",
            "properties": {
                "target_dir": {"type": "string", "description": "Target directory containing GNN files"},
                "steps": {"type": "array", "items": {"type": "integer"}, "description": "Specific steps to run (optional)"},
                "skip_steps": {"type": "array", "items": {"type": "integer"}, "description": "Steps to skip (optional)"},
                "verbose": {"type": "boolean", "default": False},
                "strict": {"type": "boolean", "default": False}
            },
            "required": ["target_dir"]
        },
        "Submit a GNN pipeline processing job.",
        module=__package__, category="api",
    )

    mcp_instance.register_tool(
        "gnn_get_job_status",
        gnn_get_job_status_mcp,
        {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "The ID of the job to query"}
            },
            "required": ["job_id"]
        },
        "Retrieve the status of a GNN pipeline job.",
        module=__package__, category="api",
    )

    mcp_instance.register_tool(
        "gnn_cancel_job",
        gnn_cancel_job_mcp,
        {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "The ID of the job to cancel"}
            },
            "required": ["job_id"]
        },
        "Cancel a GNN pipeline job.",
        module=__package__, category="api",
    )

    mcp_instance.register_tool(
        "gnn_list_jobs",
        gnn_list_jobs_mcp,
        {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 50, "description": "Maximum number of jobs to return"}
            }
        },
        "List recent GNN pipeline jobs.",
        module=__package__, category="api",
    )

    mcp_instance.register_tool(
        "gnn_get_pipeline_tools",
        gnn_get_pipeline_tools_mcp,
        {
            "type": "object",
            "properties": {}
        },
        "List available pipeline steps.",
        module=__package__, category="api",
    )

    logger.info("api module MCP tools registered.")


# We don't delete save_mcp_manifest and register_mcp_tools completely in case
# other scripts rely on them.

MCP_TOOLS = [
    {
        "name": "gnn_submit_job",
        "description": "Submit a GNN pipeline processing job. Accepts a target directory and optional step selection. Returns a job ID for status polling.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_dir": {"type": "string", "description": "Directory containing GNN files", "default": "input/gnn_files"},
                "steps": {"type": "array", "items": {"type": "integer"}, "description": "Steps to run (omit for all steps)"},
                "verbose": {"type": "boolean", "default": False}
            },
            "required": []
        }
    },
    {
        "name": "gnn_job_status",
        "description": "Check the status of a submitted GNN pipeline job.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID from gnn_submit_job"}
            },
            "required": ["job_id"]
        }
    },
    {
        "name": "gnn_list_tools",
        "description": "List all available GNN pipeline steps and their descriptions.",
        "inputSchema": {"type": "object", "properties": {}}
    }
]


def register_mcp_tools() -> Dict[str, Any]:
    """Return MCP tool registration manifest."""
    return {
        "module": "api",
        "tools": MCP_TOOLS,
        "endpoint": "http://localhost:8000/api/v1",
        "version": "1.0.0"
    }


def save_mcp_manifest(output_dir: Path) -> bool:
    """Save MCP tool manifest to output directory."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = register_mcp_tools()
        manifest_path = output_dir / "api_mcp_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"MCP manifest saved to {manifest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save MCP manifest: {e}")
        return False
