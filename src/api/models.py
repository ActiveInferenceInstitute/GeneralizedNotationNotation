#!/usr/bin/env python3
"""
Pydantic models for GNN API request/response validation.

These models define the API contract — request shapes and response schemas.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum

try:
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "pydantic is required for the GNN API module. "
        "Install with: uv sync --extra api"
    )


class JobStatus(str, Enum):
    """Pipeline job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessRequest(BaseModel):
    """Request to process GNN files through pipeline steps."""

    target_dir: str = Field(
        default="input/gnn_files",
        description="Directory containing GNN files to process"
    )
    steps: Optional[List[int]] = Field(
        default=None,
        description="Specific pipeline steps to run (e.g., [3,5,8]). None = all steps."
    )
    skip_steps: Optional[List[int]] = Field(
        default=None,
        description="Pipeline steps to skip"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging output"
    )
    strict: bool = Field(
        default=False,
        description="Treat warnings as errors"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "target_dir": "input/gnn_files",
            "steps": [3, 5, 6, 8],
            "verbose": True
        }
    }}


class ToolRequest(BaseModel):
    """Request to invoke a single pipeline step/tool."""

    target_dir: str = Field(
        default="input/gnn_files",
        description="Directory containing GNN files"
    )
    verbose: bool = Field(default=False)
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Step-specific parameters"
    )


class JobResponse(BaseModel):
    """Response containing job ID and initial status."""

    job_id: str = Field(description="Unique job identifier")
    status: JobStatus = Field(description="Current job status")
    created_at: datetime = Field(description="Job creation timestamp")
    steps_requested: Optional[List[int]] = Field(default=None)
    message: str = Field(default="Job queued for execution")


class JobStatusResponse(BaseModel):
    """Detailed job status response."""

    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_step: Optional[int] = Field(
        default=None,
        description="Currently executing step number"
    )
    steps_completed: List[int] = Field(default_factory=list)
    steps_failed: List[int] = Field(default_factory=list)
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    output_dir: Optional[str] = None


class ToolInfo(BaseModel):
    """Information about an available pipeline tool/step."""

    step_number: int
    name: str
    description: str
    script: str


class ToolsResponse(BaseModel):
    """List of available pipeline tools."""

    tools: List[ToolInfo]
    total: int


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = "healthy"
    version: str
    pipeline_steps: int
    active_jobs: int
    timestamp: datetime = Field(default_factory=datetime.now)
