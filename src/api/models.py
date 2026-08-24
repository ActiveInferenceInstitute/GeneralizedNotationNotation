#!/usr/bin/env python3
"""
Pydantic models for GNN API request/response validation.

These models define the API contract — request shapes and response schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
except ImportError as e:
    raise ImportError(
        "pydantic is required for the GNN API module. Install with: uv sync --extra api"
    ) from e


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
        min_length=1,
        description="Directory containing GNN files to process",
    )
    output_dir: str = Field(
        default="output",
        min_length=1,
        description="Directory where pipeline outputs should be written",
    )
    steps: Optional[List[int]] = Field(
        default=None,
        description="Specific pipeline steps to run (e.g., [3,5,8]). None = all steps.",
    )
    skip_steps: Optional[List[int]] = Field(
        default=None, description="Pipeline steps to skip"
    )
    verbose: bool = Field(default=False, description="Enable verbose logging output")
    strict: bool = Field(default=False, description="Treat warnings as errors")

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "target_dir": "input/gnn_files",
                "output_dir": "output",
                "steps": [3, 5, 6, 8],
                "verbose": True,
            }
        },
    )

    @field_validator("steps", "skip_steps")
    @classmethod
    def validate_step_numbers(cls, values: Optional[List[int]]) -> Optional[List[int]]:
        """Require unique pipeline step numbers in the supported 0-24 range."""
        if values is None:
            return None
        invalid = sorted({step for step in values if step < 0 or step > 24})
        if invalid:
            raise ValueError(f"Pipeline steps must be between 0 and 24: {invalid}")
        if len(values) != len(set(values)):
            raise ValueError("Pipeline step lists must not contain duplicates")
        return values

    @model_validator(mode="after")
    def validate_step_selection(self) -> "ProcessRequest":
        """Reject contradictory include and skip selections."""
        overlap = set(self.steps or ()) & set(self.skip_steps or ())
        if overlap:
            raise ValueError(
                f"Pipeline steps cannot be both requested and skipped: {sorted(overlap)}"
            )
        return self


class ToolRequest(BaseModel):
    """Request to invoke a single pipeline step/tool."""

    target_dir: str = Field(
        default="input/gnn_files",
        min_length=1,
        description="Directory containing GNN files",
    )
    output_dir: str = Field(
        default="output",
        min_length=1,
        description="Directory where pipeline outputs should be written",
    )
    verbose: bool = Field(default=False)
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Reserved for future step-specific parameters; currently must be empty",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    @field_validator("kwargs")
    @classmethod
    def reject_unsupported_kwargs(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Reject parameters the subprocess dispatcher cannot honor."""
        if value:
            raise ValueError("Step-specific kwargs are not supported by this endpoint")
        return value


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
        default=None, description="Currently executing step number"
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
