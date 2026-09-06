#!/usr/bin/env python3
"""
Pydantic models for GNN API request/response validation.

These models define the API contract — request shapes and response schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional

try:
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        ValidationInfo,
        field_validator,
        model_validator,
    )
except ImportError as e:
    raise ImportError(
        "pydantic is required for the GNN API module. Install with: uv sync --extra api"
    ) from e


def validate_step_numbers(
    values: Optional[List[int]], *, field_name: str
) -> Optional[List[int]]:
    """Validate an optional list of unique pipeline step numbers (0-24).

    Single source of truth shared by ``ProcessRequest``, ``RunRequest``, and
    the job manager. Rejects non-integers (including ``bool``), out-of-range
    numbers, and duplicate selections with an explicit ``ValueError``.
    """
    if values is None:
        return None
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list of integers")
    invalid = sorted(
        [
            step
            for step in values
            if isinstance(step, bool)
            or not isinstance(step, int)
            or not 0 <= step <= 24
        ],
        key=str,
    )
    if invalid:
        raise ValueError(
            f"{field_name} must contain integers between 0 and 24: {invalid}"
        )
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicate step numbers")
    return list(values)


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

    @field_validator("steps", "skip_steps", mode="before")
    @classmethod
    def check_step_numbers(
        cls, values: Optional[List[int]], info: ValidationInfo
    ) -> Optional[List[int]]:
        """Require unique pipeline step numbers in the supported 0-24 range."""
        return validate_step_numbers(values, field_name=str(info.field_name))

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


class RunRequest(BaseModel):
    """Pipeline run request (``api.app`` run surface)."""

    target_dir: str = Field(default="input/gnn_files", min_length=1)
    output_dir: str = Field(default="output", min_length=1)
    skip_steps: List[int] = Field(default_factory=list)
    skip_llm: bool = False
    strict: bool = Field(default=False, description="Treat warnings as errors")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Reserved for future run configuration; currently must be empty",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    @field_validator("skip_steps", mode="before")
    @classmethod
    def validate_skip_steps(cls, values: List[int]) -> List[int]:
        """Require unique pipeline step numbers in the supported range."""
        checked = validate_step_numbers(values, field_name="skip_steps")
        return checked if checked is not None else []

    @field_validator("config")
    @classmethod
    def reject_unsupported_config(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Reject configuration that the background runner cannot honor."""
        if value:
            raise ValueError("Custom run config is not supported by this endpoint")
        return value


class RunStatus(BaseModel):
    """Pipeline run status response (``api.app`` run surface)."""

    run_hash: str
    status: str  # queued, running, completed, failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    current_step: Optional[str] = None
    steps_completed: int = 0
    total_steps: int = 25
    errors: List[str] = Field(default_factory=list)


class RunHealthResponse(BaseModel):
    """Health response for the ``api.app`` run surface (renderer availability)."""

    status: str = "healthy"
    version: str = "2.0.0"
    pipeline_steps: int = 25
    renderers: Dict[str, bool] = Field(default_factory=dict)
    uptime_seconds: float = 0.0
