#!/usr/bin/env python3
"""
Pipeline Schemas — Pydantic models for typed step inputs/outputs.

Provides validated, serializable contracts for pipeline data flow.
Uses Pydantic v2 if available, otherwise falls back to dataclasses.
"""

import logging
from datetime import datetime
from typing import List, Optional

try:
    from pipeline.context import StepStatus
except ImportError:
    from typing import Literal, TypeAlias
    StepStatus: TypeAlias = Literal["PENDING", "SUCCESS", "FAILED", "WARNING", "SKIPPED", "UNKNOWN"]

logger = logging.getLogger(__name__)

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logger.debug("pydantic not installed — using dataclass recovery")

if PYDANTIC_AVAILABLE:

    class GNNModelSummary(BaseModel):
        """Summary of a parsed GNN model."""
        name: str = ""
        file_path: str = ""
        section_count: int = 0
        variable_count: int = 0
        connection_count: int = 0

    class GNNParseOutput(BaseModel):
        """Output from Step 3 (GNN parsing)."""
        models: List[GNNModelSummary] = Field(default_factory=list)
        file_count: int = 0
        parse_errors: List[str] = Field(default_factory=list)
        parse_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    class ValidationOutput(BaseModel):
        """Output from Step 6 (validation)."""
        valid_count: int = 0
        error_count: int = 0
        warnings: List[str] = Field(default_factory=list)

    class RenderOutput(BaseModel):
        """Output from Step 11 (rendering)."""
        framework: str = ""
        output_path: str = ""
        success: bool = True
        error: Optional[str] = None

    class ExecutionResult(BaseModel):
        """Result of a single pipeline step."""
        step_name: str
        step_num: int = -1
        status: StepStatus = "PENDING"
        duration: float = 0.0
        artifacts: List[str] = Field(default_factory=list)
        errors: List[str] = Field(default_factory=list)

    class PipelineSummary(BaseModel):
        """Complete pipeline execution summary."""
        timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
        total_duration: float = 0.0
        success: bool = True
        steps: List[ExecutionResult] = Field(default_factory=list)
        errors: List[str] = Field(default_factory=list)
        model_count: int = 0
        artifact_count: int = 0

else:
    # Dataclass recovery when pydantic is not installed
    from dataclasses import dataclass, field

    @dataclass
    class GNNModelSummary:
        name: str = ""
        file_path: str = ""
        section_count: int = 0
        variable_count: int = 0
        connection_count: int = 0

    @dataclass
    class GNNParseOutput:
        models: List = field(default_factory=list)
        file_count: int = 0
        parse_errors: List = field(default_factory=list)
        parse_timestamp: str = ""

    @dataclass
    class ValidationOutput:
        valid_count: int = 0
        error_count: int = 0
        warnings: List = field(default_factory=list)

    @dataclass
    class RenderOutput:
        framework: str = ""
        output_path: str = ""
        success: bool = True
        error: Optional[str] = None

    @dataclass
    class ExecutionResult:
        step_name: str = ""
        step_num: int = -1
        status: StepStatus = "PENDING"
        duration: float = 0.0
        artifacts: List = field(default_factory=list)
        errors: List = field(default_factory=list)

    @dataclass
    class PipelineSummary:
        timestamp: str = ""
        total_duration: float = 0.0
        success: bool = True
        steps: List = field(default_factory=list)
        errors: List = field(default_factory=list)
        model_count: int = 0
        artifact_count: int = 0
