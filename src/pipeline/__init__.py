#!/usr/bin/env python3
from __future__ import annotations

"""
Pipeline module for centralized configuration and utilities.
"""
from .config import (
    STEP_METADATA,
    PipelineConfig,
    StepConfig,
    get_output_dir_for_script,
    get_pipeline_config,
    set_pipeline_config,
)
from .execution import (
    StepExecutionResult,
    create_pipeline_config,
    execute_pipeline_step,
    execute_pipeline_steps,
    get_pipeline_info,
    get_pipeline_status,
    run_pipeline,
    validate_pipeline_config,
)
from .health_check import EnhancedHealthChecker, run_enhanced_health_check

# Module metadata
__version__ = "1.6.0"
__author__ = "Active Inference Institute"
__description__ = "GNN pipeline orchestration and execution"

# Feature availability flags
FEATURES = {
    "pipeline_orchestration": True,
    "step_execution": True,
    "configuration_management": True,
    "performance_monitoring": True,
    "error_handling": True,
    "dependency_validation": True,
}


class PipelineOrchestrator:
    """Small programmatic adapter over the real ``main.py`` orchestrator."""

    def __init__(
        self,
        target_dir: str = "input/gnn_files",
        output_dir: str = "output",
        steps: list[str] | str | None = "all",
        verbose: bool = False,
    ):
        self.target_dir = target_dir
        self.output_dir = output_dir
        self.steps = steps
        self.verbose = verbose

    def run(self) -> bool:
        """Execute configured steps through ``pipeline.execution.run_pipeline``."""
        result = run_pipeline(
            target_dir=self.target_dir,
            output_dir=self.output_dir,
            steps=self.steps,
            verbose=self.verbose,
        )
        return bool(result.get("success"))

    def get_pipeline_steps(self) -> list[str]:
        cfg = get_pipeline_config()
        steps = cfg.get("steps") if isinstance(cfg, dict) else None
        if steps is None:
            steps = list(STEP_METADATA.keys())
        return list(steps)

    def execute_pipeline(self, pipeline_data: dict | None = None) -> dict:
        data = pipeline_data or {}
        result = run_pipeline(
            pipeline_data=data,
            target_dir=data.get("target_dir")
            or data.get("input_dir")
            or data.get("temp_dir")
            or self.target_dir,
            output_dir=data.get("output_dir") or self.output_dir,
            steps=data.get("steps", self.steps),
            verbose=bool(data.get("verbose", self.verbose)),
        )
        result.setdefault("status", "SUCCESS" if result.get("success") else "FAILED")
        result.setdefault("executed", len(result.get("steps_executed", [])))
        return result


class PipelineStep:
    """Programmatic adapter for executing one registered pipeline step."""

    def __init__(
        self,
        name: str,
        target_dir: str = "input/gnn_files",
        output_dir: str = "output",
        verbose: bool = False,
    ):
        self.name = name
        self.target_dir = target_dir
        self.output_dir = output_dir
        self.verbose = verbose

    def execute(self) -> bool:
        """Execute this step through the same subprocess path used by ``main.py``."""
        result = execute_pipeline_step(
            self.name,
            {"verbose": self.verbose},
            {"target_dir": self.target_dir, "output_dir": self.output_dir},
        )
        return bool(result.success)

    def validate(self) -> bool:
        """Return whether this step name resolves to the registered step metadata."""
        return validate_pipeline_step(self.name) or any(
            self.name == key.split("_", 1)[-1] for key in STEP_METADATA
        )


def get_module_info() -> dict:
    """Return pipeline module metadata for composability and MCP discovery."""
    return {
        "version": __version__,
        "description": "GNN pipeline orchestration and execution",
        "features": FEATURES,
        "pipeline_steps": list(STEP_METADATA.keys()),
        "author": "Active Inference Institute",
    }


def validate_pipeline_step(step_name: str) -> bool:
    """Validate that a pipeline step configuration is well-formed."""
    return step_name in STEP_METADATA


def discover_pipeline_steps() -> list[str]:
    """Discover all available pipeline step modules in the src directory."""
    return list(STEP_METADATA.keys())


# Main API functions
__all__ = [
    # Configuration
    "get_pipeline_config",
    "set_pipeline_config",
    "PipelineConfig",
    "StepConfig",
    "STEP_METADATA",
    "get_output_dir_for_script",
    # Health Check
    "run_enhanced_health_check",
    "EnhancedHealthChecker",
    # Execution
    "run_pipeline",
    "get_pipeline_status",
    "validate_pipeline_config",
    "get_pipeline_info",
    "create_pipeline_config",
    "execute_pipeline_step",
    "execute_pipeline_steps",
    "StepExecutionResult",
    "PipelineOrchestrator",
    "PipelineStep",
    # Metadata
    "FEATURES",
    "__version__",
]
