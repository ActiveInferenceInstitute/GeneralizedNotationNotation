#!/usr/bin/env python3
"""Programmatic pipeline execution adapters.

The main orchestration engine lives in :mod:`main`. This module provides a
small importable surface that delegates actual work to the numbered step
scripts through ``main.py``.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from ._version import __version__ as _PACKAGE_VERSION
from .config import DEFAULT_OUTPUT_DIR, DEFAULT_TARGET_DIR, STEP_METADATA

_STEP_SCRIPT_BY_NUM = {int(key.split("_", 1)[0]): f"{key}.py" for key in STEP_METADATA}
_STEP_NUM_BY_ALIAS: dict[str, int] = {}
for _num, _script in _STEP_SCRIPT_BY_NUM.items():
    _stem = Path(_script).stem
    _suffix = _stem.split("_", 1)[1] if "_" in _stem else _stem
    _STEP_NUM_BY_ALIAS[str(_num)] = _num
    _STEP_NUM_BY_ALIAS[_stem] = _num
    _STEP_NUM_BY_ALIAS[_script] = _num
    _STEP_NUM_BY_ALIAS[_suffix] = _num
_STEP_NUM_BY_ALIAS["advanced_visualization"] = 9

# Step outcomes that count as success across the pipeline summary contract.
_SUCCESS_STATUSES = frozenset({"SUCCESS", "SUCCESS_WITH_WARNINGS", "SKIPPED"})


def _ensure_src_on_path() -> None:
    """Handle ensure src on path for internal callers."""
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _main_module() -> Any:
    """Handle main module for internal callers."""
    _ensure_src_on_path()
    import main

    return main


def resolve_step_numbers(
    steps: Any,
    pipeline_data: dict | None = None,
) -> list[int]:
    """Normalize user-provided step identifiers to registered step numbers.

    Accepts: ``None``/""/"all"/"pipeline" (→ every registered step), a single
    step number or name ("11", "11_render", "11_render.py"), comma-separated
    lists ("3,5" — mirrors the ``--only-steps`` CLI form), or any iterable
    mixing those forms. ``pipeline_data`` may carry a fallback ``steps`` /
    ``only_steps`` entry when ``steps`` is ``None``. Unknown or empty tokens
    are dropped; duplicates collapse; output is sorted and de-duplicated.

    Raises no exceptions for unrecognized names — callers decide whether an
    empty result is an error (see :func:`run_pipeline`).
    """
    return _coerce_steps(steps, pipeline_data)


def _coerce_steps(steps: Any, pipeline_data: dict | None = None) -> list[int]:
    """Backward-compatible alias for :func:`resolve_step_numbers`."""
    if steps is None and pipeline_data:
        steps = pipeline_data.get("steps") or pipeline_data.get("only_steps")

    if steps in (None, "", "pipeline", "all"):
        return list(_STEP_SCRIPT_BY_NUM)
    if isinstance(steps, (str, int)):
        steps = [steps]

    out: list[int] = []
    for step in steps:
        tokens = str(step).split(",") if isinstance(step, str) else [str(step)]
        for token in tokens:
            key = token.strip()
            if not key:
                continue
            key = key[:-3] if key.endswith(".py") else key
            num = _STEP_NUM_BY_ALIAS.get(key)
            if num is None and key.isdigit():
                num = int(key)
            if num is not None and num in _STEP_SCRIPT_BY_NUM:
                out.append(num)
    return sorted(dict.fromkeys(out))


def _script_for_step(step_name: str) -> str | None:
    """Handle script for step for internal callers."""
    steps = _coerce_steps([step_name])
    if not steps:
        return None
    return _STEP_SCRIPT_BY_NUM.get(steps[0])


def _path_from_sources(
    key: str,
    *,
    pipeline_data: dict | None,
    step_config: dict | None = None,
    fallback: str,
) -> Path:
    """Handle path from sources for internal callers."""
    for source in (step_config or {}, pipeline_data or {}):
        value = source.get(key)
        if value is not None:
            return Path(value)
    if key == "target_dir" and pipeline_data:
        for alias in ("input_dir", "temp_dir"):
            value = pipeline_data.get(alias)
            if value is not None:
                return Path(value)
    return Path(fallback)


@dataclass
class StepExecutionResult:
    """Result of a pipeline step execution."""

    step_name: str
    success: bool
    duration: float
    output: Optional[str] = None
    error: Optional[str] = None
    warnings: Optional[List[str]] = None
    remediation: Optional[str] = None

    def __post_init__(self) -> Any:
        """Normalize fields after dataclass initialization."""
        if self.warnings is None:
            self.warnings = []


def run_pipeline(
    pipeline_data: dict | None = None,
    *,
    target_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    steps: List[str] | str | None = "all",
    verbose: bool = False,
) -> dict:
    """Execute pipeline steps through ``main.py`` and return a compact summary."""
    start = datetime.now()
    pipeline_data = pipeline_data or {}
    resolved_target = (
        Path(target_dir)
        if target_dir is not None
        else _path_from_sources(
            "target_dir", pipeline_data=pipeline_data, fallback=DEFAULT_TARGET_DIR
        )
    )
    resolved_output = (
        Path(output_dir)
        if output_dir is not None
        else _path_from_sources(
            "output_dir", pipeline_data=pipeline_data, fallback=DEFAULT_OUTPUT_DIR
        )
    )
    step_numbers = _coerce_steps(steps, pipeline_data)

    results: dict[str, Any] = {
        "success": False,
        "steps_executed": [],
        "errors": [],
        "warnings": [],
        "target_dir": str(resolved_target),
        "output_dir": str(resolved_output),
        "exit_code": None,
    }

    try:
        resolved_output.mkdir(parents=True, exist_ok=True)
        if not step_numbers:
            results["errors"].append(f"No valid pipeline steps requested: {steps!r}")
            return results

        main = _main_module()
        from utils.argument_utils import PipelineArguments

        args = PipelineArguments(
            target_dir=resolved_target,
            output_dir=resolved_output,
            only_steps=",".join(str(step) for step in step_numbers),
            verbose=verbose,
        )
        config_override: dict[str, Any] = {
            "pipeline": {"only_steps": args.only_steps, "skip_steps": []},
            "testing_matrix": {"enabled": False},
        }
        exit_code = int(main.main(override_args=args, override_config=config_override))
        results["exit_code"] = exit_code
        results["success"] = exit_code == 0
        summary_file = (
            resolved_output / "00_pipeline_summary" / "pipeline_execution_summary.json"
        )
        if summary_file.exists():
            import json

            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            results["summary_file"] = str(summary_file)
            results["overall_status"] = summary.get("overall_status")
            for step in summary.get("steps", []):
                results["steps_executed"].append(
                    {
                        "step_name": step.get("script_name"),
                        "success": step.get("status") in _SUCCESS_STATUSES,
                        "duration": step.get("duration_seconds", 0.0),
                        "output": step.get("stdout", ""),
                    }
                )
        else:
            for step in step_numbers:
                results["steps_executed"].append(
                    {
                        "step_name": _STEP_SCRIPT_BY_NUM[step],
                        "success": exit_code == 0,
                        "duration": 0.0,
                        "output": "",
                    }
                )

    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Pipeline execution failed: {e}")

    results["duration"] = (datetime.now() - start).total_seconds()
    return results


def get_pipeline_status() -> dict:
    """Get the current pipeline status."""
    return {
        "status": "ready",
        "timestamp": datetime.now().isoformat(),
        "steps_available": len(_STEP_SCRIPT_BY_NUM),
        "steps_completed": 0,
    }


def validate_pipeline_config(config: dict) -> bool:
    """Validate pipeline configuration."""
    try:
        required_keys: list[str] = ["steps", "output_dir"]
        return all(key in config for key in required_keys)
    except (TypeError, KeyError):
        return False


def get_pipeline_info() -> dict:
    """Get pipeline information."""
    return {
        "name": "GNN Pipeline",
        "version": _PACKAGE_VERSION,
        "description": "GeneralizedNotationNotation processing pipeline",
        "steps": sorted(_STEP_SCRIPT_BY_NUM),
    }


def create_pipeline_config() -> dict:
    """Create a default pipeline configuration."""
    return {
        "project_name": "GeneralizedNotationNotation",
        "version": _PACKAGE_VERSION,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "steps": {},
    }


def execute_pipeline_step(
    step_name: str, step_config: dict, pipeline_data: dict
) -> StepExecutionResult:
    """Execute a single numbered pipeline step via ``main.execute_pipeline_step``."""
    script_name = _script_for_step(step_name)
    if step_config.get("script_path"):
        candidate = Path(step_config["script_path"])
        if not candidate.exists():
            return StepExecutionResult(
                step_name=step_name,
                success=False,
                duration=0.0,
                error=f"Step script not found: {candidate}",
            )
        script_name = candidate.name
    if not script_name:
        return StepExecutionResult(
            step_name=step_name,
            success=False,
            duration=0.0,
            error=f"Unknown pipeline step: {step_name}",
        )

    start = datetime.now()
    try:
        main = _main_module()
        from utils.argument_utils import PipelineArguments

        args = PipelineArguments(
            target_dir=_path_from_sources(
                "target_dir",
                pipeline_data=pipeline_data,
                step_config=step_config,
                fallback=DEFAULT_TARGET_DIR,
            ),
            output_dir=_path_from_sources(
                "output_dir",
                pipeline_data=pipeline_data,
                step_config=step_config,
                fallback=DEFAULT_OUTPUT_DIR,
            ),
            verbose=bool(
                step_config.get("verbose") or pipeline_data.get("verbose", False)
            ),
        )
        logger = logging.getLogger(f"pipeline.{Path(script_name).stem}")
        raw = main.execute_pipeline_step(script_name, args, logger)
        duration = (datetime.now() - start).total_seconds()
        success = raw.get("status") in _SUCCESS_STATUSES
        return StepExecutionResult(
            step_name=script_name,
            success=success,
            duration=duration,
            output=raw.get("stdout", ""),
            error=raw.get("stderr") if not success else None,
            warnings=raw.get("dependency_warnings", []),
        )
    except Exception as e:
        return StepExecutionResult(
            step_name=script_name,
            success=False,
            duration=(datetime.now() - start).total_seconds(),
            error=str(e),
        )


def execute_pipeline_steps(
    steps: List[str], pipeline_data: dict
) -> List[StepExecutionResult]:
    """Execute multiple pipeline steps."""
    results: list[StepExecutionResult] = []
    for step_name in steps:
        step_config: dict[str, Any] = {}
        result = execute_pipeline_step(step_name, step_config, pipeline_data)
        results.append(result)
    return results
