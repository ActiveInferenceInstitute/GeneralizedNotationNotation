#!/usr/bin/env python3
"""Consolidated in-process step execution (S2-11 / V4-STAGE, first slice).

The 25-step numbered-script contract stays the pipeline's CLI surface:
``main.py`` still launches one subprocess per numbered script by default.
This module adds the opt-in counterpart selected by ``--consolidated-steps``:
steps whitelisted in ``step_registry.CONSOLIDATED_IN_PROCESS_STEMS`` run
inside the main process. Resolution stays registry-driven — the executor
imports ``gnn.<stem>`` and calls ``StepInfo.module_function``, exactly the
surface asserted by ``tests/pipeline/test_step_registry.py`` — so the
numbered scripts remain the single source of the step contract.

Each call mirrors the argument contract of
``gnn.utils.pipeline_template.create_standardized_pipeline_script``: the
per-step CLI surface from ``StepConfiguration`` is forwarded as keyword
values, ``target_dir``/``output_dir`` resolve to the standard numbered
``<stem>_output`` directory, and the return value coerces through the shared
``coerce_step_exit_code`` contract. The returned receipt dict matches
``execute_pipeline_step``'s ``step_result`` shape plus an ``execution_mode``
field, so ``main._record_step_result`` treats both modes identically.

First-slice limits (docs/decisions/0001-consolidated-pipeline-execution.md):
no per-step timeout, no stdout/stderr capture, and testing-matrix folder
dispatch stays on the subprocess path.
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

from gnn.pipeline.config import get_output_dir_for_script
from gnn.pipeline.step_registry import (
    CONSOLIDATED_IN_PROCESS_STEMS,
    canonical_step_stem,
    step_for_stem,
)
from gnn.utils.error_handling import coerce_step_exit_code, status_from_exit_code
from gnn.utils.pipeline_validator import validate_step_prerequisites
from gnn.utils.resource_manager import get_current_memory_usage
from gnn.utils.step_config import StepConfiguration
from gnn.utils.structured_logging import log_step_warning

__all__ = [
    "UnsupportedStepError",
    "can_execute_in_process",
    "execute_step_in_process",
    "resolve_step_function",
]

# Handled positionally exactly like the numbered-script wrapper handles them;
# never forwarded a second time through **kwargs.
_ARGS_EXCLUDED_FROM_KWARGS = frozenset(
    {"target_dir", "output_dir", "recursive", "verbose"}
)


class UnsupportedStepError(ValueError):
    """Raised when a step is refused by the consolidated executor."""


def canonical_stem_for_script(script_name: str) -> str:
    """Normalize ``11_render.py`` / ``15_llm`` style names to a canonical stem."""
    stem = script_name[:-3] if script_name.endswith(".py") else script_name
    return canonical_step_stem(stem)


def _global_steps_skip_note(
    stem: str, pipeline_config: Optional[Dict[str, Any]]
) -> Optional[str]:
    """Mirror execute_pipeline_step's global_steps skip for a whitelisted step."""
    if not pipeline_config:
        return None
    matrix = pipeline_config.get("testing_matrix", {})
    if not isinstance(matrix, dict):
        return None
    global_steps = matrix.get("global_steps", {})
    if (
        isinstance(global_steps, dict)
        and stem in global_steps
        and not global_steps[stem]
    ):
        return f"Skipped by global_steps config (testing_matrix.global_steps.{stem}: false)"
    return None


def _matrix_dispatch_active(
    stem: str, args: Any, pipeline_config: Optional[Dict[str, Any]]
) -> bool:
    """Mirror execute_pipeline_step's per-folder dispatch condition.

    The subprocess path forks one run per target subfolder only when the
    testing matrix is enabled and at least one subfolder allows this step;
    flat target dirs run standard mode and stay consolidated-eligible.
    """
    if not pipeline_config:
        return False
    matrix = pipeline_config.get("testing_matrix", {})
    if not isinstance(matrix, dict) or not matrix.get("enabled", False):
        return False
    try:
        step_num = int(stem.split("_")[0])
    except ValueError:
        return False
    if step_num < 3:  # execute_pipeline_step computes folders only for >= 3
        return False
    target_dir = Path(args.target_dir)
    if not (target_dir.exists() and target_dir.is_dir()):
        return False
    folders_config = matrix.get("folders", {})
    default_steps = matrix.get("default_steps", [])
    for item in target_dir.iterdir():
        if item.is_dir() and item.name != "archived_gnn_files":
            allowed_steps = folders_config.get(item.name, default_steps)
            if step_num in allowed_steps:
                return True
    return False


def can_execute_in_process(
    script_name: str,
    args: Any,
    *,
    pipeline_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True when *script_name* is eligible for consolidated execution.

    Eligible means: whitelisted in ``CONSOLIDATED_IN_PROCESS_STEMS``, not
    folder-dispatched by the testing matrix, and not skipped by
    ``testing_matrix.global_steps``.
    """
    stem = canonical_stem_for_script(script_name)
    if stem not in CONSOLIDATED_IN_PROCESS_STEMS:
        return False
    if _matrix_dispatch_active(stem, args, pipeline_config):
        return False
    return _global_steps_skip_note(stem, pipeline_config) is None


def resolve_step_function(script_name: str) -> Callable[..., Any]:
    """Resolve the registry's ``module_function`` for *script_name*.

    The numbered script module (``gnn.<stem>``) remains the resolution
    surface — registry stem to script module to function name — so the
    executor adds no second step mapping of its own.
    """
    step = step_for_stem(canonical_stem_for_script(script_name))
    if step is None:
        raise UnsupportedStepError(f"Unknown pipeline step: {script_name!r}")
    module = importlib.import_module(f"gnn.{step.script_stem}")
    function = getattr(module, step.module_function, None)
    if not callable(function):
        raise UnsupportedStepError(
            f"{step.script_stem}: registry function "
            f"{step.module_function!r} does not resolve to a callable"
        )
    return cast("Callable[..., Any]", function)


def _in_process_step_kwargs(stem: str, args: Any) -> Dict[str, Any]:
    """Forward exactly the step-configured CLI arguments as keyword values.

    Mirrors ``build_step_command_args``: the per-step argument surface comes
    from ``StepConfiguration`` (required_args + optional_args), values from
    ``PipelineArguments``, and optional ``None`` values are omitted exactly
    as the subprocess command omits them.
    """
    config = StepConfiguration.get_step_config(stem)
    required = frozenset(config.get("required_args", []))
    kwargs: Dict[str, Any] = {}
    for name in list(config.get("required_args", [])) + list(
        config.get("optional_args", [])
    ):
        if name in _ARGS_EXCLUDED_FROM_KWARGS or not hasattr(args, name):
            continue
        value = getattr(args, name)
        if value is None and name not in required:
            continue
        kwargs[name] = value
    return kwargs


def execute_step_in_process(
    script_name: str,
    args: Any,
    logger: logging.Logger,
    *,
    run_id: Optional[str] = None,
    pipeline_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one whitelisted step in-process and return its execution receipt.

    The receipt matches ``execute_pipeline_step``'s ``step_result`` contract
    (same keys, same status mapping) so the shared recording tail in ``main``
    treats both modes identically; ``execution_mode`` records which mode ran.
    Raises :class:`UnsupportedStepError` for steps outside the consolidated
    whitelist or under testing-matrix folder dispatch.
    """
    stem = canonical_stem_for_script(script_name)
    if stem not in CONSOLIDATED_IN_PROCESS_STEMS:
        raise UnsupportedStepError(
            f"Step {script_name!r} is not enabled for consolidated in-process execution"
        )
    if _matrix_dispatch_active(stem, args, pipeline_config):
        raise UnsupportedStepError(
            f"Step {script_name!r} dispatches testing-matrix folders; "
            "the consolidated executor refuses it"
        )

    step = step_for_stem(stem)
    if step is None:  # pragma: no cover - whitelist check above guarantees this
        raise UnsupportedStepError(f"Unknown pipeline step: {script_name!r}")

    skip_note = _global_steps_skip_note(stem, pipeline_config)
    if skip_note is not None:
        return {
            "status": "SKIPPED",
            "stdout": f"{skip_note}\n",
            "stderr": "",
            "memory_usage_mb": 0.0,
            "peak_memory_mb": 0.0,
            "memory_delta_mb": 0.0,
            "exit_code": 0,
            "retry_count": 0,
            "prerequisite_check": True,
            "dependency_warnings": [],
            "execution_mode": "consolidated",
        }

    function = resolve_step_function(script_name)
    step_output_dir = get_output_dir_for_script(stem, Path(args.output_dir))

    start_memory = get_current_memory_usage()
    step_result: Dict[str, Any] = {
        "status": "UNKNOWN",
        "stdout": "",
        "stderr": "",
        "memory_usage_mb": 0.0,
        "peak_memory_mb": 0.0,
        "memory_delta_mb": 0.0,
        "exit_code": -1,
        "retry_count": 0,
        "prerequisite_check": True,
        "dependency_warnings": [],
        "execution_mode": "consolidated",
    }

    # Prerequisite validation mirrors execute_pipeline_step: record the
    # outcome on the receipt, never gate the step on it.
    config_skip_steps: list[Any] = []
    pipeline_section = (pipeline_config or {}).get("pipeline", {})
    if isinstance(pipeline_section, dict):
        config_skip_steps = list(pipeline_section.get("skip_steps", []) or [])
    prereq_result = validate_step_prerequisites(
        script_name, args, logger, skip_steps=config_skip_steps
    )
    step_result["prerequisite_check"] = prereq_result["passed"]
    step_result["dependency_warnings"] = prereq_result.get("warnings", [])
    for prereq_warning in prereq_result.get("warnings", []):
        logger.warning(f"Prerequisite notice for {script_name}: {prereq_warning}")

    if run_id is not None:
        os.environ["GNN_RUN_ID"] = run_id

    call_started = time.time()
    try:
        result = function(
            target_dir=Path(args.target_dir),
            output_dir=step_output_dir,
            logger=logger,
            recursive=args.recursive,
            verbose=args.verbose,
            **_in_process_step_kwargs(stem, args),
        )
    except Exception as error:  # receipt contract owns the failure shape
        logger.error(
            "Consolidated in-process execution of %s failed: %s",
            step.script_name,
            error,
        )
        step_result["stderr"] = f"{type(error).__name__}: {error}\n"
        result = False

    exit_code = coerce_step_exit_code(
        result,
        step_name=stem,
        logger=logger,
        warning_callback=lambda message: log_step_warning(logger, message),
    )
    end_memory = get_current_memory_usage()

    step_result["exit_code"] = exit_code
    step_result["status"] = status_from_exit_code(
        exit_code, step_result["dependency_warnings"]
    )
    step_result["stdout"] = (
        f"{step.script_name}: consolidated in-process execution completed "
        f"in {time.time() - call_started:.2f}s\n"
    )
    step_result["memory_usage_mb"] = end_memory
    step_result["peak_memory_mb"] = max(start_memory, end_memory)
    step_result["memory_delta_mb"] = end_memory - start_memory
    return step_result
