#!/usr/bin/env python3
"""
Execute Processor module for GNN Processing Pipeline.

This module provides execute processing capabilities for rendered implementations.
"""

import copy
import json
import logging
import os
import subprocess  # nosec B404
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from utils.logging.logging_utils import (
    log_step_error,
    log_step_start,
    log_step_success,
    log_step_warning,
)

from .data_extractors import (
    collect_execution_outputs,
)
from .data_extractors import (
    extract_simulation_data as _extract_simulation_data,
)
from .data_extractors import (
    extract_simulation_data_from_files as _extract_simulation_data_from_files,
)
from .detection import (
    _build_script_execution_context,
    _detect_accelerator_type,
    _resolve_render_output_dir,
    determine_script_framework,
    find_executable_scripts,
    parse_frameworks_parameter,
)
from .julia_env import (
    _build_script_execution_command,
    _julia_project_for_framework,
    check_julia_dependencies,
)
from .metadata import (
    _STATUS_SEVERITY,
    _execution_detail_key,
    _load_render_summary_contract,
    _load_rxinfer_execution_metadata_from_script,
    _load_rxinfer_execution_metadata_sidecar,
    _merge_prior_execution_summary,
    _sha256_file,
    _slim_execution_detail,
    _summarize_collected_outputs,
    generate_execution_report,
)
from .types import (
    _EXECUTABLE_SUFFIXES,
    ExecutionFrameworkName,
    ExecutionOutcome,
    ScriptExecutionContext,
)

logger = logging.getLogger(__name__)

# Phase 2.3: framework-availability helpers moved to utils.framework_availability
# so execute and render stay in sync. The import-check dict and predicate are
# re-exported here via thin aliases to preserve any external callers that
# previously imported them from execute.processor.
from utils.framework_availability import (  # noqa: E402
    FRAMEWORK_IMPORT_CHECK as _FRAMEWORK_IMPORT_CHECK,
)
from utils.framework_availability import (
    is_framework_available as _is_framework_available_by_name,
)


def _is_python_framework_dependency_available(
    framework: str, executor: str, logger: Any
) -> bool:
    """Return True if the framework's required Python module is importable.

    Delegates to ``utils.framework_availability.is_framework_available``, passing
    ``executor`` so the check targets the subprocess-invoked interpreter rather
    than the caller's. Preserves the pre-Phase-2.3 call-site signature.
    """
    return _is_framework_available_by_name(framework, executor=executor, logger=logger)


def _base_execution_envelope(
    *,
    script_path: str,
    script_name: str,
    framework: str,
    model_name: str,
    executor: str,
    status: str,
    skipped: bool,
) -> Dict[str, Any]:
    """Shared 14-key prefix for every per-script execution result envelope.

    Failure and skip builders append their ``error``/``error_type`` (and any
    dispatch-specific extras) on top; the success path mutates the envelope in
    place. Keeping one construction site prevents key-set drift between the
    factories.
    """
    return {
        "script_path": script_path,
        "script_name": script_name,
        "framework": framework,
        "model_name": model_name,
        "executor": executor,
        "success": False,
        "skipped": skipped,
        "status": status,
        "attempts_started": 0,
        "return_code": None,
        "stdout": "",
        "stderr": "",
        "execution_time": 0,
        "timestamp": datetime.now().isoformat(),
    }


def _model_framework_from_path(script_info: Dict[str, Any]) -> Tuple[str, str]:
    """Derive ``(model_name, framework)`` fallbacks from a rendered script path.

    Rendered layouts are ``<...>/<model>/<framework>/<script>``, so the third
    and second path components from the end are authoritative; shorter paths
    fall back to the discovery metadata.
    """
    path_parts = Path(script_info["path"]).parts
    model_name = path_parts[-3] if len(path_parts) >= 3 else "unknown_model"
    framework = path_parts[-2] if len(path_parts) >= 3 else script_info["framework"]
    return model_name, framework


def _make_skipped_result(
    script_info: Dict[str, Any],
    framework: str,
    model_name: str,
    executor: str,
    logger: Any,
) -> Dict[str, Any]:
    """Build an execution result dict for a script skipped due to missing dependency."""
    module_name, install_hint = _FRAMEWORK_IMPORT_CHECK.get(framework, ("", ""))
    reason = (
        f"Dependency not installed: {module_name}"
        if module_name
        else "Dependency not installed"
    )
    if install_hint and not logger.isEnabledFor(logging.DEBUG):
        logger.info(
            f"Skipping {script_info['name']} ({framework}): {module_name} not installed. Install with: {install_hint}"
        )
    envelope = _base_execution_envelope(
        script_path=str(script_info["path"]),
        script_name=script_info["name"],
        framework=framework,
        model_name=model_name,
        executor=executor,
        status="skipped",
        skipped=True,
    )
    envelope["error"] = reason
    envelope["error_type"] = "DependencyNotInstalled"
    envelope["execution_metadata"] = (
        _load_rxinfer_execution_metadata_from_script(Path(script_info["path"]))
        if framework == "rxinfer"
        else {}
    )
    return envelope


def _coerce_execution_workers(value: Any) -> int:
    """Normalize the configured local/distributed worker count."""
    try:
        workers = int(value)
    except (TypeError, ValueError):
        workers = 1
    return max(1, workers)


def _coerce_dispatch_retries(value: Any) -> int:
    """Normalize the distributed task retry limit."""
    try:
        retries = int(value)
    except (TypeError, ValueError):
        retries = 3
    return max(0, retries)


def _execute_script_worker(
    bundle: Tuple[Dict[str, Any], Path, bool, int, int],
) -> Dict[str, Any]:
    """Process-pool entry point for a single rendered script."""
    script_info, results_dir, verbose, timeout, repeats = bundle
    worker_logger = logging.getLogger("execute.worker")
    worker_logger.setLevel(logging.INFO)
    result = execute_single_script(
        script_info,
        results_dir,
        verbose,
        worker_logger,
        timeout,
        execution_benchmark_repeats=repeats,
    )
    result.setdefault("skipped", False)
    return result


def _make_local_worker_pool_failure_result(
    script_info: Dict[str, Any],
    exc: BaseException,
) -> Dict[str, Any]:
    """Return a per-script failure envelope when local process dispatch fails."""
    script_path = Path(script_info["path"])
    model_name, framework = _model_framework_from_path(script_info)
    envelope = _base_execution_envelope(
        script_path=str(script_path),
        script_name=script_info["name"],
        framework=framework,
        model_name=model_name,
        executor=script_info["executor"],
        status="failed",
        skipped=False,
    )
    envelope["error"] = f"Local worker pool failed before script completion: {exc}"
    envelope["error_type"] = "LocalWorkerPoolError"
    envelope["worker_pool_error_type"] = type(exc).__name__
    return envelope


def _make_distributed_dispatch_failure_result(
    script_info: Dict[str, Any],
    exc: BaseException,
    backend: str,
    max_retries: int,
) -> Dict[str, Any]:
    """Return one explicit failure when distributed dispatch cannot complete."""
    script_path = Path(script_info["path"])
    model_name, framework = _model_framework_from_path(script_info)
    envelope = _base_execution_envelope(
        script_path=str(script_path),
        script_name=script_info["name"],
        framework=framework,
        model_name=model_name,
        executor=script_info["executor"],
        status="failed",
        skipped=False,
    )
    envelope["error"] = (
        f"Distributed {backend} dispatch failed before completion: {exc}"
    )
    envelope["error_type"] = "DistributedDispatchError"
    envelope["dispatch_error_type"] = type(exc).__name__
    envelope["dispatch_max_retries"] = max_retries
    return envelope


def _run_scripts_with_local_workers(
    executable_scripts: List[Dict[str, Any]],
    results_dir: Path,
    verbose: bool,
    logger: logging.Logger,
    timeout: int,
    execution_workers: int,
    execution_benchmark_repeats: int,
) -> List[Dict[str, Any]]:
    """Execute rendered scripts locally, using multiple processes when requested."""
    repeats = max(1, int(execution_benchmark_repeats))
    if execution_workers <= 1 or len(executable_scripts) <= 1:
        details: list[Any] = []
        for script_info in executable_scripts:
            exec_result = execute_single_script(
                script_info,
                results_dir,
                verbose,
                logger,
                timeout,
                execution_benchmark_repeats=repeats,
            )
            exec_result.setdefault("skipped", False)
            details.append(exec_result)
        return details

    bounded_workers = min(execution_workers, len(executable_scripts))
    logger.info(
        "Dispatching %s executable scripts with %s local workers",
        len(executable_scripts),
        bounded_workers,
    )
    bundles = [
        (info, results_dir, verbose, timeout, repeats) for info in executable_scripts
    ]
    try:
        with ProcessPoolExecutor(max_workers=bounded_workers) as pool:
            return list(pool.map(_execute_script_worker, bundles))
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Local worker pool failed while executing %d scripts: %s",
            len(executable_scripts),
            exc,
        )
        return [
            _make_local_worker_pool_failure_result(script_info, exc)
            for script_info in executable_scripts
        ]


def _init_execution_summary(
    target_dir: Path,
    output_dir: Path,
    execution_benchmark_repeats: int,
    execution_summary_detail: bool,
) -> Dict[str, Any]:
    """Build the empty Step 12 execution summary envelope."""
    return {
        "timestamp": datetime.now().isoformat(),
        "target_directory": str(target_dir),
        "output_directory": str(output_dir),
        "total_scripts_found": 0,
        "successful_executions": 0,
        "failed_executions": 0,
        "skipped_executions": 0,
        "execution_details": [],
        "framework_status": {},
        "execution_mode": "local",
        "execution_workers": 1,
        "backend": None,
        "execution_benchmark_repeats": execution_benchmark_repeats,
        "execution_summary_detail": execution_summary_detail,
        "dispatch_max_retries": 0,
        "success": False,
        "status": "pending",
        "exit_code": None,
    }


def _update_framework_status(
    execution_results: Dict[str, Any], details: List[Dict[str, Any]]
) -> None:
    """Fold per-script results into aggregate counters and per-framework status."""
    for exec_result in details:
        execution_results["execution_details"].append(exec_result)

        # Update framework status
        framework = exec_result.get("framework", "unknown")
        if framework not in execution_results["framework_status"]:
            execution_results["framework_status"][framework] = {
                "status": "unknown",
                "executions": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0,
            }

        framework_summary = execution_results["framework_status"][framework]
        framework_summary["executions"] += 1

        if exec_result.get("skipped"):
            execution_results["skipped_executions"] = (
                execution_results.get("skipped_executions", 0) + 1
            )
            framework_summary["skipped"] += 1
            if "error" in exec_result:
                framework_summary["error"] = exec_result["error"]
        elif exec_result["success"]:
            execution_results["successful_executions"] += 1
            framework_summary["successful"] += 1
        else:
            execution_results["failed_executions"] += 1
            framework_summary["failed"] += 1
            if "error" in exec_result:
                framework_summary["error"] = exec_result["error"]

    for framework_summary in execution_results["framework_status"].values():
        if framework_summary["failed"]:
            framework_summary["status"] = "failed"
        elif framework_summary["successful"] and framework_summary["skipped"]:
            framework_summary["status"] = "success_with_skips"
        elif framework_summary["successful"]:
            framework_summary["status"] = "success"
        else:
            framework_summary["status"] = "skipped"


def _classify_execute_outcome(
    *,
    total_found: int,
    successful: int,
    failed: int,
    skipped: int,
    render_failures: List[Dict[str, str]],
    missing_render_scripts: List[str],
    missing_render_summary: Optional[str],
    strict_requested_frameworks: bool,
) -> ExecutionOutcome:
    """Classify a finished Step 12 run into its outcome contract.

    Pure function of the run counters: the durable summary and the API result
    are derived from the same classification, so they can never disagree.
    """
    attempted = total_found - skipped
    if missing_render_summary:
        outcome: Union[bool, int] = False
        status = "failed"
        reason = "required_render_summary_missing"
    elif strict_requested_frameworks and render_failures:
        outcome = False
        status = "failed"
        reason = "requested_framework_render_failure"
    elif missing_render_scripts:
        outcome = False
        status = "failed"
        reason = "rendered_script_missing"
    elif total_found == 0:
        outcome = False if strict_requested_frameworks else 2
        status = "failed" if strict_requested_frameworks else "skipped"
        reason = "no_executable_scripts"
    elif strict_requested_frameworks and (failed > 0 or skipped > 0):
        outcome = False
        status = "failed"
        reason = "requested_framework_execution_incomplete"
    elif failed > 0:
        outcome = False
        status = "failed"
        reason = "script_execution_failure"
    elif skipped > 0:
        outcome = True
        status = "success_with_skips"
        reason = "optional_dependencies_unavailable"
    elif render_failures:
        outcome = True
        status = "success_with_render_failures"
        reason = "best_effort_render_subset_executed"
    else:
        outcome = True
        status = "success"
        reason = "all_scripts_succeeded"

    exit_code = 0 if outcome is True else outcome if outcome == 2 else 1
    return ExecutionOutcome(
        outcome=outcome,
        status=status,
        reason=reason,
        exit_code=exit_code,
        attempted=attempted,
    )


def _write_execution_summaries(
    results_dir: Path,
    execution_results: Dict[str, Any],
    execution_summary_detail: bool,
    logger: Any,
) -> None:
    """Persist the slim aggregate (+ optional detail) and regenerate the report.

    The slim aggregate is what lands on disk; full per-script payloads are
    restored on ``execution_results`` afterwards for in-memory consumers.
    """
    summaries_dir = results_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    results_file = summaries_dir / "execution_summary.json"

    # The pipeline invokes this step once per top-level input folder, all
    # writing the same summary file. Carry the earlier folders' script
    # results forward so the durable summary covers every executed script
    # (mirrors the Step 11 render-summary merge).
    _merge_prior_execution_summary(execution_results, results_file, logger)

    full_details_snapshot = copy.deepcopy(execution_results["execution_details"])
    execution_results["execution_details"] = [
        _slim_execution_detail(d) for d in full_details_snapshot
    ]
    execution_results["execution_summary_format"] = "slim_v1"

    with open(results_file, "w") as f:
        json.dump(execution_results, f, indent=2, default=str)

    if execution_summary_detail:
        detail_path = summaries_dir / "execution_summary_detail.json"
        detail_payload = dict(execution_results)
        detail_payload["execution_details"] = full_details_snapshot
        detail_payload["execution_summary_format"] = "detail_v1"
        with open(detail_path, "w") as f:
            json.dump(detail_payload, f, indent=2, default=str)

    # Generate execution report (uses slim execution_details)
    generate_execution_report(execution_results, results_dir, logger)

    # Restore full details in-memory for any downstream callers of this function
    execution_results["execution_details"] = full_details_snapshot


def process_execute(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    frameworks: str = "all",
    **kwargs: Any,
) -> Union[bool, int]:
    """
    Execute rendered implementations from 11_render_output directory.

    This function searches for executable scripts generated by 11_render.py
    and executes them using subprocess, capturing their outputs and results.

    Args:
        target_dir: Directory containing rendered executable scripts (typically 11_render_output)
        output_dir: Directory to save execution results
        verbose: Enable verbose output
        **kwargs: Additional arguments

    Returns:
        True if processing successful, False otherwise
    """
    logger = logging.getLogger("execute")

    try:
        log_step_start(
            logger, "Processing execute - searching for rendered implementations"
        )

        # Phase 1.3: validate frameworks arg before parsing. Rejects non-string
        # input and fully-unknown framework lists early with a clear error.
        try:
            from utils.validation_schemas import validate_frameworks_arg

            frameworks = validate_frameworks_arg(frameworks, context="process_execute")
        except ValueError as _verr:
            log_step_error(logger, f"Invalid frameworks argument: {_verr}")
            return False

        # Parse frameworks parameter
        requested_frameworks = parse_frameworks_parameter(frameworks, logger)
        strict_requested_frameworks = str(frameworks).lower() not in {"all", "lite"}
        logger.info(f"Requested frameworks: {requested_frameworks}")

        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        execution_benchmark_repeats = max(
            1, int(kwargs.get("execution_benchmark_repeats", 1))
        )
        execution_summary_detail = bool(kwargs.get("execution_summary_detail", False))
        require_render_summary = bool(kwargs.get("require_render_summary", False))

        # Initialize execution results
        execution_results: dict[str, Any] = _init_execution_summary(
            target_dir,
            output_dir,
            execution_benchmark_repeats,
            execution_summary_detail,
        )

        # Look for rendered implementations from render output
        render_output_dir = _resolve_render_output_dir(
            target_dir, kwargs, output_dir=results_dir
        )
        if render_output_dir is not None and render_output_dir != target_dir:
            logger.info(f"Found render output directory: {render_output_dir}")

        if verbose:
            logger.info(f"Searching for executable scripts in: {render_output_dir}")

        if not render_output_dir or not render_output_dir.exists():
            log_step_warning(
                logger, f"Render output directory not found: {render_output_dir}"
            )
            execution_results["skipped_reason"] = "no_render_output"
            execution_results["message"] = "No rendered implementations found"
            if require_render_summary:
                execution_results["missing_render_summary"] = (
                    "render output directory unavailable"
                )
        else:
            # Scope the render contract to the current invocation. The
            # pipeline runs Step 12 once per input folder, so a folder-scoped
            # ``target_dir`` must not re-execute every other folder's scripts.
            # When ``target_dir`` is itself the render-output dir (direct
            # invocation), don't scope.
            scope_target: Optional[Path] = (
                target_dir
                if render_output_dir is not None and render_output_dir != target_dir
                else None
            )
            allowed_render_scripts, render_failures = _load_render_summary_contract(
                render_output_dir,
                requested_frameworks,
                logger,
                target_dir=scope_target,
            )
            execution_results["render_failures"] = render_failures

            if allowed_render_scripts is None and require_render_summary:
                log_step_warning(
                    logger,
                    f"Render summary contract not found or invalid: {render_output_dir / 'render_processing_summary.json'}",
                )
                execution_results["missing_render_summary"] = str(
                    render_output_dir / "render_processing_summary.json"
                )
                executable_scripts = []
            else:
                executable_scripts = find_executable_scripts(
                    render_output_dir,
                    verbose,
                    logger,
                    requested_frameworks,
                    allowed_scripts=allowed_render_scripts,
                )

            if allowed_render_scripts is not None:
                before_filter = len(executable_scripts)
                executable_scripts = [
                    script
                    for script in executable_scripts
                    if script["path"].resolve() in allowed_render_scripts
                ]
                found_allowed_scripts = {
                    script["path"].resolve() for script in executable_scripts
                }
                missing_render_scripts = sorted(
                    str(path) for path in allowed_render_scripts - found_allowed_scripts
                )
                execution_results["missing_render_scripts"] = missing_render_scripts
                if missing_render_scripts:
                    logger.error(
                        "Latest render summary references %d requested scripts not discoverable for execution",
                        len(missing_render_scripts),
                    )
                filtered_count = before_filter - len(executable_scripts)
                if filtered_count:
                    logger.info(
                        "Ignoring %d rendered scripts not present in the latest render summary",
                        filtered_count,
                    )
            execution_results["total_scripts_found"] = len(executable_scripts)
            execution_results["requested_frameworks"] = requested_frameworks

            if not executable_scripts:
                log_step_warning(logger, "No executable scripts found in render output")
                execution_results["message"] = "No executable scripts found"
                execution_results["skipped_reason"] = "no_executable_scripts"
            else:
                logger.info(
                    f"Found {len(executable_scripts)} executable scripts to run"
                )

                # Extract args
                timeout = kwargs.get("timeout", 3600)
                is_distributed = kwargs.get("distributed", False)
                execution_workers = _coerce_execution_workers(
                    kwargs.get("execution_workers", 1)
                )
                execution_results["execution_mode"] = (
                    "distributed" if is_distributed else "local"
                )
                execution_results["execution_workers"] = execution_workers
                execution_results["backend"] = (
                    kwargs.get("backend", "ray") if is_distributed else None
                )
                dispatch_max_retries = _coerce_dispatch_retries(
                    kwargs.get("distributed_max_retries", 3)
                )
                execution_results["dispatch_max_retries"] = (
                    dispatch_max_retries if is_distributed else 0
                )
                details: list[Any] = []

                if is_distributed:
                    from .distributed import Dispatcher

                    backend = kwargs.get("backend", "ray")
                    dispatcher = Dispatcher(
                        backend=backend,
                        num_cpus=execution_workers,
                        max_retries=dispatch_max_retries,
                    )

                    def ray_script_runner(info: Any, **kws: Any) -> Any:
                        """Execute a rendered simulation script using Ray for distributed processing."""
                        import logging

                        local_logger = logging.getLogger("execute.worker")
                        local_logger.setLevel(logging.INFO)
                        return execute_single_script(
                            info,
                            kws["results_dir"],
                            kws["verbose"],
                            local_logger,
                            kws["timeout"],
                            execution_benchmark_repeats=kws.get(
                                "execution_benchmark_repeats", 1
                            ),
                        )

                    try:
                        details = dispatcher.run_scripts_parallel(
                            executable_scripts,
                            ray_script_runner,
                            results_dir=results_dir,
                            verbose=verbose,
                            timeout=timeout,
                            execution_benchmark_repeats=execution_benchmark_repeats,
                        )
                        if len(details) != len(executable_scripts):
                            raise RuntimeError(
                                "distributed dispatcher returned "
                                f"{len(details)} results for "
                                f"{len(executable_scripts)} scripts"
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "Distributed %s dispatch failed for %d scripts: %s",
                            backend,
                            len(executable_scripts),
                            exc,
                        )
                        details = [
                            _make_distributed_dispatch_failure_result(
                                script_info,
                                exc,
                                str(backend),
                                dispatch_max_retries,
                            )
                            for script_info in executable_scripts
                        ]
                    finally:
                        dispatcher.shutdown()
                else:
                    details = _run_scripts_with_local_workers(
                        executable_scripts,
                        results_dir,
                        verbose,
                        logger,
                        timeout,
                        execution_workers,
                        execution_benchmark_repeats,
                    )

                # Update aggregated results
                _update_framework_status(execution_results, details)

        # Classify the outcome before serialising it. The previous flow wrote
        # ``success: true`` and only afterwards returned False for failures,
        # leaving the durable summary in direct conflict with the API result.
        total_found = execution_results["total_scripts_found"]
        successful = execution_results["successful_executions"]
        failed = execution_results["failed_executions"]
        skipped = execution_results.get("skipped_executions", 0)
        render_failures = execution_results.get("render_failures", [])
        missing_render_scripts = execution_results.get("missing_render_scripts", [])
        missing_render_summary = execution_results.get("missing_render_summary")

        classification = _classify_execute_outcome(
            total_found=total_found,
            successful=successful,
            failed=failed,
            skipped=skipped,
            render_failures=render_failures,
            missing_render_scripts=missing_render_scripts,
            missing_render_summary=missing_render_summary,
            strict_requested_frameworks=strict_requested_frameworks,
        )
        outcome = classification.outcome
        status = classification.status
        reason = classification.reason
        attempted = classification.attempted

        execution_results["total_scripts"] = total_found
        execution_results["attempted_scripts"] = attempted
        execution_results["success_rate"] = (
            round(successful / attempted * 100, 2) if attempted > 0 else 0.0
        )
        execution_results["success"] = outcome is True
        execution_results["status"] = status
        execution_results["exit_code"] = classification.exit_code
        execution_results["outcome_reason"] = reason

        # Save detailed results to summaries subfolder (slim aggregate +
        # optional full detail file). ``_write_execution_summaries`` merges any
        # prior summary, writes the slim + optional detail artifacts, regenerates
        # the report, and restores full details on ``execution_results``.
        _write_execution_summaries(
            results_dir, execution_results, execution_summary_detail, logger
        )

        if reason == "requested_framework_render_failure":
            failure_preview = "; ".join(
                f"{item['file']}:{item['framework']}" for item in render_failures[:5]
            )
            log_step_error(
                logger,
                f"Execute blocked by requested-framework render failures: {failure_preview}",
            )
        elif reason == "required_render_summary_missing":
            log_step_error(logger, "Execute requires a valid render summary contract")
        elif reason == "rendered_script_missing":
            log_step_error(
                logger,
                "Execute blocked because rendered scripts in the summary were not discoverable",
            )
        elif reason == "no_executable_scripts":
            log_step_warning(logger, "No executable scripts found to run")
        elif reason == "requested_framework_execution_incomplete":
            log_step_error(
                logger,
                f"Execute failed for requested frameworks: {successful} succeeded, "
                f"{failed} failed, {skipped} skipped",
            )
        elif reason == "script_execution_failure":
            log_step_error(
                logger,
                f"Execute failed: {successful} succeeded, {failed} failed, {skipped} skipped",
            )
        elif status == "success_with_skips":
            log_step_success(
                logger,
                f"Execute completed: {successful} succeeded, {skipped} skipped",
            )
        elif status == "success_with_render_failures":
            log_step_warning(
                logger,
                f"Execute completed {successful} scripts from the best-effort render subset",
            )
        else:
            log_step_success(logger, "Execute processing completed successfully")

        return outcome

    except Exception as e:
        log_step_error(logger, f"Execute processing failed: {e}")
        return False


def _aggregate_benchmark_samples(samples: List[float]) -> Dict[str, Any]:
    """Aggregate repeated execution durations (median + population std)."""
    import statistics

    if not samples:
        return {}
    med = float(statistics.median(samples))
    mean = float(statistics.mean(samples))
    std = float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0
    return {
        "execution_time": med,
        "execution_time_mean": mean,
        "execution_time_std": std,
        "execution_time_samples": list(samples),
    }


def _new_execution_result(context: ScriptExecutionContext) -> Dict[str, Any]:
    """Create the standard execution result envelope."""
    return _base_execution_envelope(
        script_path=str(context.script_path),
        script_name=context.script_name,
        framework=context.framework,
        model_name=context.model_name,
        executor=context.executor,
        status="failed",
        skipped=False,
    )


def _build_execution_environment(
    context: ScriptExecutionContext,
    results_dir: Path,
) -> Dict[str, str]:
    """Build environment variables for a rendered-script subprocess."""
    env = os.environ.copy()
    # Julia frameworks: default JULIA_PROJECT to the committed framework
    # environment so `using GnnRxInferModels` / `using ActiveInference`
    # resolve without an ambient env (e.g. a /tmp test env that may not
    # exist). An explicitly set JULIA_PROJECT still wins.
    julia_project = _julia_project_for_framework(context.framework)
    if julia_project is not None:
        env.setdefault("JULIA_PROJECT", str(julia_project))
    if context.framework == "pymdp":
        env["PYTHONPATH"] = (
            str(context.script_path.parent) + os.pathsep + env.get("PYTHONPATH", "")
        )
        proc_path = Path(__file__).resolve()
        repo_root = proc_path.parent.parent.parent
        env["GNN_PROJECT_ROOT"] = str(repo_root)
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        jax_platform = os.environ.get("GNN_JAX_PLATFORM")
        if jax_platform and str(jax_platform).strip():
            env["JAX_PLATFORM_NAME"] = str(jax_platform).strip()

    simulation_data_dir = (
        results_dir / context.model_name / context.framework / "simulation_data"
    )
    output_env_vars = {
        "jax": "GNN_OUTPUT_DIR",
        "numpyro": "NUMPYRO_OUTPUT_DIR",
        "pytorch": "PYTORCH_OUTPUT_DIR",
        "stan": "STAN_OUTPUT_DIR",
    }
    if context.framework in output_env_vars:
        simulation_data_dir.mkdir(parents=True, exist_ok=True)
        env[output_env_vars[context.framework]] = str(simulation_data_dir)

    return env


def _framework_for_data_helpers(framework: str) -> ExecutionFrameworkName:
    """Narrow framework names for typed data-collection helper calls."""
    return cast(ExecutionFrameworkName, framework)


#: Environment escape hatch: set to "1" to bypass the pre-execution security
#: gate (trusted-local research use only; see SECURITY.md).
_GNN_ALLOW_UNSAFE_EXEC = "GNN_ALLOW_UNSAFE_EXEC"


def _gnn_allow_unsafe_exec() -> bool:
    """Whether the operator has explicitly opted out of the pre-exec gate."""
    return os.environ.get(_GNN_ALLOW_UNSAFE_EXEC, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _sandbox_mode() -> str:
    """Effective sandbox mode from ``GNN_SANDBOX`` (default ``off``)."""
    from .sandbox import SANDBOX_MODES

    mode = os.environ.get("GNN_SANDBOX", "off").strip().lower()
    return mode if mode in SANDBOX_MODES else "off"


def _sandbox_command_prefix(mode: str) -> tuple[list[str], Optional[str]]:
    """Return ``(prefix, blocked_reason)`` for the requested sandbox mode.

    ``blocked_reason`` is non-None only when ``require`` cannot find a backend.
    """
    if mode == "off":
        return [], None
    from .sandbox import detect_sandbox

    spec = detect_sandbox()
    if spec is None:
        if mode == "require":
            return [], (
                "GNN_SANDBOX=require but no sandbox backend "
                "(firejail/bwrap/nsjail) is installed"
            )
        logger.warning(
            "GNN_SANDBOX=%s but no sandbox backend found; running unsandboxed",
            mode,
        )
        return [], None
    return list(spec.prefix), None


def execute_single_script(
    script_info: Dict[str, Any],
    results_dir: Path,
    verbose: bool,
    logger: Any,
    timeout: int = 3600,
    *,
    execution_benchmark_repeats: int = 1,
) -> Dict[str, Any]:
    """
    Execute a single script using subprocess.

    Args:
        script_info: Dictionary containing script information
        results_dir: Directory to save execution results (will create implementation-specific subfolders)
        verbose: Enable verbose logging
        logger: Logger instance

    Returns:
        Dictionary with execution results
    """
    context = _build_script_execution_context(script_info)
    script_path = context.script_path
    executor = context.executor
    model_name = context.model_name
    framework = context.framework

    # Pre-flight skip: do not run Python frameworks when optional dependency is missing
    if executor == sys.executable and not _is_python_framework_dependency_available(
        framework, executor, logger
    ):
        return _make_skipped_result(
            script_info, framework, model_name, executor, logger
        )

    # Prepare execution result
    exec_result = _new_execution_result(context)

    # Pre-execution security gate (RED_TEAM_REVIEW V-01/V-06): scan rendered
    # code BEFORE running it. Step 18 stays forensic; this closes the gap.
    if not _gnn_allow_unsafe_exec():
        try:
            from security.processor import scan_script_for_execution

            verdict = scan_script_for_execution(script_path)
            if not verdict.get("ok", True):
                blocked = verdict.get("blocked", [])
                detail = "; ".join(
                    f"{b.get('vulnerability_type', 'unknown')}@{b.get('line', '?')}"
                    for b in blocked[:5]
                )
                exec_result["error"] = (
                    f"Pre-execution security gate blocked {script_info['name']}: "
                    f"{detail}"
                )
                exec_result["error_type"] = "SecurityGateBlocked"
                exec_result["security_findings"] = blocked
                logger.error(exec_result["error"])
                return exec_result
        except ImportError:
            logger.debug("security.processor unavailable; pre-exec gate skipped")

    if framework == "rxinfer":
        exec_result["execution_metadata"] = (
            _load_rxinfer_execution_metadata_from_script(script_path)
        )

    try:
        if verbose:
            logger.info(
                f"Executing {script_info['framework']} script: {script_info['name']}"
            )

        # Check if the executor is available
        try:
            # For Python scripts, check if Python is available (most are Python scripts)
            if executor in ["python", "python3"]:
                subprocess.run(
                    [executor, "--version"],  # nosec B603
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )

                # For PyMDP, specifically check if it's importable
                if framework == "pymdp":
                    try:
                        import_check = subprocess.run(  # nosec B603
                            [executor, "-c", 'import pymdp; print("ok")'],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if import_check.returncode != 0:
                            logger.warning(
                                f"PyMDP package appears missing or broken: {import_check.stderr}"
                            )
                            exec_result["error"] = (
                                f"PyMDP dependency missing: {import_check.stderr}"
                            )
                            # Continue anyway as it might be a local import, but log warning
                    except Exception as e:
                        logger.debug(f"Error checking PyMDP importability: {e}")

            # For Julia scripts, check availability and dependencies
            elif executor == "julia":
                if not check_julia_dependencies(verbose, logger, [framework]):
                    skipped = _make_skipped_result(
                        script_info, framework, model_name, executor, logger
                    )
                    skipped["error"] = (
                        f"Julia packages required for {framework} are not available"
                    )
                    return skipped

                subprocess.run(
                    [executor, "--version"],  # nosec B603
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
            # For other executors, try a basic check
            else:
                subprocess.run(
                    [executor, "--version"],  # nosec B603
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ) as e:
            exec_result["error"] = (
                f"Executor '{executor}' is not available or not working: {e}"
            )
            exec_result["error_type"] = "ExecutorUnavailable"
            exec_result["return_code"] = -1
            logger.warning(
                f"Executor unavailable for {script_info['name']}: {executor}"
            )
            return exec_result

        # Execute the script with improved error handling. Failure carriers use
        # ``subprocess.CompletedProcess`` so the success and failure paths share
        # one return-code/stdout/stderr shape (no per-call class needed).
        result: subprocess.CompletedProcess[str] | None = None

        K = max(1, int(execution_benchmark_repeats))
        exec_result["execution_benchmark_repeats"] = K

        durations_success: List[float] = []
        broke_early = False

        try:
            env = _build_execution_environment(context, results_dir)

            sandbox_mode = _sandbox_mode()
            sandbox_prefix, sandbox_blocked = _sandbox_command_prefix(sandbox_mode)
            if sandbox_blocked is not None:
                exec_result["error"] = sandbox_blocked
                exec_result["error_type"] = "SandboxUnavailable"
                logger.error(sandbox_blocked)
                return exec_result
            base_command = _build_script_execution_command(context, sandbox_prefix)

            for rep in range(K):
                exec_result["attempts_started"] = rep + 1
                rep_start = datetime.now()
                try:
                    run_result = subprocess.run(  # nosec B603
                        base_command,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        cwd=script_path.parent,
                        env=env,
                    )
                except subprocess.TimeoutExpired:
                    exec_result["execution_time"] = (
                        datetime.now() - rep_start
                    ).total_seconds()
                    exec_result["error"] = (
                        f"Script execution timed out after {timeout} seconds"
                    )
                    exec_result["error_type"] = "TimeoutExpired"
                    exec_result["return_code"] = -1
                    exec_result["stdout"] = ""
                    exec_result["stderr"] = "Timeout"
                    logger.warning(
                        f"⏰ Script {script_info['name']} timed out after {timeout} seconds "
                        f"(rep {rep + 1}/{K})"
                    )
                    result = subprocess.CompletedProcess(
                        args=[], returncode=-1, stdout="", stderr="Timeout"
                    )
                    broke_early = True
                    break

                elapsed_rep = (datetime.now() - rep_start).total_seconds()

                if run_result.returncode != 0:
                    exec_result["execution_time"] = elapsed_rep
                    exec_result["return_code"] = run_result.returncode
                    exec_result["stdout"] = run_result.stdout
                    exec_result["stderr"] = run_result.stderr
                    exec_result["error"] = (
                        f"Script failed with return code {run_result.returncode}"
                    )

                    if "ModuleNotFoundError" in run_result.stderr:
                        exec_result["error_type"] = "DependencyError"
                        logger.error(
                            f"Missing dependency in {script_info['name']}: "
                            f"{run_result.stderr.splitlines()[-1]}"
                        )
                    elif "SyntaxError" in run_result.stderr:
                        exec_result["error_type"] = "SyntaxError"
                        logger.error(f"Syntax error in {script_info['name']}")
                    else:
                        exec_result["error_type"] = "RuntimeError"

                    logger.warning(
                        f"⚠️ Script {script_info['name']} failed with return code "
                        f"{run_result.returncode} (rep {rep + 1}/{K})"
                    )
                    if run_result.stderr:
                        logger.warning(f"Error output: {run_result.stderr[:500]}...")
                    result = run_result
                    broke_early = True
                    break

                durations_success.append(elapsed_rep)
                result = run_result
                if verbose and K > 1:
                    logger.info(
                        f"Benchmark rep {rep + 1}/{K} for {script_info['name']}: {elapsed_rep:.3f}s"
                    )

            if not broke_early and result is not None and len(durations_success) == K:
                agg = _aggregate_benchmark_samples(durations_success)
                exec_result.update(agg)
                exec_result["success"] = True
                exec_result["status"] = "success"
                exec_result["return_code"] = result.returncode
                exec_result["stdout"] = result.stdout
                exec_result["stderr"] = result.stderr
                if K == 1:
                    logger.info(f"✅ Successfully executed {script_info['name']}")
                else:
                    logger.info(
                        f"✅ Successfully executed {script_info['name']} "
                        f"({K} reps, median {exec_result['execution_time']:.3f}s)"
                    )
                if verbose and result.stdout:
                    logger.info(f"Script output: {result.stdout[:200]}...")
        except Exception as e:
            exec_result["execution_time"] = exec_result.get("execution_time", 0)
            exec_result["error"] = f"Script execution failed: {e}"
            exec_result["error_type"] = type(e).__name__
            exec_result["return_code"] = -2
            exec_result["stdout"] = ""
            exec_result["stderr"] = str(e)
            logger.warning(f"❌ Script {script_info['name']} execution failed: {e}")
            result = subprocess.CompletedProcess(
                args=[], returncode=-2, stdout="", stderr=str(e)
            )

        # Ensure result is defined before using it
        if result is None:
            result = subprocess.CompletedProcess(
                args=[], returncode=-3, stdout="", stderr="Unknown error"
            )

        # Save individual script output in implementation-specific subdirectory
        # Create the implementation-specific directory structure
        impl_specific_dir = results_dir / model_name / framework / "execution_logs"
        impl_specific_dir.mkdir(parents=True, exist_ok=True)

        # Note: Framework-specific subdirectories (visualizations, simulation_data, etc.)
        # are created on-demand by collect_execution_outputs() only when actual content
        # is copied to them, avoiding empty folder creation.

        # Extract simulation data from stdout/stderr
        data_framework = _framework_for_data_helpers(framework)
        simulation_data = _extract_simulation_data(
            result.stdout, result.stderr, data_framework, logger
        )
        exec_result["simulation_data"] = simulation_data

        # Hardware / accelerator metadata
        accelerator_type = _detect_accelerator_type()

        # Save structured execution results in JSON format
        structured_result: dict[str, Any] = {
            "framework": framework,
            "model_name": model_name,
            "script_name": script_info["name"],
            "script_path": str(script_path),
            "success": exec_result["success"],
            "return_code": exec_result.get("return_code"),
            "execution_time": exec_result.get("execution_time", 0),
            "timestamp": exec_result["timestamp"],
            "simulation_data": simulation_data,
            "execution_benchmark_repeats": exec_result.get(
                "execution_benchmark_repeats", 1
            ),
            "execution_metadata": {
                "executor": executor,
                "accelerator_type": accelerator_type,
                "stdout_length": len(result.stdout),
                "stderr_length": len(result.stderr),
                "output_directory": str(impl_specific_dir.parent),
                **exec_result.get("execution_metadata", {}),
            },
        }
        for bench_key in (
            "execution_time_mean",
            "execution_time_std",
            "execution_time_samples",
        ):
            if bench_key in exec_result:
                structured_result[bench_key] = exec_result[bench_key]

        # Save structured JSON result
        json_output_file = impl_specific_dir / f"{script_info['name']}_results.json"
        with open(json_output_file, "w") as f:
            json.dump(structured_result, f, indent=2, default=str)

        exec_result["structured_result_file"] = str(json_output_file)

        # Also save human-readable log
        output_file = impl_specific_dir / f"{script_info['name']}_execution.log"
        with open(output_file, "w") as f:
            f.write(f"Execution Results for {script_info['name']}\n")
            f.write(f"Timestamp: {exec_result['timestamp']}\n")
            f.write(f"Return Code: {result.returncode}\n")
            f.write(
                f"Benchmark repeats: {exec_result.get('execution_benchmark_repeats', 1)}\n"
            )
            f.write(
                f"Execution Time (median wall-clock): {exec_result['execution_time']:.2f} seconds\n"
            )
            if exec_result.get("execution_time_samples"):
                f.write(
                    f"Sample durations (s): {exec_result['execution_time_samples']}\n"
                )
            if exec_result.get("execution_time_std") is not None:
                f.write(
                    f"Duration mean/std (s): {exec_result.get('execution_time_mean', 0):.4f} / "
                    f"{exec_result.get('execution_time_std', 0):.4f}\n"
                )
            f.write(f"Model: {model_name}\n")
            f.write(f"Framework: {framework}\n")
            f.write(f"Output Directory: {impl_specific_dir.parent}\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)

        exec_result["output_file"] = str(output_file)
        exec_result["implementation_directory"] = str(impl_specific_dir.parent)

        # Collect execution outputs (visualizations, simulation data, traces)
        if exec_result["success"]:
            try:
                logger.info(
                    f"Collecting execution outputs for {framework} script {script_info['name']}"
                )
                collected_outputs = collect_execution_outputs(
                    script_path, impl_specific_dir.parent, data_framework, logger
                )
                exec_result["collected_outputs"] = collected_outputs

                # Update structured result with collected file paths
                structured_result["collected_outputs"] = collected_outputs

                # Re-save structured result with collected outputs
                with open(json_output_file, "w") as f:
                    json.dump(structured_result, f, indent=2, default=str)
                logger.debug("Updated results JSON with collected outputs")

                # Enhance simulation data extraction from collected files
                if collected_outputs:
                    logger.info(
                        f"Extracting simulation data from collected files for {framework}"
                    )
                    enhanced_data = _extract_simulation_data_from_files(
                        impl_specific_dir.parent, data_framework, logger
                    )
                    if enhanced_data:
                        logger.info(
                            f"Extracted {len(enhanced_data)} data fields from files"
                        )
                        simulation_data.update(enhanced_data)
                        exec_result["simulation_data"] = simulation_data
                        structured_result["simulation_data"] = simulation_data

                        # Re-save again with enhanced data
                        with open(json_output_file, "w") as f:
                            json.dump(structured_result, f, indent=2, default=str)
                        logger.debug(
                            "Updated results JSON with enhanced simulation data"
                        )
                    else:
                        logger.debug(
                            f"No additional data extracted from files for {framework}"
                        )

                if framework == "pymdp":
                    sim_dir = impl_specific_dir.parent / "simulation_data"
                    sr_candidates = list(sim_dir.glob("*simulation_results.json"))
                    if (
                        not sr_candidates
                        and (sim_dir / "simulation_results.json").exists()
                    ):
                        sr_candidates = [sim_dir / "simulation_results.json"]
                    if sr_candidates:
                        try:
                            with open(sr_candidates[0], encoding="utf-8") as sf:
                                payload = json.load(sf)
                            n_steps = payload.get("num_timesteps")
                            if n_steps is None:
                                n_steps = len(payload.get("observations", []))
                            logger.info(
                                "pymdp_execution_summary model=%s script=%s simulation_results=%s timesteps=%s",
                                model_name,
                                script_info["name"],
                                sr_candidates[0],
                                n_steps,
                            )
                        except (OSError, json.JSONDecodeError, TypeError) as ex:
                            logger.debug("pymdp_execution_summary skipped: %s", ex)

            except Exception as e:
                logger.warning(f"Failed to collect execution outputs: {e}")
                import traceback

                logger.debug(traceback.format_exc())

    except subprocess.TimeoutExpired:
        exec_result["error"] = f"Script execution timed out ({timeout} seconds)"
        logger.error(f"Script {script_info['name']} timed out")

    except Exception as e:
        exec_result["error"] = str(e)
        exec_result["error_type"] = type(e).__name__
        logger.error(f"Error executing {script_info['name']}: {e}")

    return exec_result


def execute_simulation_from_gnn(gnn_file: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Execute simulation from GNN file.

    Args:
        gnn_file: Path to GNN file
        output_dir: Output directory

    Returns:
        Dictionary with execution results
    """
    try:
        logger.info(f"Executing simulation for {gnn_file}")

        from .executor import GNNExecutor

        engine = GNNExecutor()

        # Execute simulation
        result = engine.execute_simulation_from_gnn(gnn_file, output_dir)

        return result

    except Exception as e:
        logger.error(f"Failed to execute simulation for {gnn_file}: {e}")
        return {"success": False, "error": str(e)}
