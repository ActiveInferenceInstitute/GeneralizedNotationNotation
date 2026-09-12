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

import contextlib
import importlib
import io
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, TextIO, Tuple, cast

from gnn.pipeline.config import get_output_dir_for_script
from gnn.pipeline.step_registry import (
    CONSOLIDATED_IN_PROCESS_STEMS,
    canonical_step_stem,
    step_for_stem,
)
from gnn.pipeline.step_timeouts import get_step_timeout
from gnn.utils.arguments.step_config import StepConfiguration
from gnn.utils.errors.error_handling import coerce_step_exit_code, status_from_exit_code
from gnn.utils.observability.structured_logging import log_step_warning
from gnn.utils.pipeline_orchestration.pipeline_validator import (
    validate_step_prerequisites,
)
from gnn.utils.runtime_safety.resource_manager import get_current_memory_usage

__all__ = [
    "UnsupportedStepError",
    "can_execute_in_process",
    "clear_parsed_model_carrier",
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


# ---------------------------------------------------------------------------
# Console capture: tee stdout/stderr into the receipt (console still streams)
# ---------------------------------------------------------------------------
class _ConsoleStreamTee:
    """Thread-scoped console tee used while a consolidated step runs.

    Writes from the step's worker thread are tee'd: forwarded to the real
    console stream and captured into that thread's sink for the receipt.
    Writes from every other thread pass through untouched, so pipeline
    logging is never mis-attributed to a step. Install/uninstall are
    reference-counted under a lock so concurrent consolidated steps (e.g.
    the parallel tier) never leave the process streams swapped.
    """

    def __init__(self, stream_attr: str) -> None:
        self._attr = stream_attr
        self._lock = threading.Lock()
        self._depth = 0
        self._saved: Any = None
        self._local = threading.local()

    def _live_target(self) -> TextIO:
        saved = self._saved
        target = saved if saved is not None else getattr(sys, self._attr)
        return cast("TextIO", target)

    def open_for_thread(self, sink: Any) -> None:
        """Route the calling thread's writes into *sink* (plus console)."""
        self._local.sink = sink

    def close_for_thread(self) -> None:
        """Stop capturing the calling thread's writes."""
        self._local.sink = None

    def install(self) -> None:
        with self._lock:
            if self._depth == 0:
                self._saved = getattr(sys, self._attr)
                setattr(sys, self._attr, self)
            self._depth += 1

    def uninstall(self) -> None:
        with self._lock:
            if self._depth > 0:
                self._depth -= 1
            if self._depth == 0 and self._saved is not None:
                setattr(sys, self._attr, self._saved)
                self._saved = None

    def write(self, text: str) -> int:
        sink = getattr(self._local, "sink", None)
        if sink is not None:
            try:
                sink.write(text)
            except Exception:  # capture must never break the step's output
                pass
        return self._live_target().write(text)

    def flush(self) -> None:
        self._live_target().flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._live_target(), name)


_STDOUT_CAPTURE = _ConsoleStreamTee("stdout")
_STDERR_CAPTURE = _ConsoleStreamTee("stderr")


@contextlib.contextmanager
def _console_capture_window() -> Generator[None, None, None]:
    """Swap in the capture tees for the calling thread's step run."""
    _STDOUT_CAPTURE.install()
    _STDERR_CAPTURE.install()
    try:
        yield
    finally:
        _STDERR_CAPTURE.uninstall()
        _STDOUT_CAPTURE.uninstall()


# ---------------------------------------------------------------------------
# Parsed-model carrier: in-memory step 3 -> steps 7/8 handoff
# ---------------------------------------------------------------------------
#: Steps whose module functions consume a ``parsed_model`` carrier kwarg.
_PARSED_MODEL_CONSUMER_STEMS = frozenset({"7_export", "8_visualization"})

#: Step-3 parse artifacts collected once per run, keyed by the resolved
#: output dir. Serial within a run; lock-guarded for concurrent tiers.
_PARSED_MODEL_CARRIERS: Dict[str, Dict[str, Any]] = {}
_CARRIER_LOCK = threading.Lock()


def _carrier_key(output_dir: Any) -> str:
    """Carrier cache key: the run's resolved output directory."""
    return os.fspath(Path(output_dir).resolve())


def _parsed_model_carrier_for(args: Any) -> Optional[Dict[str, Any]]:
    """Return the collected step-3 carrier for *args*' run, if any."""
    with _CARRIER_LOCK:
        return _PARSED_MODEL_CARRIERS.get(_carrier_key(args.output_dir))


def _collect_parsed_model_carrier(
    output_dir: Path, logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Read step 3's on-disk parse artifacts exactly once.

    The carrier bundles ``gnn_processing_results.json`` plus every
    successful ``{model}_parsed.json`` it references, keyed by source-file
    stem — the same bytes steps 7/8 would otherwise re-read from disk.
    """
    step3_dir = get_output_dir_for_script("3_gnn", Path(output_dir))
    results_file = step3_dir / "gnn_processing_results.json"
    if not results_file.is_file():
        return None
    try:
        with open(results_file, encoding="utf-8") as handle:
            results = json.load(handle)
    except (OSError, ValueError) as error:
        logger.warning(
            "Parsed-model carrier: could not read %s: %s", results_file, error
        )
        return None
    if not isinstance(results, dict) or not isinstance(
        results.get("processed_files"), list
    ):
        return None
    models: Dict[str, Any] = {}
    for entry in results["processed_files"]:
        if not isinstance(entry, dict) or not entry.get("parse_success"):
            continue
        parsed_path_text = entry.get("parsed_model_file")
        if not parsed_path_text:
            continue
        parsed_path = Path(parsed_path_text)
        try:
            with open(parsed_path, encoding="utf-8") as handle:
                model = json.load(handle)
        except (OSError, ValueError) as error:
            logger.warning(
                "Parsed-model carrier: could not read %s: %s", parsed_path, error
            )
            continue
        source_path = entry.get("file_path")
        key = Path(source_path).stem if source_path else parsed_path.parent.name
        models[key] = model
    return {"results": results, "models": models}


def _refresh_parsed_model_carrier(
    args: Any,
    logger: logging.Logger,
    *,
    collect: bool,
    succeeded: bool,
) -> None:
    """Refresh or drop the run's carrier after an in-process step 3.

    Collected only when the caller opted in and step 3 succeeded; a
    non-collecting (or failed) step 3 rewrites the artifacts the carrier
    describes, so any cached carrier for the run is dropped rather than
    left stale.
    """
    key = _carrier_key(args.output_dir)
    if not collect or not succeeded:
        with _CARRIER_LOCK:
            _PARSED_MODEL_CARRIERS.pop(key, None)
        return
    carrier = _collect_parsed_model_carrier(Path(args.output_dir), logger)
    with _CARRIER_LOCK:
        if carrier is None:
            _PARSED_MODEL_CARRIERS.pop(key, None)
        else:
            _PARSED_MODEL_CARRIERS[key] = carrier


def clear_parsed_model_carrier(output_dir: Optional[Path] = None) -> None:
    """Drop cached parsed-model carriers (one run's, or all when omitted).

    Exposed for callers that re-run steps outside the executor (tests,
    tooling): the cache must never outlive the artifacts it mirrors.
    """
    with _CARRIER_LOCK:
        if output_dir is None:
            _PARSED_MODEL_CARRIERS.clear()
        else:
            _PARSED_MODEL_CARRIERS.pop(_carrier_key(output_dir), None)


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
    timeout_seconds: Optional[int] = None,
    collect_parsed_model: bool = False,
) -> Dict[str, Any]:
    """Run one whitelisted step in-process and return its execution receipt.

    The receipt matches ``execute_pipeline_step``'s ``step_result`` contract
    (same keys, same status mapping) so the shared recording tail in ``main``
    treats both modes identically; ``execution_mode`` records which mode ran.
    Raises :class:`UnsupportedStepError` for steps outside the consolidated
    whitelist or under testing-matrix folder dispatch.

    The step runs on a worker thread with a wall-clock timeout sourced from
    ``gnn.pipeline.step_timeouts`` — the same knob as the subprocess tier —
    and its ``print()``/``sys.stdout``/``sys.stderr`` output is tee'd to the
    console and captured into the receipt's ``stdout``/``stderr`` fields.
    A timeout records the subprocess tier's timeout receipt semantics (exit
    code -1, FAILED status, partial captured streams) with the receipt
    schema unchanged; an in-process step cannot be force-killed, which the
    receipt's stderr text states honestly.

    With ``collect_parsed_model=True`` the executor participates in the
    parsed-model carrier: a successful in-process step 3 is read from disk
    exactly once and cached for the run, and in-process steps 7/8 receive
    it as a ``parsed_model`` kwarg (absent flag: steps behave exactly as
    before).
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

    call_kwargs: Dict[str, Any] = {
        "target_dir": Path(args.target_dir),
        "output_dir": step_output_dir,
        "logger": logger,
        "recursive": args.recursive,
        "verbose": args.verbose,
    }
    call_kwargs.update(_in_process_step_kwargs(stem, args))
    if collect_parsed_model and stem in _PARSED_MODEL_CONSUMER_STEMS:
        carrier = _parsed_model_carrier_for(args)
        if carrier is not None:
            call_kwargs["parsed_model"] = carrier

    if timeout_seconds is None:
        # Same budget knob as the subprocess tier (main.py): the per-step
        # STEP_TIMEOUTS value with the GNN_STEP_TIMEOUT_{N} /
        # GNN_STEP_TIMEOUT_SCALE env overrides.
        comprehensive_requested = any("--comprehensive" in str(arg) for arg in sys.argv)
        timeout_seconds = get_step_timeout(
            script_name, comprehensive=comprehensive_requested
        )

    call_started = time.time()
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()

    def _invoke() -> Tuple[Any, Optional[BaseException]]:
        _STDOUT_CAPTURE.open_for_thread(captured_stdout)
        _STDERR_CAPTURE.open_for_thread(captured_stderr)
        try:
            with _console_capture_window():
                return function(**call_kwargs), None
        except BaseException as error:  # receipt contract owns the failure shape
            return None, error
        finally:
            _STDERR_CAPTURE.close_for_thread()
            _STDOUT_CAPTURE.close_for_thread()

    worker_pool = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=f"gnn-consolidated-{stem}"
    )
    try:
        future = worker_pool.submit(_invoke)
        try:
            result, error = future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            # Mirror the subprocess tier's timeout receipt
            # (execute_command_streaming -> execute_pipeline_step): exit
            # code -1, FAILED status, partial captured streams, unchanged
            # receipt schema. In-process steps cannot be force-killed; that
            # limit is stated in the stderr text.
            future.cancel()  # best effort: running steps cannot be cancelled
            end_memory = get_current_memory_usage()
            step_result["exit_code"] = -1
            step_result["status"] = status_from_exit_code(
                step_result["exit_code"], step_result["dependency_warnings"]
            )
            step_result["stdout"] = captured_stdout.getvalue()
            timeout_notice = (
                f"Consolidated in-process execution of {step.script_name} "
                f"exceeded its {timeout_seconds}s wall-clock timeout "
                f"(TIMEOUT; recorded as FAILED with exit code -1). "
                "In-process steps cannot be force-killed: the step thread "
                "keeps running and any further output is discarded from "
                "this receipt."
            )
            partial_stderr = captured_stderr.getvalue()
            step_result["stderr"] = (
                partial_stderr
                + ("\n" if partial_stderr else "")
                + timeout_notice
                + "\n"
            )
            step_result["memory_usage_mb"] = end_memory
            step_result["peak_memory_mb"] = max(start_memory, end_memory)
            step_result["memory_delta_mb"] = end_memory - start_memory
            logger.error(
                "Consolidated in-process execution of %s timed out after "
                "%ss; reported as FAILED (exit code -1). The step thread "
                "was not cancelled (in-process steps cannot be "
                "force-killed).",
                step.script_name,
                timeout_seconds,
            )
            return step_result
    finally:
        worker_pool.shutdown(wait=False)

    if error is not None:
        logger.error(
            "Consolidated in-process execution of %s failed: %s",
            step.script_name,
            error,
        )
        step_result["stderr"] = (
            captured_stderr.getvalue() + f"{type(error).__name__}: {error}\n"
        )
        result = False
    else:
        step_result["stderr"] = captured_stderr.getvalue()

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
    step_result["stdout"] = captured_stdout.getvalue() + (
        f"{step.script_name}: consolidated in-process execution completed "
        f"in {time.time() - call_started:.2f}s\n"
    )
    step_result["memory_usage_mb"] = end_memory
    step_result["peak_memory_mb"] = max(start_memory, end_memory)
    step_result["memory_delta_mb"] = end_memory - start_memory

    if stem == "3_gnn":
        _refresh_parsed_model_carrier(
            args,
            logger,
            collect=collect_parsed_model,
            succeeded=step_result["status"] in ("SUCCESS", "SUCCESS_WITH_WARNINGS"),
        )
    return step_result
