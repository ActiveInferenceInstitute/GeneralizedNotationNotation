#!/usr/bin/env python3
"""
GNN Processing Pipeline

This script orchestrates the 25-step GNN processing pipeline (steps 0-24).
The pipeline transforms GNN specifications into executable simulations, visualizations,
and advanced representations including audio sonification.

Pipeline Steps:
0. Template initialization (0_template.py)
1. Environment setup (1_setup.py)
2. Test suite execution (2_tests.py)
3. GNN file processing (3_gnn.py)
4. Model registry (4_model_registry.py)
5. Type checking (5_type_checker.py)
6. Validation (6_validation.py)
7. Multi-format export (7_export.py)
8. Visualization (8_visualization.py)
9. Advanced visualization (9_advanced_viz.py)
10. Ontology processing (10_ontology.py)
11. Code rendering (11_render.py)
12. Execution (12_execute.py)
13. LLM processing (13_llm.py)
14. ML integration (14_ml_integration.py)
15. Audio processing (15_audio.py)
16. Analysis (16_analysis.py)
17. Integration (17_integration.py)
18. Security (18_security.py)
19. Research (19_research.py)
20. Website generation (20_website.py)
21. Model Context Protocol processing (21_mcp.py)
22. GUI (Interactive GNN Constructor) (22_gui.py)
23. Report generation (23_report.py)
24. Intelligent analysis (24_intelligent_analysis.py)

Usage:
    python src/main.py [options]

Examples:
    # Run full pipeline
    python src/main.py --target-dir input/gnn_files --verbose

    # Run specific steps only
    python src/main.py --only-steps "0,1,2,3" --verbose

    # Skip certain steps
    python src/main.py --skip-steps "15,16" --verbose

For complete usage information, see:
- README.md: Project overview and quick start
- doc/pipeline/README.md: Detailed pipeline documentation
- src/README.md: Pipeline safety and reliability documentation
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_module_logger = logging.getLogger(__name__)

# Detect project root and ensure we're working from there
SCRIPT_DIR = Path(__file__).parent  # src/
PROJECT_ROOT = SCRIPT_DIR.parent  # project root (one level up from src/)

# Change working directory to project root if not already there
if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)
    logging.getLogger(__name__).info(
        f"Changed working directory to project root: {PROJECT_ROOT}"
    )

# Add src to path for imports
sys.path.insert(0, str(SCRIPT_DIR))

from dataclasses import fields

from utils.argument_utils import ArgumentParser, PipelineArguments

# Structured logging and visual progress tracking are maintained pipeline
# surfaces; import errors should fail loudly during startup.
from utils.logging.logging_utils import (
    PipelineLogger,
    PipelineProgressTracker,
    log_pipeline_summary,
    log_step_error,
    log_step_start,
    reset_progress_tracker,
    rotate_logs,
    setup_step_logging,
)
from utils.pipeline_config_merge import apply_input_config_defaults
from utils.pipeline_step_dependencies import resolve_step_dependencies
from utils.pipeline_validator import (
    validate_pipeline_step_sequence,
    validate_step_prerequisites,
)
from utils.resource_manager import get_current_memory_usage
from utils.visual_logging import (
    VisualConfig,
    VisualLogger,
    create_visual_logger,
    print_completion_summary,
    print_pipeline_banner,
    print_step_summary,
)

STRUCTURED_LOGGING_AVAILABLE = True
PipelineStep = tuple[str, str]

PIPELINE_STEPS: tuple[PipelineStep, ...] = (
    ("0_template.py", "Template initialization"),
    ("1_setup.py", "Environment setup"),
    ("2_tests.py", "Test suite execution"),
    ("3_gnn.py", "GNN file processing"),
    ("4_model_registry.py", "Model registry"),
    ("5_type_checker.py", "Type checking"),
    ("6_validation.py", "Validation"),
    ("7_export.py", "Multi-format export"),
    ("8_visualization.py", "Visualization"),
    ("9_advanced_viz.py", "Advanced visualization"),
    ("10_ontology.py", "Ontology processing"),
    ("11_render.py", "Code rendering"),
    ("12_execute.py", "Execution"),
    ("13_llm.py", "LLM processing"),
    ("14_ml_integration.py", "ML integration"),
    ("15_audio.py", "Audio processing"),
    ("16_analysis.py", "Analysis"),
    ("17_integration.py", "Integration"),
    ("18_security.py", "Security"),
    ("19_research.py", "Research"),
    ("20_website.py", "Website generation"),
    ("21_mcp.py", "Model Context Protocol processing"),
    ("22_gui.py", "GUI (Interactive GNN Constructor)"),
    ("23_report.py", "Report generation"),
    ("24_intelligent_analysis.py", "Intelligent pipeline analysis"),
)

CRITICAL_SCRIPTS: set[str] = {
    "0_template.py",
    "1_setup.py",
    "3_gnn.py",
    "5_type_checker.py",
    "6_validation.py",
    "7_export.py",
    "11_render.py",
}

SAFE_WARNING_PATTERNS: tuple[str, ...] = (
    r"matplotlib.*?backend",
    r"using agg backend",
    r"no display",
    r"pymdp.*?not available",
    r"optional.*?dependency",
    r"plotly.*?not available",
    r"numpy.*?not available",
    r"seaborn.*?not available",
    r"bokeh.*?not available",
    r"d2.*?not available",
    r"d2 cli.*?not available",
    r"d2 visualizer.*?not available",
    r"d2 cli.*?install",
    r"interactive.*?limited",
    r"numeric.*?limited",
    r"warnings: 0",
    r"optional test tooling not installed",
)

WARNING_PATTERN = re.compile(r"(WARNING|⚠️|warn)", re.IGNORECASE)
SAFE_WARNING_PATTERN = re.compile(
    "|".join(f"({pattern})" for pattern in SAFE_WARNING_PATTERNS),
    re.IGNORECASE,
)


def _build_main_args(
    override_args: Optional[PipelineArguments],
) -> tuple[PipelineArguments, Optional[argparse.Namespace]]:
    """Build pipeline arguments from an override object or CLI parsing."""
    if override_args is not None:
        return override_args, None

    parser = ArgumentParser.create_main_parser()
    parsed = parser.parse_args()
    field_names = {f.name for f in fields(PipelineArguments)}
    kwargs = {k: getattr(parsed, k) for k in field_names if hasattr(parsed, k)}
    args = PipelineArguments(**kwargs)

    if getattr(args, "skip_llm", False):
        existing = args.skip_steps or ""
        existing_nums = (
            [s.strip() for s in str(existing).split(",") if s.strip()]
            if existing
            else []
        )
        if "13" not in existing_nums:
            existing_nums.append("13")
        args.skip_steps = ",".join(existing_nums)

    return args, parsed


def _create_pipeline_visual_logger(
    args: PipelineArguments,
) -> tuple[VisualLogger, str]:
    """Create the visual logger and attach a short correlation ID."""
    visual_config = VisualConfig(
        enable_colors=True,
        enable_progress_bars=True,
        enable_emoji=True,
        enable_animation=True,
        show_timestamps=args.verbose,
        show_correlation_ids=True,
        compact_mode=False,
    )
    visual_logger = create_visual_logger("pipeline", visual_config)
    correlation_id = str(uuid.uuid4())[:8]
    visual_logger.set_correlation_id(correlation_id)
    return visual_logger, correlation_id


def _setup_pipeline_logger(args: PipelineArguments) -> logging.Logger:
    """Initialize structured pipeline logging for a fresh run."""
    if STRUCTURED_LOGGING_AVAILABLE:
        log_dir = args.output_dir / "00_pipeline_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        rotate_logs(log_dir)

        PipelineLogger._initialized = False
        if PipelineLogger._log_file_handler:
            logging.getLogger().removeHandler(PipelineLogger._log_file_handler)
            PipelineLogger._log_file_handler.close()
            PipelineLogger._log_file_handler = None

        PipelineLogger.initialize(
            log_dir=log_dir, enable_structured=True, log_format=args.log_format
        )
        PipelineLogger.enable_json_logging(log_dir)
        logger = setup_step_logging(
            "pipeline", args.verbose, enable_structured=True, log_format=args.log_format
        )
        reset_progress_tracker()
        return logger

    return setup_step_logging(
        "pipeline",
        args.verbose,
        log_format=args.log_format if hasattr(args, "log_format") else "human",
    )


def _log_backend_versions(logger: logging.Logger) -> None:
    """Log optional execution backend versions during verbose runs."""
    parts: list[str] = []
    for mod, label in (
        ("jax", "jax"),
        ("jaxlib", "jaxlib"),
        ("torch", "torch"),
        ("numpyro", "numpyro"),
        ("discopy", "discopy"),
    ):
        try:
            module = __import__(mod)
            parts.append(f"{label}={getattr(module, '__version__', '?')}")
        except ImportError:
            parts.append(f"{label}=missing")
    logger.info("Step 12 backends: %s", "; ".join(parts))


def _load_pipeline_config(
    override_config: Optional[Dict[str, Any]],
    logger: logging.Logger,
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Load `input/config.yaml` unless a test or caller supplies config directly."""
    full_config: dict[Any, Any]
    if override_config is not None:
        full_config = override_config
        return full_config, full_config.get("pipeline", {})

    full_config = {}
    config_pipeline_settings: dict[Any, Any] = {}
    input_config_path = Path("input/config.yaml")
    if not input_config_path.exists():
        return full_config, config_pipeline_settings

    try:
        import yaml

        with open(input_config_path, "r") as f:
            full_config = yaml.safe_load(f) or {}
        config_pipeline_settings = full_config.get("pipeline", {})
    except Exception as e:
        logger.warning(f"Could not load pipeline settings from input/config.yaml: {e}")

    return full_config, config_pipeline_settings


def _resolve_steps_to_execute(
    args: PipelineArguments,
    config_pipeline_settings: dict[Any, Any],
    logger: logging.Logger,
) -> list[PipelineStep]:
    """Apply only/skip step filters and automatic dependency resolution."""
    pipeline_steps = list(PIPELINE_STEPS)
    steps_to_execute = pipeline_steps

    only_steps_val = args.only_steps or config_pipeline_settings.get("only_steps")
    if only_steps_val:
        requested_step_numbers = parse_step_list(only_steps_val)
        resolved_step_numbers = resolve_step_dependencies(requested_step_numbers)
        added_dependencies = sorted(
            set(resolved_step_numbers) - set(requested_step_numbers)
        )

        if added_dependencies:
            logger.info(f"Auto-including dependency steps: {added_dependencies}")

        steps_to_execute = [
            pipeline_steps[i]
            for i in resolved_step_numbers
            if 0 <= i < len(pipeline_steps)
        ]
        logger.info(f"Executing steps: {[step[0] for step in steps_to_execute]}")

    cmd_skip = parse_step_list(args.skip_steps)
    cfg_skip = parse_step_list(config_pipeline_settings.get("skip_steps"))
    skip_numbers = sorted(set(cmd_skip + cfg_skip))

    if skip_numbers:
        original_indices = {script: i for i, (script, _) in enumerate(pipeline_steps)}
        steps_to_execute = [
            step
            for step in steps_to_execute
            if original_indices.get(step[0], -1) not in skip_numbers
        ]
        logger.info(
            f"Skipping steps: {[pipeline_steps[i][0] for i in skip_numbers if 0 <= i < len(pipeline_steps)]}"
        )

    return steps_to_execute


def _initialize_pipeline_summary(
    args: PipelineArguments,
    steps_to_execute: list[PipelineStep],
    config_pipeline_settings: dict[Any, Any],
) -> dict[str, Any]:
    """Create the initial summary payload before step execution starts."""
    from pipeline.hasher import compute_run_hash_with_files

    run_hash, file_hashes = compute_run_hash_with_files(
        args.target_dir,
        config=config_pipeline_settings,
    )

    return {
        "run_hash": run_hash,
        "file_hashes": file_hashes,
        "start_time": datetime.now().isoformat(),
        "arguments": args.to_dict(),
        "steps": [],
        "end_time": None,
        "overall_status": "RUNNING",
        "total_duration_seconds": None,
        "environment_info": get_environment_info(),
        "performance_summary": {
            "peak_memory_mb": 0.0,
            "total_steps": len(steps_to_execute),
            "failed_steps": 0,
            "critical_failures": 0,
            "successful_steps": 0,
            "warnings": 0,
        },
    }


def _prepare_pipeline_context(
    override_args: Optional[PipelineArguments],
    override_config: Optional[Dict[str, Any]],
) -> tuple[
    PipelineArguments,
    dict[Any, Any],
    list[PipelineStep],
    dict[str, Any],
    VisualLogger,
    str,
    logging.Logger,
]:
    """Prepare args, config, logging, step list, and initial summary."""
    args, parsed = _build_main_args(override_args)
    visual_logger, correlation_id = _create_pipeline_visual_logger(args)
    logger = _setup_pipeline_logger(args)

    if args.verbose:
        _log_backend_versions(logger)

    full_config, config_pipeline_settings = _load_pipeline_config(
        override_config, logger
    )
    apply_input_config_defaults(args, full_config, parsed)

    steps_to_execute = _resolve_steps_to_execute(args, config_pipeline_settings, logger)
    pipeline_summary = _initialize_pipeline_summary(
        args, steps_to_execute, config_pipeline_settings
    )

    return (
        args,
        config_pipeline_settings,
        steps_to_execute,
        pipeline_summary,
        visual_logger,
        correlation_id,
        logger,
    )


def _start_pipeline_run(
    args: PipelineArguments,
    steps_to_execute: list[PipelineStep],
    pipeline_summary: dict[str, Any],
    visual_logger: VisualLogger,
    correlation_id: str,
    logger: logging.Logger,
) -> Optional[PipelineProgressTracker]:
    """Print start banners, wire progress tracking, and validate the step sequence."""
    print_pipeline_banner(
        "Generalized Notation Notation (GNN)",
        "https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation | Active Inference Institute",
    )
    print_pipeline_banner(
        "🚀 GNN Processing Pipeline",
        f"Starting execution with {len(steps_to_execute)} steps | Correlation ID: {correlation_id}",
    )
    visual_logger.print_progress(0, len(steps_to_execute), "Pipeline initialization")

    if STRUCTURED_LOGGING_AVAILABLE:
        PipelineLogger.log_structured(
            logger,
            logging.INFO,
            "🚀 Starting GNN Processing Pipeline",
            total_steps=len(steps_to_execute),
            target_dir=str(args.target_dir),
            output_dir=str(args.output_dir),
            event_type="pipeline_start",
        )
    else:
        log_step_start(logger, "Starting GNN Processing Pipeline")

    pipeline_summary["performance_summary"]["total_steps"] = len(steps_to_execute)

    progress_tracker = None
    if STRUCTURED_LOGGING_AVAILABLE:
        progress_tracker = PipelineProgressTracker(len(steps_to_execute))
        from utils.logging.logging_utils import set_global_progress_tracker

        set_global_progress_tracker(progress_tracker)

    sequence_validation = validate_pipeline_step_sequence(steps_to_execute, logger)
    for warning in sequence_validation["warnings"]:
        logger.warning(f"Pipeline sequence: {warning}")
    for rec in sequence_validation["recommendations"]:
        logger.info(f"Recommendation: {rec}")

    return progress_tracker


def _write_preliminary_pipeline_summary(
    pipeline_summary: dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Write a current-run summary for report and intelligent-analysis steps."""
    try:
        prelim_summary = dict(pipeline_summary)
        prelim_end = datetime.now()
        prelim_start = datetime.fromisoformat(prelim_summary["start_time"])
        prelim_summary["end_time"] = prelim_end.isoformat()
        prelim_summary["total_duration_seconds"] = (
            prelim_end - prelim_start
        ).total_seconds()
        prelim_summary["overall_status"] = "SUCCESS"
        prelim_summary["preliminary"] = True

        prelim_path = (
            output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"
        )
        prelim_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prelim_path, "w") as f:
            json.dump(prelim_summary, f, indent=4, default=str)
        logger.info(
            f"📝 Preliminary pipeline summary written for intelligent analysis ({len(prelim_summary.get('steps', []))} steps)"
        )
    except Exception as prelim_err:
        logger.warning(f"Could not write preliminary summary: {prelim_err}")


def _log_pipeline_step_start(
    actual_step_number: int,
    script_name: str,
    description: str,
    total_steps: int,
    progress_tracker: Optional[PipelineProgressTracker],
    logger: logging.Logger,
) -> None:
    """Log step start through structured logging when available."""
    if STRUCTURED_LOGGING_AVAILABLE and progress_tracker:
        progress_tracker.start_step(actual_step_number, description)
        from utils.logging.logging_utils import (
            log_step_start as structured_log_step_start,
        )

        structured_log_step_start(
            logger,
            f"Starting {description}",
            step_number=actual_step_number,
            total_steps=total_steps,
            script_name=script_name,
        )
    else:
        logger.info(f"🔄 Executing step {actual_step_number}: {description}")


def _annotate_step_result(
    step_result: dict[str, Any],
    actual_step_number: int,
    script_name: str,
    description: str,
    step_start_datetime: datetime,
    step_end_datetime: datetime,
    step_duration: float,
) -> None:
    """Attach timing and standard metadata to a raw step result."""
    step_result.update(
        {
            "step_number": actual_step_number,
            "script_name": script_name,
            "description": description,
            "start_time": step_start_datetime.isoformat(),
            "end_time": step_end_datetime.isoformat(),
            "duration_seconds": step_duration,
            "exit_code": step_result.get("exit_code", 0),
            "retry_count": step_result.get("retry_count", 0),
            "prerequisite_check": step_result.get("prerequisite_check", True),
            "dependency_warnings": step_result.get("dependency_warnings", []),
            "recoverable": step_result.get("recoverable", False),
            "memory_usage_mb": step_result.get("memory_usage_mb", 0.0),
            "peak_memory_mb": step_result.get("peak_memory_mb", 0.0),
            "memory_delta_mb": step_result.get("memory_delta_mb", 0.0),
        }
    )


def _step_has_actionable_warning(step_result: dict[str, Any]) -> bool:
    """Return true when step output contains warnings not on the safe-noise list."""
    combined_output = (
        f"{step_result.get('stdout', '')}\n{step_result.get('stderr', '')}"
    )
    warning_lines = [
        line for line in combined_output.splitlines() if WARNING_PATTERN.search(line)
    ]
    if not warning_lines:
        return False

    return any(not SAFE_WARNING_PATTERN.search(line) for line in warning_lines)


def _update_performance_summary(
    pipeline_summary: dict[str, Any],
    step_result: dict[str, Any],
    script_name: str,
    has_warning: bool,
    total_steps: int,
) -> None:
    """Update aggregate counts and peak memory after a step finishes."""
    perf_summary = pipeline_summary["performance_summary"]
    if step_result["status"] in ("SUCCESS", "SUCCESS_WITH_WARNINGS", "SKIPPED"):
        perf_summary["successful_steps"] += 1
    elif step_result["status"] == "FAILED":
        perf_summary["failed_steps"] += 1
        if script_name in CRITICAL_SCRIPTS and step_result.get("exit_code", 0) != 0:
            perf_summary["critical_failures"] += 1

    if has_warning or step_result.get("status") == "SUCCESS_WITH_WARNINGS":
        perf_summary["warnings"] += 1

    step_memory = step_result.get("memory_usage_mb", 0.0)
    step_peak_memory = step_result.get("peak_memory_mb", 0.0)
    perf_summary["peak_memory_mb"] = max(
        step_memory,
        step_peak_memory,
        perf_summary["peak_memory_mb"],
    )
    perf_summary["total_steps"] = total_steps


def _log_pipeline_step_completion(
    actual_step_number: int,
    description: str,
    step_result: dict[str, Any],
    step_duration: float,
    progress_tracker: Optional[PipelineProgressTracker],
    logger: logging.Logger,
) -> None:
    """Print and log the completion state for one pipeline step."""
    status_for_logging = step_result["status"]
    step_stats: dict[str, Any] = {
        "Status": status_for_logging,
        "Duration": f"{step_duration:.2f}s",
        "Memory": f"{step_result.get('peak_memory_mb', 0):.1f}MB",
        "Exit Code": step_result.get("exit_code", 0),
    }
    print_step_summary(
        actual_step_number,
        description,
        status_for_logging,
        step_duration,
        step_stats,
    )

    if STRUCTURED_LOGGING_AVAILABLE and progress_tracker:
        if "WARNING" in status_for_logging:
            from utils.logging.logging_utils import (
                log_step_warning as structured_log_step_warning,
            )

            structured_log_step_warning(
                logger,
                f"{description} completed with warnings",
                step_number=actual_step_number,
                duration=step_duration,
                status=status_for_logging,
            )
        elif status_for_logging.startswith("SUCCESS"):
            from utils.logging.logging_utils import (
                log_step_success as structured_log_step_success,
            )

            structured_log_step_success(
                logger,
                f"{description} completed",
                step_number=actual_step_number,
                duration=step_duration,
                status=status_for_logging,
            )
        else:
            from utils.logging.logging_utils import (
                log_step_error as structured_log_step_error,
            )

            structured_log_step_error(
                logger,
                f"{description} failed",
                step_number=actual_step_number,
                duration=step_duration,
                status=status_for_logging,
            )
        return

    if status_for_logging == "PARTIAL_SUCCESS" or "WARNING" in status_for_logging:
        logger.warning(
            f"⚠️ Step {actual_step_number} completed with warnings in {step_duration:.2f}s"
        )
    elif str(status_for_logging).startswith("SUCCESS"):
        logger.info(
            f"✅ Step {actual_step_number} completed successfully in {step_duration:.2f}s"
        )
    else:
        logger.error(
            f"❌ Step {actual_step_number} failed with status: {status_for_logging}"
        )


def _execute_pipeline_iteration(
    step_index: int,
    script_name: str,
    description: str,
    steps_to_execute: list[PipelineStep],
    args: PipelineArguments,
    pipeline_summary: dict[str, Any],
    visual_logger: VisualLogger,
    progress_tracker: Optional[PipelineProgressTracker],
    logger: logging.Logger,
) -> None:
    """Execute and record one selected pipeline step."""
    actual_step_number = step_index + 1
    step_start_time = time.time()
    step_start_datetime = datetime.now()

    visual_logger.print_step_header(
        actual_step_number, description, len(steps_to_execute)
    )
    _log_pipeline_step_start(
        actual_step_number,
        script_name,
        description,
        len(steps_to_execute),
        progress_tracker,
        logger,
    )

    if script_name in ("23_report.py", "24_intelligent_analysis.py"):
        _write_preliminary_pipeline_summary(pipeline_summary, args.output_dir, logger)

    step_result = execute_pipeline_step(script_name, args, logger)
    step_duration = time.time() - step_start_time
    step_end_datetime = datetime.now()

    _annotate_step_result(
        step_result,
        actual_step_number,
        script_name,
        description,
        step_start_datetime,
        step_end_datetime,
        step_duration,
    )

    has_warning = _step_has_actionable_warning(step_result)
    if step_result["status"] == "SUCCESS" and has_warning:
        step_result["status"] = "SUCCESS_WITH_WARNINGS"

    pipeline_summary["steps"].append(step_result)
    _update_performance_summary(
        pipeline_summary,
        step_result,
        script_name,
        has_warning,
        len(steps_to_execute),
    )
    _log_pipeline_step_completion(
        actual_step_number,
        description,
        step_result,
        step_duration,
        progress_tracker,
        logger,
    )


def _finalize_pipeline_summary(pipeline_summary: dict[str, Any]) -> None:
    """Set end time, duration, and final overall status."""
    end_time_dt = datetime.now()
    pipeline_summary["end_time"] = end_time_dt.isoformat()
    start_time_dt = datetime.fromisoformat(pipeline_summary["start_time"])
    pipeline_summary["total_duration_seconds"] = (
        end_time_dt - start_time_dt
    ).total_seconds()

    perf_summary = pipeline_summary["performance_summary"]
    if perf_summary["critical_failures"] > 0:
        pipeline_summary["overall_status"] = "FAILED"
    elif perf_summary["failed_steps"] > 0:
        total_steps = perf_summary["total_steps"]
        failed_ratio = perf_summary["failed_steps"] / total_steps if total_steps else 0

        if failed_ratio > 0.5:
            pipeline_summary["overall_status"] = "FAILED"
        elif failed_ratio > 0.2:
            pipeline_summary["overall_status"] = "PARTIAL_SUCCESS"
        else:
            pipeline_summary["overall_status"] = "SUCCESS_WITH_WARNINGS"
    elif perf_summary.get("warnings", 0) > 0 or any(
        step.get("status") == "SUCCESS_WITH_WARNINGS"
        for step in pipeline_summary.get("steps", [])
    ):
        pipeline_summary["overall_status"] = "SUCCESS_WITH_WARNINGS"
    else:
        pipeline_summary["overall_status"] = "SUCCESS"


def _pipeline_exit_code(overall_status: str) -> int:
    """Map final pipeline status to the public process exit contract."""
    if overall_status == "SUCCESS":
        return 0
    if overall_status in {"SUCCESS_WITH_WARNINGS", "PARTIAL_SUCCESS"}:
        return 2
    return 1


def _pipeline_summary_path(output_dir: Path) -> Path:
    """Return the canonical pipeline execution summary path."""
    return output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"


def _write_performance_dashboard(
    summary_path: Path,
    pipeline_summary: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Render the D3 performance dashboard when the template is available."""
    try:
        template_path = (
            Path(__file__).parent / "pipeline" / "performance_dashboard.template.html"
        )
        if template_path.exists():
            template_content = template_path.read_text()
            json_payload = json.dumps(pipeline_summary)
            final_html = template_content.replace("{SUMMARY_JSON}", json_payload)
            db_path = summary_path.parent / "performance_dashboard.html"
            db_path.write_text(final_html)
            logger.info(f"📊 D3 Performance Dashboard rendered to: {db_path}")
    except Exception as e:
        logger.warning(f"Could not render performance dashboard: {e}")


def _write_final_pipeline_report(
    output_dir: Path,
    summary_path: Path,
    logger: logging.Logger,
) -> None:
    """Regenerate the final pipeline report from the final summary."""
    try:
        from report.pipeline_report import generate_pipeline_report

        report_path = output_dir / "PIPELINE_REPORT.md"
        report_path.write_text(
            generate_pipeline_report(
                output_dir,
                summary_path=summary_path,
                mode="final",
            ),
            encoding="utf-8",
        )
        logger.info(f"📄 Final pipeline report written to: {report_path}")
    except Exception as e:
        logger.warning(f"Could not render final pipeline report: {e}")


def _log_pipeline_summary_counts(
    pipeline_summary: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Log a compact success/failure count after saving the final summary."""
    steps = pipeline_summary["steps"]
    successful = sum(
        1 for step in steps if step["status"] in ("SUCCESS", "SUCCESS_WITH_WARNINGS")
    )
    failed = sum(1 for step in steps if step["status"] == "FAILED")
    logger.info(f"Summary: {successful}/{len(steps)} steps successful, {failed} failed")


def _save_minimal_pipeline_summary(
    summary_path: Path,
    pipeline_summary: dict[str, Any],
    error: Exception,
    logger: logging.Logger,
) -> None:
    """Persist a reduced summary when full summary rendering fails."""
    try:
        minimal_summary: dict[str, Any] = {
            "start_time": pipeline_summary.get("start_time"),
            "end_time": datetime.now().isoformat(),
            "overall_status": "FAILED",
            "error": str(error),
            "arguments": pipeline_summary.get("arguments", {}),
            "steps_count": len(pipeline_summary.get("steps", [])),
            "performance_summary": pipeline_summary.get("performance_summary", {}),
            "steps": pipeline_summary.get("steps", []),
        }
        with open(summary_path, "w") as f:
            json.dump(minimal_summary, f, indent=4, default=str)
        logger.info("Minimal summary saved as recovery")
    except Exception as fallback_error:
        logger.error(f"Failed to save even minimal summary: {fallback_error}")


def _write_pipeline_summary_outputs(
    args: PipelineArguments,
    config_pipeline_settings: dict[Any, Any],
    pipeline_summary: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Validate, save, index, and render all final pipeline summary artifacts."""
    from pipeline.hasher import index_run

    summary_path = _pipeline_summary_path(args.output_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving pipeline summary to: {summary_path}")

    try:
        validate_pipeline_summary(pipeline_summary, logger)
        with open(summary_path, "w") as f:
            json.dump(pipeline_summary, f, indent=4, default=str)
        logger.info("Pipeline summary saved successfully")

        run_hash_value = pipeline_summary.get("run_hash")
        if isinstance(run_hash_value, str) and run_hash_value:
            index_run(
                run_hash=run_hash_value,
                summary_path=summary_path,
                config={
                    "args": args.to_dict(),
                    "pipeline": config_pipeline_settings,
                },
                file_hashes=pipeline_summary.get("file_hashes"),
            )

        _write_performance_dashboard(summary_path, pipeline_summary, logger)
        _write_final_pipeline_report(args.output_dir, summary_path, logger)
        _log_pipeline_summary_counts(pipeline_summary, logger)
    except Exception as e:
        logger.error(f"Failed to save pipeline summary: {e}")
        _save_minimal_pipeline_summary(summary_path, pipeline_summary, e, logger)


def _print_pipeline_completion(
    pipeline_summary: dict[str, Any],
    progress_tracker: Optional[PipelineProgressTracker],
    logger: logging.Logger,
) -> None:
    """Print the final visual completion summary and structured log summary."""
    total_duration = float(pipeline_summary["total_duration_seconds"] or 0.0)
    perf_summary = pipeline_summary["performance_summary"]
    completion_stats: dict[str, Any] = {
        "Total Steps": len(pipeline_summary.get("steps", [])),
        "Successful": perf_summary["successful_steps"],
        "Failed": perf_summary["failed_steps"],
        "Warnings": perf_summary["warnings"],
        "Peak Memory": f"{perf_summary['peak_memory_mb']:.1f}MB",
        "Duration": f"{total_duration:.1f}s",
    }

    success = pipeline_summary["overall_status"] == "SUCCESS"
    print_completion_summary(success, total_duration, completion_stats)

    if STRUCTURED_LOGGING_AVAILABLE:
        if progress_tracker:
            logger.info(progress_tracker.get_overall_progress())
        log_pipeline_summary(logger, pipeline_summary)
    elif success:
        logger.info(f"🎯 Pipeline completed successfully in {total_duration:.2f}s")
    else:
        logger.info(f"⚠️ Pipeline completed with issues in {total_duration:.2f}s")


def _handle_pipeline_failure(
    error: Exception,
    args: PipelineArguments,
    pipeline_summary: dict[str, Any],
    logger: logging.Logger,
) -> int:
    """Record a failed summary if the pipeline crashes outside a step."""
    end_time_dt = datetime.now()
    pipeline_summary["end_time"] = end_time_dt.isoformat()
    pipeline_summary["overall_status"] = "FAILED"
    start_time_dt = datetime.fromisoformat(pipeline_summary["start_time"])
    pipeline_summary["total_duration_seconds"] = (
        end_time_dt - start_time_dt
    ).total_seconds()

    summary_path = _pipeline_summary_path(args.output_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(pipeline_summary, f, indent=4)

    log_step_error(logger, f"Pipeline failed: {str(error)}")
    return 1


def main(
    override_args: Optional[PipelineArguments] = None,
    override_config: Optional[Dict[str, Any]] = None,
) -> int:
    """Main pipeline orchestration function."""
    (
        args,
        config_pipeline_settings,
        steps_to_execute,
        pipeline_summary,
        visual_logger,
        correlation_id,
        logger,
    ) = _prepare_pipeline_context(override_args, override_config)

    if getattr(args, "autonomous", False):
        from pipeline.autonomous import run_autonomous_proposal_loop

        report = run_autonomous_proposal_loop(args.target_dir, args.output_dir)
        logger.info(
            "Autonomous proposal loop wrote %d candidate(s) under %s/autonomous",
            report.get("candidate_count", 0),
            args.output_dir,
        )
        return 0

    progress_tracker: Optional[PipelineProgressTracker] = None
    try:
        progress_tracker = _start_pipeline_run(
            args,
            steps_to_execute,
            pipeline_summary,
            visual_logger,
            correlation_id,
            logger,
        )

        for step_index, (script_name, description) in enumerate(steps_to_execute):
            _execute_pipeline_iteration(
                step_index,
                script_name,
                description,
                steps_to_execute,
                args,
                pipeline_summary,
                visual_logger,
                progress_tracker,
                logger,
            )

        _finalize_pipeline_summary(pipeline_summary)
        _write_pipeline_summary_outputs(
            args, config_pipeline_settings, pipeline_summary, logger
        )
        _print_pipeline_completion(pipeline_summary, progress_tracker, logger)
        return _pipeline_exit_code(pipeline_summary["overall_status"])

    except Exception as e:
        return _handle_pipeline_failure(e, args, pipeline_summary, logger)


def execute_pipeline_step(
    script_name: str, args: PipelineArguments, logger: Any
) -> Dict[str, Any]:
    """Execute a single pipeline step with comprehensive monitoring."""
    import os

    # Initialize performance tracking
    start_memory = get_current_memory_usage()
    peak_memory = start_memory

    step_result: dict[str, Any] = {
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
    }

    try:
        # Load config.yaml once — used for both skip_steps (prereq validation) and testing_matrix
        config_skip_steps: list[Any] = []
        testing_matrix: dict[Any, Any] = {}
        input_config_path = Path("input/config.yaml")
        if input_config_path.exists():
            try:
                import yaml

                with open(input_config_path, "r") as f:
                    _full_cfg = yaml.safe_load(f) or {}
                config_skip_steps = _full_cfg.get("pipeline", {}).get("skip_steps", [])
                testing_matrix = _full_cfg.get("testing_matrix", {})
            except (ImportError, OSError, ValueError, Exception) as e:
                logger.debug(f"Could not parse input/config.yaml: {e}")

        # Validate step prerequisites
        prereq_result = validate_step_prerequisites(
            script_name, args, logger, skip_steps=config_skip_steps
        )
        step_result["prerequisite_check"] = prereq_result["passed"]
        step_result["dependency_warnings"] = prereq_result.get("warnings", [])

        # Log prerequisite warnings if any
        if prereq_result.get("warnings"):
            for warning in prereq_result["warnings"]:
                logger.warning(f"Prerequisite warning for {script_name}: {warning}")

        # Get script path
        script_path = Path(__file__).parent / script_name

        # Get virtual environment Python path
        project_root = Path(__file__).parent.parent
        venv_python = project_root / ".venv" / "bin" / "python"

        # Use virtual environment Python if available, otherwise fall back to system Python
        python_executable = str(venv_python) if venv_python.exists() else sys.executable

        # Extract step number
        step_num_match = re.match(r"^(\d+)_", script_name)
        step_num = int(step_num_match.group(1)) if step_num_match else -1

        matrix_enabled = testing_matrix.get("enabled", False)
        target_folders: list[Any] = []

        # Check global_steps: if a global step (0, 1, 2) is disabled, skip it entirely
        if matrix_enabled:
            global_steps = testing_matrix.get("global_steps", {})
            script_stem = script_name.replace(".py", "")
            if script_stem in global_steps and not global_steps[script_stem]:
                logger.info(
                    f"⏭️ Skipping {script_name}: disabled in testing_matrix.global_steps"
                )
                step_result["status"] = "SKIPPED"
                step_result["exit_code"] = 0
                step_result["stdout"] = (
                    f"Skipped by global_steps config (testing_matrix.global_steps.{script_stem}: false)\n"
                )
                return step_result

        # Only apply folder-matrix logic if enabled and the step is a processing step (>= 3)
        if matrix_enabled and step_num >= 3:
            base_target_dir = args.target_dir
            if base_target_dir.exists() and base_target_dir.is_dir():
                folders_config = testing_matrix.get("folders", {})
                default_steps = testing_matrix.get("default_steps", [])

                # Check all subdirectories in the base target directory
                for item in base_target_dir.iterdir():
                    if item.is_dir() and item.name != "archived_gnn_files":
                        # Determine allowed steps for this folder
                        allowed_steps = folders_config.get(item.name, default_steps)
                        if step_num in allowed_steps:
                            target_folders.append(item)

        from pipeline.step_timeouts import get_step_timeout
        from utils.argument_utils import build_step_command_args
        from utils.execution_utils import execute_command_streaming

        # Prepare environment
        _env = os.environ.copy()
        _env.setdefault("PYTHONUNBUFFERED", "1")
        comprehensive_requested = any("--comprehensive" in str(arg) for arg in sys.argv)
        step_timeout_seconds = get_step_timeout(
            script_name, comprehensive=comprehensive_requested
        )

        if matrix_enabled and step_num >= 2 and target_folders:
            # MATRIX MODE: We found specific folders to run for this step
            if args.verbose:
                logger.info(
                    f"Testing matrix enabled. Running step {step_num} on {len(target_folders)} specific folders: {[f.name for f in target_folders]}"
                )

            combined_stdout = ""
            combined_stderr = ""
            worst_exit_code = 0

            for folder in target_folders:
                if args.verbose:
                    logger.info(f"  -> Executing for folder: {folder.name}")

                # Creating a modified args copy to point to the specific subfolder
                import copy

                folder_args = copy.copy(args)
                folder_args.target_dir = folder

                cmd = build_step_command_args(
                    script_name.replace(".py", ""),
                    folder_args,
                    python_executable,
                    script_path,
                )

                # We do not use print_stdout=True for every subfolder iteration to avoid extreme terminal spam,
                # but we will print if requested verbose globally
                if args.verbose:
                    logger.info(f"  -> CMD ACTUALLY IS: {' '.join(cmd)}")

                result = execute_command_streaming(
                    cmd,
                    cwd=project_root,
                    env=_env,
                    timeout=step_timeout_seconds,
                    print_stdout=args.verbose,
                    print_stderr=True,
                    capture_output=True,
                )

                combined_stdout += (
                    f"\n--- Output for {folder.name} ---\n{result.get('stdout', '')}\n"
                )
                if result.get("stderr"):
                    combined_stderr += f"\n--- Stderr for {folder.name} ---\n{result.get('stderr', '')}\n"

                exit_code = result.get("exit_code", -1)
                if exit_code != 0:
                    worst_exit_code = exit_code
                    logger.warning(
                        f"  -> Folder {folder.name} execution returned code {exit_code}"
                    )

            end_memory = get_current_memory_usage()
            peak_memory = max(peak_memory, end_memory)

            step_result["stdout"] = combined_stdout
            step_result["stderr"] = combined_stderr
            step_result["exit_code"] = worst_exit_code
            step_result["memory_usage_mb"] = end_memory
            step_result["peak_memory_mb"] = peak_memory
            step_result["memory_delta_mb"] = end_memory - start_memory

        else:
            # STANDARD MODE
            cmd = build_step_command_args(
                script_name.replace(".py", ""), args, python_executable, script_path
            )

            if args.verbose:
                logger.info(f"Executing command: {' '.join(cmd)}")

            result = execute_command_streaming(
                cmd,
                cwd=project_root,
                env=_env,
                timeout=step_timeout_seconds,
                print_stdout=True,
                print_stderr=True,
                capture_output=True,
            )

            end_memory = get_current_memory_usage()
            peak_memory = max(peak_memory, end_memory)

            step_result["stdout"] = result.get("stdout", "")
            step_result["stderr"] = result.get("stderr", "")
            step_result["exit_code"] = result.get("exit_code", -1)
            step_result["memory_usage_mb"] = end_memory
            step_result["peak_memory_mb"] = peak_memory
            step_result["memory_delta_mb"] = end_memory - start_memory

            if args.verbose:
                logger.info(
                    "Command completed with exit code: " + str(step_result["exit_code"])
                )

        # Determine status
        if step_result["exit_code"] == 0:
            step_result["status"] = "SUCCESS"
            # Check for any dependency warnings that might affect downstream steps
            if step_result["dependency_warnings"]:
                step_result["status"] = "SUCCESS_WITH_WARNINGS"
        else:
            # Respect the child process exit code to avoid masking failures
            step_result["status"] = "FAILED"
            # Log detailed failure information
            logger.error(
                f"Step {script_name} failed with exit code {step_result['exit_code']}"
            )
            if step_result["stderr"]:
                logger.error(
                    f"Error output: {step_result['stderr'][:500]}..."
                )  # Limit to first 500 chars

        # Steps determine their own output directories via get_output_dir_for_script

        # Add recovery status to result
        if not step_result.get("recoverable"):
            step_result["recoverable"] = False

        return step_result

    except Exception as e:
        logger.error(f"Exception in execute_pipeline_step for {script_name}: {e}")
        step_result["status"] = "FAILED"
        step_result["exit_code"] = -1
        step_result["stderr"] = str(e)
        return step_result


def parse_step_list(step_input: Any) -> List[int]:
    """Parse step input (string or list) into list of integers."""
    if step_input is None:
        return []
    if isinstance(step_input, list):
        return [int(s) for s in step_input if str(s).isdigit() or isinstance(s, int)]
    if isinstance(step_input, str):
        try:
            return [int(s.strip()) for s in step_input.split(",") if s.strip()]
        except ValueError:
            _module_logger.debug(
                "Could not parse step list '%s' as comma-separated integers", step_input
            )
            return []
    return []


def get_environment_info() -> Dict[str, Any]:
    """Get environment information."""
    import os

    info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "working_directory": str(Path.cwd()),
        "user": os.getenv("USER", "unknown"),
    }

    try:
        import psutil

        info["memory_total_gb"] = f"{psutil.virtual_memory().total / 1024**3:.1f}"
    except ImportError:
        info["memory_total_gb"] = "unavailable (psutil not installed)"

    try:
        import psutil

        info["disk_free_gb"] = f"{psutil.disk_usage('/').free / 1024**3:.1f}"
    except ImportError:
        info["disk_free_gb"] = "unavailable (psutil not installed)"

    return info


def validate_pipeline_summary(summary: dict, logger: Any) -> None:
    """
    Validate pipeline summary structure and data integrity.

    Args:
        summary: Pipeline summary dictionary to validate
        logger: Logger instance for validation messages
    """
    required_fields: list[Any] = [
        "start_time",
        "arguments",
        "steps",
        "end_time",
        "overall_status",
        "total_duration_seconds",
        "environment_info",
        "performance_summary",
    ]

    # Check required fields
    for field in required_fields:
        if field not in summary:
            logger.warning(f"Pipeline summary missing required field: {field}")
        elif summary[field] is None and field not in ["end_time"]:
            logger.warning(f"Pipeline summary field '{field}' is None")

    # Validate steps structure
    steps = summary.get("steps", [])
    if not isinstance(steps, list):
        logger.error("Pipeline summary 'steps' should be a list")
        return

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            logger.error(f"Step {i} should be a dictionary")
            continue

        step_required: list[Any] = [
            "status",
            "step_number",
            "script_name",
            "description",
        ]
        for field in step_required:
            if field not in step:
                logger.warning(f"Step {i} missing required field: {field}")

        # Validate step status values
        if "status" in step:
            valid_statuses: list[Any] = [
                "SUCCESS",
                "SUCCESS_WITH_WARNINGS",
                "PARTIAL_SUCCESS",
                "FAILED",
                "TIMEOUT",
                "SKIPPED",
            ]
            if step["status"] not in valid_statuses:
                logger.warning(f"Step {i} has invalid status: {step['status']}")

        # Validate numeric fields
        numeric_fields: list[Any] = [
            "step_number",
            "exit_code",
            "retry_count",
            "duration_seconds",
            "memory_usage_mb",
            "peak_memory_mb",
            "memory_delta_mb",
        ]
        for field in numeric_fields:
            if field in step and not isinstance(step[field], (int, float)):
                logger.warning(
                    f"Step {i} field '{field}' should be numeric, got {type(step[field])}"
                )

    # Validate performance summary
    perf = summary.get("performance_summary", {})
    if not isinstance(perf, dict):
        logger.error("Performance summary should be a dictionary")
        return

    # Validate numeric fields
    numeric_fields = [
        "peak_memory_mb",
        "total_steps",
        "failed_steps",
        "critical_failures",
        "successful_steps",
        "warnings",
    ]
    for field in numeric_fields:
        if field in perf:
            if not isinstance(perf[field], (int, float)):
                logger.warning(
                    f"Performance summary field '{field}' should be numeric, got {type(perf[field])}"
                )

    # Validate overall status
    if "overall_status" in summary:
        valid_statuses = [
            "SUCCESS",
            "SUCCESS_WITH_WARNINGS",
            "PARTIAL_SUCCESS",
            "FAILED",
        ]
        if summary["overall_status"] not in valid_statuses:
            logger.warning(f"Invalid overall status: {summary['overall_status']}")

    # Validate timing consistency
    if (
        "start_time" in summary
        and "end_time" in summary
        and "total_duration_seconds" in summary
    ):
        try:
            start_dt = datetime.fromisoformat(summary["start_time"])
            end_dt = datetime.fromisoformat(summary["end_time"])
            calculated_duration = (end_dt - start_dt).total_seconds()
            reported_duration = summary["total_duration_seconds"]

            # Allow for small timing differences due to processing
            if abs(calculated_duration - reported_duration) > 1.0:
                logger.warning(
                    f"Timing inconsistency: calculated {calculated_duration:.2f}s vs reported {reported_duration:.2f}s"
                )
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not validate timing: {e}")


# All pipeline utility functions have been moved to appropriate utils modules:
# - validate_step_prerequisites, validate_pipeline_step_sequence → utils/pipeline_validator.py
# - get_current_memory_usage → utils/resource_manager.py
# - attempt_step_recovery and recovery functions → utils/error_recovery.py
# - generate_pipeline_health_report → utils/pipeline_monitor.py
# - generate_execution_plan → utils/pipeline_planner.py


if __name__ == "__main__":
    raise SystemExit(main())
