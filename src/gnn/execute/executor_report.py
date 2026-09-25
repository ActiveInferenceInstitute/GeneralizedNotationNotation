"""
Batch execution and report helpers for the GNN executor: framework
directory/result bookkeeping, dependency checks, per-framework dispatch,
markdown/JSON summaries, and the rendered-simulators entry point.

Split from ``gnn.execute.executor`` (W7-A band split); the moved ranges are
byte-verbatim. ``_framework_specs`` and the ``log_step_*`` helpers resolve
through ``gnn.execute.executor`` at call time so executor-namespace
monkeypatches stay observable.
"""

from __future__ import annotations

import json
import logging
import subprocess  # nosec B404
import time
from pathlib import Path
from typing import Any, Optional, cast

from gnn.execute.executor_specs import FRAMEWORK_DIR_NAMES, ExecutorFrameworkSpec


def _framework_specs(*args: Any, **kwargs: Any) -> Any:
    """Resolve through ``gnn.execute.executor`` (call-time indirection)."""
    from gnn.execute import executor

    return executor._framework_specs(*args, **kwargs)


def _log_step(name: str, *args: Any, **kwargs: Any) -> Any:
    """Resolve a ``log_step_*`` helper through ``gnn.execute.executor``."""
    from gnn.execute import executor

    return getattr(executor, name)(*args, **kwargs)


def log_step_error(*args: Any, **kwargs: Any) -> Any:
    """See ``gnn.execute.executor.log_step_error`` (call-time indirection)."""
    return _log_step("log_step_error", *args, **kwargs)


def log_step_start(*args: Any, **kwargs: Any) -> Any:
    """See ``gnn.execute.executor.log_step_start`` (call-time indirection)."""
    return _log_step("log_step_start", *args, **kwargs)


def log_step_success(*args: Any, **kwargs: Any) -> Any:
    """See ``gnn.execute.executor.log_step_success`` (call-time indirection)."""
    return _log_step("log_step_success", *args, **kwargs)


def log_step_warning(*args: Any, **kwargs: Any) -> Any:
    """See ``gnn.execute.executor.log_step_warning`` (call-time indirection)."""
    return _log_step("log_step_warning", *args, **kwargs)


from gnn.pipeline.config import get_output_dir_for_script
from gnn.utils import performance_tracker


def _create_framework_dirs(
    execution_output_dir: Path, logger: logging.Logger
) -> dict[str, Path]:
    """Create and return framework-specific execution directories."""
    framework_dirs = {name: execution_output_dir / name for name in FRAMEWORK_DIR_NAMES}
    for framework_dir in framework_dirs.values():
        framework_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created framework directory: {framework_dir}")
    return framework_dirs


def _initialize_execution_results(
    target_dir: Path, framework_dirs: dict[str, Path]
) -> dict[str, Any]:
    """Build the common execution summary envelope.

    The per-framework ``*_executions`` lists are derived from the
    :func:`_framework_specs` registry so adding a framework is a one-line
    change (the registry is the single source of truth for both the result
    keys and the dispatch wiring).
    """
    result: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_directory": str(target_dir),
        "framework_execution_dirs": {k: str(v) for k, v in framework_dirs.items()},
        "total_successes": 0,
        "total_failures": 0,
        "dependency_issues": [],
        "syntax_errors": [],
        "execution_details": {},
    }
    for spec in _framework_specs():
        result[spec.result_key] = []
    return result


def _check_python_dependencies(
    execution_results: dict[str, Any], logger: logging.Logger
) -> None:
    """Record missing Python dependencies before runner execution starts."""
    missing_python_deps: list[str] = []
    for dep in ["numpy", "pymdp", "flax", "jax", "optax"]:
        try:
            __import__(dep)
            logger.debug(f"✅ Python dependency available: {dep}")
        except ImportError:
            missing_python_deps.append(dep)
            logger.warning(f"⚠️ Python dependency missing: {dep}")

    if missing_python_deps:
        execution_results["dependency_issues"].append(
            f"Missing Python dependencies: {', '.join(missing_python_deps)}"
        )


def _check_julia_availability(
    execution_results: dict[str, Any], logger: logging.Logger
) -> None:
    """Record Julia availability for Julia-backed execution frameworks."""
    try:
        result = subprocess.run(
            ["julia", "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )  # nosec B607 B603
        if result.returncode == 0:
            logger.info(f"✅ Julia available: {result.stdout.strip()}")
        else:
            logger.warning("⚠️ Julia not available or not working properly")
            execution_results["dependency_issues"].append("Julia not available")
    except FileNotFoundError:
        logger.warning("⚠️ Julia not found in PATH")
        execution_results["dependency_issues"].append("Julia not found in PATH")


def _validate_pymdp_script_syntax(
    target_dir: Path, execution_results: dict[str, Any], logger: logging.Logger
) -> None:
    """Compile rendered PyMDP scripts so syntax errors appear in the summary."""
    pymdp_dir = target_dir / "pymdp"
    if not pymdp_dir.exists():
        return

    for script in pymdp_dir.glob("*.py"):
        try:
            with open(script, "r") as f:
                compile(f.read(), script.name, "exec")
            logger.debug(f"✅ PyMDP script syntax valid: {script.name}")
        except SyntaxError as e:
            logger.warning(f"⚠️ PyMDP script syntax error in {script.name}: {e}")
            execution_results["syntax_errors"].append(f"PyMDP: {script.name} - {e}")


def _append_framework_result(
    execution_results: dict[str, Any],
    spec: ExecutorFrameworkSpec,
    status: str,
    message: str,
    output_dir: Path,
) -> None:
    """Append a normalized framework execution record."""
    execution_results[spec.result_key].append(
        {
            "status": status,
            "message": message,
            "output_dir": str(output_dir),
        }
    )


def _execute_framework_spec(
    spec: ExecutorFrameworkSpec,
    target_dir: Path,
    framework_dirs: dict[str, Path],
    execution_results: dict[str, Any],
    logger: logging.Logger,
    recursive: bool,
    verbose: bool,
    timeout: Optional[int] = None,
) -> None:
    """Execute one framework runner and record its status."""
    output_dir = framework_dirs[spec.framework_dir_key]
    if not spec.available:
        logger.info(spec.unavailable_log)
        _append_framework_result(
            execution_results, spec, "SKIPPED", spec.unavailable_message, output_dir
        )
        return

    try:
        with performance_tracker.track_operation(spec.operation_name):
            logger.info(spec.start_message)
            if spec.framework_dir_key == "pymdp":
                _validate_pymdp_script_syntax(target_dir, execution_results, logger)

            success = spec.runner(
                rendered_simulators_dir=target_dir,
                execution_output_dir=output_dir,
                recursive_search=recursive,
                verbose=verbose,
                timeout=timeout,
            )

            if success:
                execution_results["total_successes"] += 1
                _append_framework_result(
                    execution_results, spec, "SUCCESS", spec.success_message, output_dir
                )
                log_step_success(logger, spec.success_log)
            else:
                execution_results["total_failures"] += 1
                _append_framework_result(
                    execution_results, spec, "FAILED", spec.failure_message, output_dir
                )
                log_step_warning(logger, spec.failure_message)
    except Exception as e:
        execution_results["total_failures"] += 1
        _append_framework_result(execution_results, spec, "ERROR", str(e), output_dir)
        log_step_warning(logger, f"{spec.warning_log_prefix}: {e}")


def _execute_configured_frameworks(
    target_dir: Path,
    framework_dirs: dict[str, Path],
    execution_results: dict[str, Any],
    logger: logging.Logger,
    recursive: bool,
    verbose: bool,
    timeout: Optional[int] = None,
) -> None:
    """Execute every supported framework according to current availability."""
    for spec in _framework_specs():
        _execute_framework_spec(
            spec,
            target_dir,
            framework_dirs,
            execution_results,
            logger,
            recursive,
            verbose,
            timeout,
        )


def _write_framework_report_section(
    file_obj: Any,
    title: str,
    executions: list[dict[str, Any]],
    default_script: str,
    include_type: bool = False,
) -> None:
    """Write one framework subsection to the markdown execution report."""
    if not executions:
        return

    file_obj.write(f"## {title}\n\n")
    for exec_info in executions:
        status_icon = "✅" if exec_info.get("status") == "SUCCESS" else "❌"
        type_text = f" ({exec_info.get('type', 'analysis')})" if include_type else ""
        file_obj.write(
            f"- {status_icon} **{exec_info.get('script', default_script)}**{type_text}: {exec_info.get('status', 'Unknown')}\n"
        )
        file_obj.write(f"  - {exec_info.get('message', 'No message')}\n")
        file_obj.write(f"  - Output Directory: {exec_info.get('output_dir', 'N/A')}\n")
        if "scripts_processed" in exec_info:
            file_obj.write(f"  - Scripts processed: {exec_info['scripts_processed']}\n")
    file_obj.write("\n")


def _write_execution_report(
    report_file: Path, execution_results: dict[str, Any]
) -> None:
    """Write the enhanced markdown execution report."""
    with open(report_file, "w") as f:
        f.write("# Enhanced Execution Results Report\n\n")
        f.write(f"**Generated:** {execution_results['timestamp']}\n")
        f.write(f"**Target Directory:** {execution_results['target_directory']}\n")
        f.write(f"**Total Successes:** {execution_results['total_successes']}\n")
        f.write(f"**Total Failures:** {execution_results['total_failures']}\n\n")

        f.write("## Framework-Specific Output Directories\n\n")
        for framework, framework_dir in execution_results[
            "framework_execution_dirs"
        ].items():
            f.write(f"- **{framework.upper()}**: {framework_dir}\n")
        f.write("\n")

        if execution_results["dependency_issues"]:
            f.write("## Dependency Issues\n\n")
            for issue in execution_results["dependency_issues"]:
                f.write(f"- ⚠️ {issue}\n")
            f.write("\n")

        if execution_results["syntax_errors"]:
            f.write("## Syntax Errors\n\n")
            for error in execution_results["syntax_errors"]:
                f.write(f"- ❌ {error}\n")
            f.write("\n")

        _write_framework_report_section(
            f,
            "PyMDP Executions",
            execution_results["pymdp_executions"],
            "PyMDP Scripts",
        )
        _write_framework_report_section(
            f,
            "RxInfer Executions",
            execution_results["rxinfer_executions"],
            "RxInfer Scripts",
        )
        _write_framework_report_section(
            f,
            "DisCoPy Analyses",
            execution_results["discopy_executions"],
            "DisCoPy Analysis",
            include_type=True,
        )
        _write_framework_report_section(
            f,
            "ActiveInference.jl Analyses",
            execution_results["activeinference_executions"],
            "ActiveInference.jl Scripts",
        )
        _write_framework_report_section(
            f, "JAX Executions", execution_results["jax_executions"], "JAX Scripts"
        )
        _write_framework_report_section(
            f,
            "NumPyro Executions",
            execution_results["numpyro_executions"],
            "NumPyro Scripts",
        )
        _write_framework_report_section(
            f,
            "PyTorch Executions",
            execution_results["pytorch_executions"],
            "PyTorch Scripts",
        )
        _write_framework_report_section(
            f,
            "ngc-learn Executions",
            execution_results["ngclearn_executions"],
            "ngc-learn Scripts",
        )
        _write_framework_report_section(
            f,
            "Lean verification",
            execution_results["lean_executions"],
            "Lean Documents",
        )
        _write_framework_report_section(
            f,
            "Stan Executions",
            execution_results["stan_executions"],
            "Stan Drivers",
        )
        _write_framework_report_section(
            f,
            "bnlearn Executions",
            execution_results["bnlearn_executions"],
            "bnlearn Scripts",
        )

        f.write("## Recommendations\n\n")
        if execution_results["dependency_issues"]:
            f.write("### Install Missing Dependencies\n\n")
            for issue in execution_results["dependency_issues"]:
                if "Python dependencies" in issue:
                    f.write(
                        "- Install missing Python packages: `uv pip install <package_name>` or add to pyproject and run `uv sync`\n"
                    )
                elif "Julia" in issue:
                    f.write("- Install Julia from https://julialang.org/downloads/\n")
            f.write("\n")

        if execution_results["syntax_errors"]:
            f.write("### Fix Syntax Errors\n\n")
            f.write("- Review and fix syntax errors in rendered scripts\n")
            f.write("- Check for stray characters or malformed code\n")
            f.write(
                "- Re-run the rendering step (src/gnn/11_render.py) to regenerate scripts\n\n"
            )


def _write_execution_artifacts(
    execution_output_dir: Path, execution_results: dict[str, Any]
) -> None:
    """Write JSON and markdown execution summaries."""
    summaries_dir = execution_output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    with open(summaries_dir / "execution_summary.json", "w") as f:
        json.dump(execution_results, f, indent=2)
    _write_execution_report(summaries_dir / "execution_report.md", execution_results)


def _count_framework_execution_records(execution_results: dict[str, Any]) -> int:
    """Count framework result records across all supported backends."""
    return sum(len(execution_results[spec.result_key]) for spec in _framework_specs())


def _log_execution_outcome(
    execution_results: dict[str, Any], logger: logging.Logger
) -> bool:
    """Log aggregate execution outcome and return success status."""
    total_executions = _count_framework_execution_records(execution_results)
    if total_executions == 0:
        log_step_warning(
            logger, "No simulator scripts or outputs found to execute/analyze"
        )
        return True

    success_rate = execution_results["total_successes"] / total_executions * 100
    log_step_success(
        logger,
        f"Execution completed with framework-specific organization. Success rate: {success_rate:.1f}% ({execution_results['total_successes']}/{total_executions})",
    )

    if execution_results["dependency_issues"]:
        logger.warning(
            f"⚠️ Dependency issues found: {len(execution_results['dependency_issues'])}"
        )
    if execution_results["syntax_errors"]:
        logger.warning(
            f"⚠️ Syntax errors found: {len(execution_results['syntax_errors'])}"
        )

    return cast("bool", execution_results["total_failures"] == 0)


def execute_rendered_simulators(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Execute rendered simulator scripts with enhanced error handling and dependency checking.
    Framework outputs are organized in separate subdirectories.

    Args:
        target_dir: Directory containing rendered simulator scripts
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional execution options

    Returns:
        True if execution succeeded, False otherwise
    """
    timeout: Optional[int] = kwargs.pop("timeout", None)
    log_step_start(
        logger,
        "Executing rendered simulator scripts with framework-specific organization",
    )

    execution_output_dir = get_output_dir_for_script("12_execute.py", output_dir)
    execution_output_dir.mkdir(parents=True, exist_ok=True)
    framework_dirs = _create_framework_dirs(execution_output_dir, logger)

    try:
        execution_results = _initialize_execution_results(target_dir, framework_dirs)
        logger.info("🔍 Pre-execution validation and dependency checking...")
        _check_python_dependencies(execution_results, logger)
        _check_julia_availability(execution_results, logger)

        _execute_configured_frameworks(
            target_dir,
            framework_dirs,
            execution_results,
            logger,
            recursive,
            verbose,
            timeout,
        )
        _write_execution_artifacts(execution_output_dir, execution_results)
        return _log_execution_outcome(execution_results, logger)

    except Exception as e:
        log_step_error(logger, f"Execution failed: {e}")
        return False
