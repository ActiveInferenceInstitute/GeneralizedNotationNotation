#!/usr/bin/env python3
"""
Execute Processor module for GNN Processing Pipeline.

This module provides execute processing capabilities for rendered implementations.
"""

import copy
import hashlib
import json
import logging
import os
import shutil
import subprocess  # nosec B404
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

from utils.logging.logging_utils import (
    log_step_error,
    log_step_start,
    log_step_success,
    log_step_warning,
)

logger = logging.getLogger(__name__)

# File suffixes Step 12 can execute; companion artifacts (``.stan``, ``.toml``,
# ``.json``) listed in a render summary are not scripts.
_EXECUTABLE_SUFFIXES = frozenset({".py", ".jl"})

ExecutionFrameworkName = Literal[
    "pymdp",
    "rxinfer",
    "jax",
    "discopy",
    "activeinference_jl",
    "pytorch",
    "numpyro",
    "stan",
]


@dataclass(frozen=True)
class ScriptExecutionContext:
    """Normalized execution metadata for one rendered script."""

    script_path: Path
    script_name: str
    framework: str
    model_name: str
    executor: str


def _julia_project_for_framework(framework: str) -> Optional[Path]:
    """Return the committed Julia environment for a supported framework."""
    execute_dir = Path(__file__).resolve().parent
    projects = {
        "rxinfer": execute_dir / "rxinfer",
        "activeinference_jl": execute_dir / "activeinference_jl",
    }
    return projects.get(framework)


def _build_script_execution_command(
    context: ScriptExecutionContext, sandbox_prefix: List[str]
) -> List[str]:
    """Build an explicit, reproducible command for a rendered script."""
    command = list(sandbox_prefix)
    if context.executor == "julia":
        project_dir = _julia_project_for_framework(context.framework)
        if project_dir is None:
            raise ValueError(
                f"No committed Julia project is registered for {context.framework}"
            )
        command.extend(
            [
                context.executor,
                "--startup-file=no",
                f"--project={project_dir}",
                context.script_path.name,
            ]
        )
        return command
    command.extend([context.executor, context.script_path.name])
    return command


def check_julia_dependencies(
    verbose: bool,
    log: Optional[logging.Logger] = None,
    frameworks: Optional[List[str]] = None,
) -> bool:
    """Check if required Julia packages are available.

    Args:
        verbose: Enable verbose logging.
        log: Optional logger instance; defaults to module logger if not provided.

    Returns:
        True if dependencies ok, False otherwise.
    """
    if log is None:
        log = logger
    try:
        # check basic julia availability
        subprocess.run(
            ["julia", "--version"], capture_output=True, check=True, timeout=10
        )  # nosec B607 B603

        requested = set(frameworks or ["rxinfer", "activeinference_jl"])

        # Each Julia-backed framework ships its own committed project
        # environment (Project.toml + Manifest.toml), so the package check must
        # run against that environment's ``--project``. A bare ``julia -e
        # "using ..."`` resolves against the global depot and always fails on a
        # clean machine, which previously skipped every Julia script.
        framework_projects = {
            "rxinfer": (
                _julia_project_for_framework("rxinfer"),
                ["JSON", "Distributions", "StatsBase", "RxInfer"],
            ),
            "activeinference_jl": (
                _julia_project_for_framework("activeinference_jl"),
                ["JSON", "Distributions", "StatsBase", "ActiveInference"],
            ),
        }

        for framework in sorted(requested):
            entry = framework_projects.get(framework)
            if entry is None:
                log.warning(f"Unknown Julia framework '{framework}'; skipping check")
                continue
            project_dir, packages = entry
            if project_dir is None:
                return False
            using_clause = ", ".join(packages)
            check_script = f"using {using_clause}"
            result = subprocess.run(  # nosec B607 B603
                [
                    "julia",
                    "--startup-file=no",
                    f"--project={project_dir}",
                    "-e",
                    check_script,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                if verbose:
                    log.warning(
                        f"Julia package check failed for {framework}: {result.stderr}"
                    )
                return False

        return True
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False


def determine_script_framework(
    script_path: Path, render_output_dir: Path, framework_dirs: Dict[str, str]
) -> str:
    """
    Determine the framework for a script based on its directory path.

    Args:
        script_path: Path to the script
        render_output_dir: Base render output directory
        framework_dirs: Mapping of directory names to framework names

    Returns:
        Framework name or 'unknown'
    """
    try:
        # Get relative path from render output directory
        relative_path = script_path.relative_to(render_output_dir)

        # Render outputs use model/framework/script.ext. Match framework
        # directories exactly so model names like "bnlearn_causal_model" do
        # not override the actual framework directory.
        for part in relative_path.parts[:-1]:
            if part.lower() in framework_dirs:
                return framework_dirs[part.lower()]

        script_name = relative_path.name.lower()
        for framework_name in framework_dirs.values():
            if script_name.endswith(f"_{framework_name}.py") or script_name.endswith(
                f"_{framework_name}.jl"
            ):
                return framework_name

        # Default recovery
        return "unknown"

    except Exception as e:
        logging.getLogger(__name__).debug(
            f"Error determining framework for script: {e}"
        )
        return "unknown"


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


def _load_rxinfer_execution_metadata_sidecar(script_path: Path) -> Dict[str, Any]:
    """Load declared RxInfer execution metadata from JSON sidecar artifacts."""
    candidates = [
        script_path.with_suffix(".metadata.json"),
        script_path.with_name(f"{script_path.stem}_metadata.json"),
    ]
    seen: set[Path] = set()
    for metadata_path in candidates:
        if metadata_path in seen or not metadata_path.exists():
            continue
        seen.add(metadata_path)
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("schema") != "gnn_rxinfer_execution_metadata_v1":
            continue
        if data.get("script_sha256") != _sha256_file(script_path):
            continue
        if "agent_count" not in data and "topology" not in data:
            continue
        topology = data.get("topology")
        if not isinstance(topology, dict):
            topology = {}
        topology.setdefault("source", str(metadata_path))
        data["topology"] = topology
        data["agent_count"] = int(data.get("agent_count") or 0)
        data["metadata_provenance"] = data.get(
            "metadata_provenance", "rendered_rxinfer_sidecar"
        )
        data["metadata_verification"] = "script_sha256_match"
        return data
    return {}


def _load_rxinfer_execution_metadata_from_script(script_path: Path) -> Dict[str, Any]:
    """Load agent population metadata from rendered RxInfer metadata artifacts."""
    if not script_path.exists():
        return {}
    sidecar_metadata = _load_rxinfer_execution_metadata_sidecar(script_path)
    if sidecar_metadata:
        return sidecar_metadata
    toml_candidates = [script_path.with_suffix(".toml")]
    seen_toml: set[Path] = set()
    for toml_path in toml_candidates:
        if toml_path in seen_toml or not toml_path.exists():
            continue
        seen_toml.add(toml_path)
        try:
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        agents = data.get("agents", [])
        model = data.get("model", {})
        agent_count = (
            len(agents) if isinstance(agents, list) else model.get("nr_agents")
        )
        agent_ids = [
            agent.get("id")
            for agent in agents
            if isinstance(agent, dict) and "id" in agent
        ]
        topology_data = data.get("topology", {})
        topology: Dict[str, Any] = {
            "type": "agent_population",
            "agent_ids": agent_ids,
            "source": str(toml_path),
        }
        if isinstance(topology_data, dict):
            topology["type"] = str(topology_data.get("type") or topology["type"])
            topology["agent_ids"] = topology_data.get("agent_ids") or agent_ids
            if "edges" in topology_data:
                topology["edges"] = topology_data["edges"]
            if "clusters" in topology_data:
                topology["clusters"] = topology_data["clusters"]
            if "message_passing" in topology_data:
                topology["message_passing"] = topology_data["message_passing"]
        return {
            "agent_count": int(agent_count or 0),
            "topology": topology,
            "metadata_provenance": "rxinfer_toml_sidecar",
            "metadata_verification": "exact_stem_toml",
        }
    return {}


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest for an executable script."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_python_framework_dependency_available(
    framework: str, executor: str, logger: Any
) -> bool:
    """Return True if the framework's required Python module is importable.

    Delegates to ``utils.framework_availability.is_framework_available``, passing
    ``executor`` so the check targets the subprocess-invoked interpreter rather
    than the caller's. Preserves the pre-Phase-2.3 call-site signature.
    """
    return _is_framework_available_by_name(framework, executor=executor, logger=logger)


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
    return {
        "script_path": str(script_info["path"]),
        "script_name": script_info["name"],
        "framework": framework,
        "model_name": model_name,
        "executor": executor,
        "success": False,
        "skipped": True,
        "status": "skipped",
        "attempts_started": 0,
        "return_code": None,
        "stdout": "",
        "stderr": "",
        "execution_time": 0,
        "timestamp": datetime.now().isoformat(),
        "error": reason,
        "error_type": "DependencyNotInstalled",
        "execution_metadata": _load_rxinfer_execution_metadata_from_script(
            Path(script_info["path"])
        )
        if framework == "rxinfer"
        else {},
    }


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
    path_parts = script_path.parts
    model_name = path_parts[-3] if len(path_parts) >= 3 else "unknown_model"
    framework = path_parts[-2] if len(path_parts) >= 3 else script_info["framework"]
    error = f"Local worker pool failed before script completion: {exc}"
    return {
        "script_path": str(script_path),
        "script_name": script_info["name"],
        "framework": framework,
        "model_name": model_name,
        "executor": script_info["executor"],
        "success": False,
        "skipped": False,
        "status": "failed",
        "attempts_started": 0,
        "return_code": None,
        "stdout": "",
        "stderr": "",
        "execution_time": 0,
        "timestamp": datetime.now().isoformat(),
        "error": error,
        "error_type": "LocalWorkerPoolError",
        "worker_pool_error_type": type(exc).__name__,
    }


def _make_distributed_dispatch_failure_result(
    script_info: Dict[str, Any],
    exc: BaseException,
    backend: str,
    max_retries: int,
) -> Dict[str, Any]:
    """Return one explicit failure when distributed dispatch cannot complete."""
    script_path = Path(script_info["path"])
    path_parts = script_path.parts
    model_name = path_parts[-3] if len(path_parts) >= 3 else "unknown_model"
    framework = path_parts[-2] if len(path_parts) >= 3 else script_info["framework"]
    return {
        "script_path": str(script_path),
        "script_name": script_info["name"],
        "framework": framework,
        "model_name": model_name,
        "executor": script_info["executor"],
        "success": False,
        "skipped": False,
        "status": "failed",
        "attempts_started": 0,
        "return_code": None,
        "stdout": "",
        "stderr": "",
        "execution_time": 0,
        "timestamp": datetime.now().isoformat(),
        "error": f"Distributed {backend} dispatch failed before completion: {exc}",
        "error_type": "DistributedDispatchError",
        "dispatch_error_type": type(exc).__name__,
        "dispatch_max_retries": max_retries,
    }


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


def parse_frameworks_parameter(frameworks: str, logger: Any) -> List[str]:
    """
    Parse the frameworks parameter into a list of framework names.

    Args:
        frameworks: Comma-separated string of framework names or preset
        logger: Logger instance

    Returns:
        List of framework names to include
    """
    if not frameworks or frameworks.lower() == "all":
        return [
            "pymdp",
            "jax",
            "discopy",
            "rxinfer",
            "activeinference_jl",
            "pytorch",
            "numpyro",
            "stan",
            "bnlearn",
        ]

    if frameworks.lower() == "lite":
        return ["pymdp", "jax", "discopy", "bnlearn"]

    # Parse comma-separated list
    framework_list = [f.strip() for f in frameworks.split(",")]
    valid_frameworks: list[Any] = [
        "pymdp",
        "jax",
        "discopy",
        "rxinfer",
        "activeinference_jl",
        "pytorch",
        "numpyro",
        "stan",
        "bnlearn",
    ]

    # Filter out invalid frameworks
    valid_list = [f for f in framework_list if f in valid_frameworks]

    if len(valid_list) != len(framework_list):
        invalid = [f for f in framework_list if f not in valid_frameworks]
        logger.warning(
            f"Invalid frameworks specified: {invalid}. Valid options: {valid_frameworks}"
        )

    return valid_list if valid_list else ["pymdp"]  # Default to pymdp if nothing valid


def _resolve_render_output_dir(
    target_dir: Path,
    kwargs: dict,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve the render output directory from kwargs and filesystem heuristics.

    Resolution priority:
    1. Explicit ``--render-output-dir`` kwarg.
    2. Sibling of the current step's output dir: when ``output_dir`` is
       ``<base>/12_execute_output``, use ``<base>/11_render_output`` (and nested layout).
    3. target_dir itself if it looks like a render output directory.
    4. Common pipeline and test output locations (searched in order).

    Returns the first existing, non-empty directory found, or None.
    """

    def _if_nonempty(p: Path) -> Optional[Path]:
        """Handle if nonempty for internal callers."""
        if p.exists() and any(p.rglob("*")):
            return p
        return None

    # Priority 1: explicit kwarg
    if kwargs.get("render_output_dir"):
        p = Path(kwargs["render_output_dir"])
        return _if_nonempty(p) or p

    # Priority 2: same pipeline base as step 12 (target often remains GNN input dir)
    if output_dir is not None:
        base = output_dir.parent
        for rel in (
            "11_render_output/11_render_output",
            "11_render_output",
        ):
            found = _if_nonempty(base / rel)
            if found is not None:
                return found

    # Priority 3: target_dir is already the render output
    if "11_render_output" in str(target_dir) or target_dir.name == "11_render_output":
        return _if_nonempty(target_dir) or target_dir

    # Priority 4: search common cwd-relative locations.
    candidates: List[Path] = [
        target_dir.parent / "output" / "11_render_output",
        target_dir / "11_render_output",
        Path("output/test_render/11_render_output/11_render_output"),
        Path("output/test_render_improved/11_render_output/11_render_output"),
        *list(Path("output").glob("*/11_render_output/11_render_output")),
        *list(Path("output").glob("**/11_render_output")),
    ]
    for candidate in candidates:
        found = _if_nonempty(candidate)
        if found is not None:
            return found
    return None


def _load_render_summary_contract(
    render_output_dir: Path,
    requested_frameworks: List[str],
    logger: logging.Logger,
    target_dir: Optional[Path] = None,
) -> Tuple[Optional[set[Path]], List[Dict[str, str]]]:
    """Load the latest Step 11 render contract for script filtering and failures.

    ``target_dir`` (when provided) scopes the contract to the current pipeline
    invocation. The pipeline runs Step 12 once per top-level input folder, so a
    folder-scoped invocation must only discover scripts rendered from that
    folder's source files; a global invocation (``target_dir`` is the base
    input dir) naturally matches every source file.
    """
    summary_file = render_output_dir / "render_processing_summary.json"
    if not summary_file.exists():
        return None, []

    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read render summary %s: %s", summary_file, exc)
        return None, []

    def _in_scope(source_file: str) -> bool:
        if target_dir is None:
            return True
        try:
            return Path(source_file).resolve().is_relative_to(target_dir.resolve())
        except (OSError, ValueError):
            return True

    requested = set(requested_frameworks)
    allowed_scripts: set[Path] = set()
    render_failures: List[Dict[str, str]] = []
    file_results = summary.get("file_results")
    if not isinstance(file_results, dict):
        return None, []

    for source_file, file_result in file_results.items():
        if not _in_scope(source_file):
            continue
        framework_results = (
            file_result.get("framework_results", {})
            if isinstance(file_result, dict)
            else {}
        )
        if not isinstance(framework_results, dict):
            continue
        for framework, framework_result in framework_results.items():
            if framework not in requested or not isinstance(framework_result, dict):
                continue
            if framework_result.get("unsupported"):
                # The renderer declared this framework cannot represent the
                # model (categorical backend, continuous model): nothing to
                # execute and nothing failed.
                continue
            if framework_result.get("success"):
                for output_file in framework_result.get("output_files") or []:
                    # Only executable artifacts are execution candidates. Stan
                    # renders a companion ``.stan`` program beside its Python
                    # driver; data/config side files are never "missing scripts".
                    if Path(output_file).suffix.lower() not in _EXECUTABLE_SUFFIXES:
                        continue
                    allowed_scripts.add(Path(output_file).resolve())
            else:
                render_failures.append(
                    {
                        "file": Path(str(source_file)).name,
                        "framework": str(framework),
                        "message": str(framework_result.get("message", "")),
                    }
                )

    return allowed_scripts, render_failures


def _summarize_collected_outputs(coll: Any) -> Any:
    """Replace bulky collected_outputs with counts safe for aggregate JSON."""
    if coll is None:
        return None
    if isinstance(coll, dict):
        out: Dict[str, Any] = {}
        for k, v in coll.items():
            if isinstance(v, list):
                out[str(k)] = {"count": len(v)}
            elif isinstance(v, dict):
                out[str(k)] = {"n_keys": len(v)}
            else:
                out[str(k)] = v
        return out
    if isinstance(coll, list):
        return {"count": len(coll)}
    return coll


def _slim_execution_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
    """Strip heavy fields from a per-script execution result for aggregate summaries."""
    keys_keep = (
        "script_path",
        "script_name",
        "framework",
        "model_name",
        "executor",
        "success",
        "skipped",
        "status",
        "attempts_started",
        "return_code",
        "error",
        "error_type",
        "execution_time",
        "timestamp",
        "execution_benchmark_repeats",
        "execution_time_mean",
        "execution_time_std",
        "execution_time_samples",
        "structured_result_file",
        "output_file",
        "implementation_directory",
        "execution_metadata",
    )
    slim: Dict[str, Any] = {}
    for k in keys_keep:
        if k in detail:
            slim[k] = detail[k]
    if isinstance(detail.get("stdout"), str):
        slim["stdout_length"] = len(detail["stdout"])
    if isinstance(detail.get("stderr"), str):
        slim["stderr_length"] = len(detail["stderr"])
    if "collected_outputs" in detail:
        slim["collected_outputs_summary"] = _summarize_collected_outputs(
            detail["collected_outputs"]
        )
    return slim


def _execution_detail_key(detail: Dict[str, Any]) -> str:
    """Stable identity for one executed script across folder invocations."""
    return str(
        detail.get("script_path")
        or detail.get("path")
        or detail.get("script_name")
        or detail.get("name")
        or ""
    )


_STATUS_SEVERITY = {
    "failed": 3,
    "skipped": 2,
    "success_with_render_failures": 1,
    "success_with_skips": 1,
    "success": 0,
}


def _merge_prior_execution_summary(
    execution_results: Dict[str, Any], results_file: Path, logger: Any
) -> None:
    """Fold a previously written ``execution_summary.json`` into the current results.

    Details are keyed by script path; a script re-executed in this invocation
    replaces its earlier record. Aggregate counts, per-framework status and the
    overall status are recomputed over the merged detail set so downstream
    consumers (Step 16/23) never see only the last folder's slice.
    """
    if not results_file.exists():
        return
    try:
        prior = json.loads(results_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not read prior execution summary %s: %s", results_file, exc
        )
        return
    if not isinstance(prior, dict):
        return
    prior_details = prior.get("execution_details") or []
    if not isinstance(prior_details, list) or not prior_details:
        return

    current_details = list(execution_results.get("execution_details") or [])
    current_keys = {
        _execution_detail_key(d) for d in current_details if isinstance(d, dict)
    }
    carried = [
        d
        for d in prior_details
        if isinstance(d, dict) and _execution_detail_key(d) not in current_keys
    ]
    if not carried:
        return

    # Preserve this invocation's own verdict before the aggregate overwrites
    # ``status``/``success``: the function's return value describes the
    # current folder, the durable summary describes every folder so far.
    execution_results["current_invocation"] = {
        "target_directory": execution_results.get("target_directory"),
        "status": execution_results.get("status"),
        "success": execution_results.get("success"),
        "outcome_reason": execution_results.get("outcome_reason"),
        "total_scripts_found": execution_results.get("total_scripts_found"),
        "successful_executions": execution_results.get("successful_executions"),
        "failed_executions": execution_results.get("failed_executions"),
        "skipped_executions": execution_results.get("skipped_executions", 0),
    }

    merged = carried + current_details
    execution_results["execution_details"] = merged
    execution_results["merged_prior_folder_runs"] = (
        int(prior.get("merged_prior_folder_runs", 0)) + 1
    )

    successful = failed = skipped = 0
    framework_status: Dict[str, Dict[str, Any]] = {}
    for d in merged:
        fw = str(d.get("framework", "unknown"))
        fs = framework_status.setdefault(
            fw,
            {
                "status": "unknown",
                "executions": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0,
            },
        )
        fs["executions"] += 1
        if d.get("skipped"):
            skipped += 1
            fs["skipped"] += 1
        elif d.get("success"):
            successful += 1
            fs["successful"] += 1
        else:
            failed += 1
            fs["failed"] += 1
        if d.get("error") and not d.get("success"):
            fs["error"] = d["error"]
    for fs in framework_status.values():
        if fs["failed"]:
            fs["status"] = "failed"
        elif fs["successful"] and fs["skipped"]:
            fs["status"] = "success_with_skips"
        elif fs["successful"]:
            fs["status"] = "success"
        else:
            fs["status"] = "skipped"

    execution_results["framework_status"] = framework_status
    execution_results["total_scripts_found"] = len(merged)
    execution_results["total_scripts"] = len(merged)
    execution_results["successful_executions"] = successful
    execution_results["failed_executions"] = failed
    execution_results["skipped_executions"] = skipped
    attempted = len(merged) - skipped
    execution_results["attempted_scripts"] = attempted
    execution_results["success_rate"] = (
        round(successful / attempted * 100, 2) if attempted > 0 else 0.0
    )
    for key in ("render_failures", "missing_render_scripts"):
        prior_items = prior.get(key) or []
        if isinstance(prior_items, list) and prior_items:
            seen = {
                json.dumps(i, sort_keys=True, default=str)
                for i in execution_results.get(key, [])
            }
            for item in prior_items:
                if json.dumps(item, sort_keys=True, default=str) not in seen:
                    execution_results.setdefault(key, []).append(item)

    prior_status = str(prior.get("status", "success"))
    current_status = str(execution_results.get("status", "success"))
    if _STATUS_SEVERITY.get(prior_status, 0) > _STATUS_SEVERITY.get(current_status, 0):
        execution_results["status"] = prior_status
        execution_results["outcome_reason"] = prior.get(
            "outcome_reason", execution_results.get("outcome_reason")
        )
        execution_results["success"] = bool(prior.get("success", False))
        execution_results["exit_code"] = prior.get(
            "exit_code", execution_results.get("exit_code")
        )
    elif failed > 0 and execution_results.get("status") != "failed":
        execution_results["status"] = "failed"
        execution_results["success"] = False
        execution_results["exit_code"] = 1
        execution_results["outcome_reason"] = "script_execution_failure"


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
        execution_results: dict[str, Any] = {
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
                    elif (
                        framework_summary["successful"] and framework_summary["skipped"]
                    ):
                        framework_summary["status"] = "success_with_skips"
                    elif framework_summary["successful"]:
                        framework_summary["status"] = "success"
                    else:
                        framework_summary["status"] = "skipped"

        # Classify the outcome before serialising it. The previous flow wrote
        # ``success: true`` and only afterwards returned False for failures,
        # leaving the durable summary in direct conflict with the API result.
        total_found = execution_results["total_scripts_found"]
        successful = execution_results["successful_executions"]
        failed = execution_results["failed_executions"]
        skipped = execution_results.get("skipped_executions", 0)
        attempted = total_found - skipped
        render_failures = execution_results.get("render_failures", [])
        missing_render_scripts = execution_results.get("missing_render_scripts", [])
        missing_render_summary = execution_results.get("missing_render_summary")

        outcome: bool | int
        if missing_render_summary:
            outcome = False
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

        execution_results["total_scripts"] = total_found
        execution_results["attempted_scripts"] = attempted
        execution_results["success_rate"] = (
            round(successful / attempted * 100, 2) if attempted > 0 else 0.0
        )
        execution_results["success"] = outcome is True
        execution_results["status"] = status
        execution_results["exit_code"] = (
            0 if outcome is True else outcome if outcome == 2 else 1
        )
        execution_results["outcome_reason"] = reason

        # Save detailed results to summaries subfolder (slim aggregate + optional full detail file)
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


def find_executable_scripts(
    render_output_dir: Path,
    verbose: bool,
    logger: Any,
    requested_frameworks: List[str],
    allowed_scripts: Optional[set[Path]] = None,
) -> List[Dict[str, Any]]:
    """Find executable scripts in the render output directory.

    **Discovery strategy (V-10)**:

    1. Manifest-first: when ``allowed_scripts`` is provided (from a
       ``render_processing_summary.json`` manifest), only those scripts the
       render step actually produced are considered. No blanket file-tree
       walk — stale or un-rendered scripts are ignored.

    2. rglob fallback: when the manifest is missing or corrupt,
       ``allowed_scripts`` is ``None`` and the function performs the
       traditional recursive file walk. A warning is emitted since this may
       pick up stale intermediate files.

    Scripts are filtered by the requested frameworks and excluded if they
    match common non-executable patterns (test files, __init__.py, etc.).

    Args:
        render_output_dir: Directory containing rendered scripts from Step 11.
        verbose: Enable verbose logging of discovered scripts.
        logger: Logger instance for output messages.
        requested_frameworks: List of framework names to include (e.g.,
            ["pymdp", "jax", "discopy"]). Scripts from other frameworks
            will be skipped.
        allowed_scripts: Optional set of resolved Paths from the Step 11
            manifest. When not None, these paths replace the rglob step.

    Returns:
        List of dictionaries, each containing:
            - path: Path to the script file
            - name: Script filename
            - framework: Detected framework name
            - executor: Command to execute the script (python/julia)
            - relative_path: Path relative to render_output_dir
            - size_bytes: File size in bytes
    """
    executable_scripts: list[Any] = []

    # Define supported script types and their executors
    script_types: dict[str, Any] = {
        "*.py": {"executor": sys.executable, "framework": "python"},
        "*.jl": {"executor": "julia", "framework": "julia"},
    }

    # Map framework directories to framework names
    framework_dirs: dict[str, Any] = {
        "pymdp": "pymdp",
        "jax": "jax",
        "discopy": "discopy",
        "rxinfer": "rxinfer",
        "activeinference_jl": "activeinference_jl",
        "activeinference.jl": "activeinference_jl",
        "pytorch": "pytorch",
        "numpyro": "numpyro",
        "stan": "stan",
        "bnlearn": "bnlearn",
    }

    # Normalise the base directory for consistent framework detection and
    # relative-path computation across both discovery modes.
    base_dir = render_output_dir.resolve()

    # --- Phase 1: Discover candidate script paths ---
    if allowed_scripts is not None:
        # Manifest-based discovery (V-10): only rendered scripts qualify.
        if verbose:
            logger.info(
                f"Discovering scripts from render manifest "
                f"({len(allowed_scripts)} rendered scripts listed)"
            )
        manifest_paths: list[Path] = []
        for p in allowed_scripts:
            candidate = Path(p).resolve()
            if not candidate.exists():
                # Old rglob only ever surfaced files that exist; a manifest
                # entry whose file is missing is reported downstream as a
                # missing rendered script rather than executed.
                logger.warning(
                    f"Render manifest references missing script: {candidate}"
                )
                continue
            manifest_paths.append(candidate)
        candidates = sorted(manifest_paths)
    else:
        # rglob fallback: crawl the directory tree when no manifest is present.
        logger.warning(
            "No render manifest provided — falling back to recursive rglob "
            "discovery. This may include stale or un-rendered scripts."
        )
        candidates = []
        for pattern, config in script_types.items():
            # rglob on an absolute path yields absolute paths.
            candidates.extend(base_dir.rglob(pattern))

    # --- Phase 2: Build script-info dicts ---
    for script_path in candidates:
        # Skip support modules in test folders without excluding rendered
        # model scripts whose model name naturally starts with "test_".
        script_name = script_path.name.lower()
        path_parts = {part.lower() for part in script_path.parts}
        if (
            script_name == "__init__.py"
            or script_name.startswith("__")
            or script_path.stem.lower().endswith("_test")
            or "tests" in path_parts
        ):
            continue

        # Determine framework from directory path
        framework = determine_script_framework(script_path, base_dir, framework_dirs)

        # Filter by requested frameworks
        if framework not in requested_frameworks:
            if verbose:
                logger.debug(
                    f"Skipping {framework} script: {script_path.name} "
                    f"(not in requested frameworks)"
                )
            continue

        # Resolve executor from the file suffix
        suffix = script_path.suffix.lower()
        if suffix == ".py":
            executor = sys.executable
        elif suffix == ".jl":
            executor = "julia"
        else:
            continue  # not a recognised script type

        # Compute relative path (best-effort — may not be under base_dir
        # when the manifest is a direct pass-through from the render step).
        try:
            rel = script_path.relative_to(base_dir)
        except ValueError:
            rel = script_path

        # Check if script is executable or can be made executable
        script_info: dict[str, Any] = {
            "path": script_path,
            "name": script_path.name,
            "framework": framework,
            "executor": executor,
            "relative_path": rel,
            "size_bytes": script_path.stat().st_size if script_path.exists() else 0,
        }

        executable_scripts.append(script_info)

        if verbose:
            logger.info(f"Found {framework} script: {rel}")

    return executable_scripts


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


def _build_script_execution_context(
    script_info: Dict[str, Any],
) -> ScriptExecutionContext:
    """Normalize model/framework metadata from a rendered script path."""
    script_path = script_info["path"]
    path_parts = script_path.parts
    if len(path_parts) >= 3:
        model_name = path_parts[-3]
        framework = path_parts[-2]
    else:
        model_name = "unknown_model"
        framework = script_info["framework"]

    return ScriptExecutionContext(
        script_path=script_path,
        script_name=script_info["name"],
        framework=framework,
        model_name=model_name,
        executor=script_info["executor"],
    )


def _new_execution_result(context: ScriptExecutionContext) -> Dict[str, Any]:
    """Create the standard execution result envelope."""
    return {
        "script_path": str(context.script_path),
        "script_name": context.script_name,
        "framework": context.framework,
        "model_name": context.model_name,
        "executor": context.executor,
        "success": False,
        "skipped": False,
        "status": "failed",
        "attempts_started": 0,
        "return_code": None,
        "stdout": "",
        "stderr": "",
        "execution_time": 0,
        "timestamp": datetime.now().isoformat(),
    }


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

        class ErrorResult:
            """Provide ErrorResult behavior."""

            def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
                """Initialize the instance."""
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        # Execute the script with improved error handling
        result: subprocess.CompletedProcess[str] | ErrorResult | None = None

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
                    result = ErrorResult(-1, "", "Timeout")
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
            result = ErrorResult(-2, "", str(e))

        # Ensure result is defined before using it
        if result is None:
            result = ErrorResult(-3, "", "Unknown error")

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
        accelerator_type = "cpu"
        try:
            if shutil.which("nvidia-smi") is not None:
                accelerator_type = "cuda"
            elif sys.platform == "darwin":
                accelerator_type = "mps"
        except Exception:
            accelerator_type = "cpu"

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


# --- Public exports from sub-modules ---
from .data_extractors import (
    collect_execution_outputs,
)
from .data_extractors import (
    extract_simulation_data as _extract_simulation_data,
)
from .data_extractors import (
    extract_simulation_data_from_files as _extract_simulation_data_from_files,
)


def generate_execution_report(
    execution_results: Dict[str, Any], results_dir: Path, logger: logging.Logger
) -> None:
    """
    Generate a comprehensive execution report.

    Args:
        execution_results: Dictionary with execution results
        results_dir: Directory to save the report
        logger: Logger instance
    """
    summaries_dir = results_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    report_file = summaries_dir / "execution_report.md"

    try:
        with open(report_file, "w") as f:
            f.write("# GNN Script Execution Report\n\n")
            f.write(f"**Generated:** {execution_results['timestamp']}\n")
            f.write(f"**Target Directory:** {execution_results['target_directory']}\n")
            f.write(
                f"**Output Directory:** {execution_results['output_directory']}\n\n"
            )

            f.write("## Summary\n\n")
            f.write(
                f"- **Total Scripts Found:** {execution_results['total_scripts_found']}\n"
            )
            f.write(
                f"- **Successful Executions:** {execution_results['successful_executions']}\n"
            )
            f.write(
                f"- **Failed Executions:** {execution_results['failed_executions']}\n"
            )
            skipped = execution_results.get("skipped_executions", 0)
            if skipped:
                f.write(f"- **Skipped (dependency not installed):** {skipped}\n")
            f.write("\n")

            if execution_results["execution_details"]:
                f.write("## Execution Details\n\n")

                for detail in execution_results["execution_details"]:
                    if detail.get("skipped"):
                        status = "⏭️ SKIPPED"
                    elif detail["success"]:
                        status = "✅ SUCCESS"
                    else:
                        status = "❌ FAILED"
                    f.write(f"### {detail['script_name']} - {status}\n\n")
                    f.write(f"- **Framework:** {detail['framework']}\n")
                    f.write(f"- **Executor:** {detail['executor']}\n")
                    f.write(f"- **Path:** `{detail['script_path']}`\n")
                    if detail.get("skipped"):
                        f.write(
                            f"- **Reason:** {detail.get('error', 'Dependency not installed')}\n"
                        )
                    else:
                        f.write(
                            f"- **Return Code:** {detail.get('return_code', 'N/A')}\n"
                        )
                        f.write(
                            f"- **Execution Time:** {detail.get('execution_time', 0):.2f} seconds\n"
                        )

                        if not detail["success"] and "error" in detail:
                            f.write(f"- **Error:** {detail['error']}\n")

                        if "output_file" in detail:
                            f.write(f"- **Detailed Output:** {detail['output_file']}\n")

                    f.write("\n")

            f.write("## Next Steps\n\n")
            if execution_results["failed_executions"] > 0:
                f.write("1. Review failed executions above\n")
                f.write(
                    "2. Check individual output files for detailed error information\n"
                )
                f.write("3. Ensure required dependencies are installed\n")
                f.write("4. Verify script syntax and functionality\n\n")
            elif skipped:
                f.write(
                    "Skipped scripts are due to missing optional dependencies or unavailable system runtimes. Run `uv sync` for core Python backends; add `uv sync --extra ml-ai --extra graphs` for optional Python extension groups, and install Julia/D2 system tools as needed.\n\n"
                )
            else:
                f.write(
                    "All scripts executed successfully! Check individual output files for results.\n\n"
                )

        logger.info(f"Generated execution report: {report_file}")

    except Exception as e:
        logger.error(f"Failed to generate execution report: {e}")


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
