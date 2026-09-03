#!/usr/bin/env python3
"""
Execution metadata and summary artifacts for GNN Step 12.

Owns the RxInfer execution-metadata sidecar/TOML loaders, the Step 11
render-summary contract, execution-summary merging/slimming, and the
markdown execution report. Extracted from ``execute.processor``.
"""

import hashlib
import json
import logging
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import _EXECUTABLE_SUFFIXES

logger = logging.getLogger(__name__)


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
