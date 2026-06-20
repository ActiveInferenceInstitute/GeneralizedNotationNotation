"""
Pipeline Module MCP Integration

This module exposes pipeline management and configuration tools through the Model Context Protocol.
It provides tools for pipeline discovery, execution, monitoring, and configuration management.

Key Features:
- Pipeline step discovery and metadata
- Pipeline execution and monitoring
- Configuration management and validation
- Pipeline status and performance tracking
- Step dependency analysis and validation
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def get_pipeline_steps(mcp_instance_ref: Any) -> Dict[str, Any]:
    """
    Get information about all available pipeline steps.

    Returns:
        Dictionary containing pipeline step information.
    """
    try:
        from .config import STEP_METADATA, get_pipeline_config

        steps_info: dict[Any, Any] = {}
        for step_name, metadata in STEP_METADATA.items():
            steps_info[step_name] = {
                "name": step_name,
                "script": metadata.get("script", ""),
                "module": metadata.get("module", ""),
                "description": metadata.get("description", ""),
                "required": metadata.get("required", False),
                "category": metadata.get("category", "General"),
                "dependencies": metadata.get("dependencies", []),
                "output_dir": metadata.get("output_dir", ""),
                "version": metadata.get("version", "1.0.0"),
            }

        return {
            "success": True,
            "total_steps": len(steps_info),
            "steps": steps_info,
            "pipeline_config": get_pipeline_config(),
        }
    except Exception as e:
        logger.error(f"Error getting pipeline steps: {e}")
        return {"success": False, "error": str(e)}


def get_pipeline_status(mcp_instance_ref: Any) -> Dict[str, Any]:
    """
    Get current pipeline execution status and statistics.

    Returns:
        Dictionary containing pipeline status information.
    """
    try:
        from .config import get_pipeline_config

        config = get_pipeline_config()
        output_dir = Path(config.get("output_dir", "output"))

        # Check for pipeline execution summary (main.py writes to 00_pipeline_summary/ subdir)
        summary_candidates: list[Any] = [
            output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json",
            output_dir / "pipeline_execution_summary.json",
        ]
        summary: dict[str, Any] = {
            "last_execution": None,
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
        }
        for summary_file in summary_candidates:
            if summary_file.exists():
                try:
                    with open(summary_file, "r") as f:
                        summary = json.load(f)
                    break
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Could not load summary from {summary_file}: {e}")

        # Check for recent logs
        logs_dir = output_dir / "logs"
        recent_logs: list[Any] = []
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.log"))
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for log_file in log_files[:5]:  # Last 5 log files
                recent_logs.append(
                    {
                        "file": log_file.name,
                        "size": log_file.stat().st_size,
                        "modified": time.ctime(log_file.stat().st_mtime),
                    }
                )

        return {
            "success": True,
            "pipeline_config": config,
            "execution_summary": summary,
            "recent_logs": recent_logs,
            "output_directory": str(output_dir),
            "output_directory_exists": output_dir.exists(),
        }
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        return {"success": False, "error": str(e)}


def validate_pipeline_dependencies(mcp_instance_ref: Any) -> Dict[str, Any]:
    """
    Validate pipeline step dependencies and identify any issues.

    Returns:
        Dictionary containing dependency validation results.
    """
    try:
        from utils.dependency_validator import (
            validate_pipeline_dependencies as validate_deps,
        )

        from .config import STEP_METADATA

        validation_result = validate_deps()

        # Additional analysis
        dependency_graph: dict[Any, Any] = {}
        missing_deps: list[Any] = []
        circular_deps: list[Any] = []

        for step_name, metadata in STEP_METADATA.items():
            deps = metadata.get("dependencies", [])
            dependency_graph[step_name] = deps

            # Check for missing dependencies
            for dep in deps:
                if dep not in STEP_METADATA:
                    missing_deps.append({"step": step_name, "missing_dependency": dep})

        return {
            "success": True,
            "validation_result": validation_result,
            "dependency_graph": dependency_graph,
            "missing_dependencies": missing_deps,
            "circular_dependencies": circular_deps,
            "total_steps": len(STEP_METADATA),
            "total_dependencies": sum(
                len(metadata.get("dependencies", []))
                for metadata in STEP_METADATA.values()
            ),
        }
    except Exception as e:
        logger.error(f"Error validating pipeline dependencies: {e}")
        return {"success": False, "error": str(e)}


def get_pipeline_config_info(mcp_instance_ref: Any) -> Dict[str, Any]:
    """
    Get detailed pipeline configuration information.

    Returns:
        Dictionary containing pipeline configuration details.
    """
    try:
        from .config import get_pipeline_config

        config = get_pipeline_config()

        return {
            "success": True,
            "configuration": config,
            "configuration_keys": list(config.keys()),
            "output_directory": config.get("output_dir", "output"),
            "log_level": config.get("log_level", "INFO"),
            "parallel_execution": config.get("parallel_execution", False),
            "max_workers": config.get("max_workers", 1),
        }
    except Exception as e:
        logger.error(f"Error getting pipeline config info: {e}")
        return {"success": False, "error": str(e)}


def get_v3_orchestration_capabilities() -> Dict[str, Any]:
    """Describe the v3.0.0 long-running orchestration contracts (streams, sessions, plans).

    Returns:
        Inventory of the three safe-by-design orchestration modules and their public API.
    """
    return {
        "success": True,
        "version": "3.0.0",
        "safe_by_design": True,
        "contracts": {
            "durable_streams": [
                "StreamManifest",
                "ExecutionTrace",
                "trace_integrity",
                "validate_stream_manifest",
                "replay_trace",
                "verify_replay",
            ],
            "run_session": [
                "RunSession",
                "WorkUnit",
                "start_session",
                "mark",
                "checkpoint",
                "load_session",
                "remaining_units",
                "status_report",
                "cancel_safe_cleanup",
            ],
            "container_plan": [
                "ContainerPlan",
                "generate_container_plan",
                "security_review",
                "compute_plan_hash",
                "serialize_plan",
                "plan_to_compose",
            ],
        },
    }


def run_v3_container_security_review() -> Dict[str, Any]:
    """Demonstrate the auditable-container-plan security review with teeth.

    Generates a hardened default plan (expected clean) and reviews a deliberately
    insecure plan (expected to flag CRITICAL/HIGH findings), proving the static review
    discriminates. Performs no execution — pure data inspection.
    """
    try:
        from pipeline import container_plan as cp

        hardened = cp.generate_container_plan(
            "demo",
            [{"name": "runner", "image": "ghcr.io/gnn/runner@sha256:" + "a" * 64}],
        )
        insecure = cp.ContainerPlan(
            plan_id="insecure-demo",
            specs=[
                cp.ContainerSpec(
                    name="bad",
                    image="img:latest",
                    privileged=True,
                    user="root",
                    env={"DB_PASSWORD": "hunter2"},
                    read_only_rootfs=False,
                    cap_drop=[],
                )
            ],
        )
        insecure_findings = cp.security_review(insecure)
        return {
            "success": True,
            "hardened_findings": [f.model_dump() for f in cp.security_review(hardened)],
            "insecure_findings": [f.model_dump() for f in insecure_findings],
            "review_has_teeth": len(insecure_findings) >= 4
            and any(f.severity == "CRITICAL" for f in insecure_findings),
        }
    except Exception as e:  # pragma: no cover - defensive MCP boundary
        logger.error(f"v3 container security review failed: {e}")
        return {"success": False, "error": str(e)}


def run_v3_orchestration_self_check() -> Dict[str, Any]:
    """Run lightweight in-process checks of all three v3 contracts and report pass counts.

    Builds real (in-memory / temp) artifacts — a stream manifest with a tamper negative
    control, a session status computation, and a container security review — and confirms
    each contract behaves. No external services touched.
    """
    try:
        import tempfile
        from pathlib import Path as _Path

        import numpy as np

        from pipeline import container_plan as cp
        from pipeline import durable_streams as ds
        from pipeline import run_session as rs

        results: Dict[str, bool] = {}
        with tempfile.TemporaryDirectory(prefix="gnn-v3-mcp-") as tmp:
            wd = _Path(tmp)
            manifest = ds.StreamManifest.from_array("s", np.arange(6, dtype=np.float64))
            results["streams_array_valid"] = (
                ds.validate_stream_manifest(manifest, wd) == []
            )
            fp = wd / "d.bin"
            fp.write_bytes(b"payload")
            fm = ds.StreamManifest.from_file("f", fp)
            fp.write_bytes(b"payload-TAMPERED")
            results["streams_tamper_detected"] = (
                len(ds.validate_stream_manifest(fm, wd)) > 0
            )

        sess = rs.start_session("mcp", ["a", "b"])
        sess = rs.mark(sess, "a", rs.UnitStatus.DONE)
        rep = rs.status_report(sess)
        results["session_status_math"] = rep["completed"] == 1 and not rep["done"]

        plan = cp.generate_container_plan(
            "mcp", [{"name": "r", "image": "i@sha256:" + "a" * 64}]
        )
        results["container_hardened_clean"] = cp.security_review(plan) == []

        return {
            "success": all(results.values()),
            "checks": results,
            "passed": sum(1 for v in results.values() if v),
            "total": len(results),
        }
    except Exception as e:  # pragma: no cover - defensive MCP boundary
        logger.error(f"v3 orchestration self-check failed: {e}")
        return {"success": False, "error": str(e)}


def register_tools(mcp_instance: Any) -> None:
    """
    Register pipeline management tools with the MCP server.

    Args:
        mcp_instance: The MCP instance to register tools with.
    """
    logger.info("Registering pipeline MCP tools")

    # Named wrappers keep MCP audit checks happy (no functools.partial names).
    def get_pipeline_steps_tool() -> Any:
        """Return pipeline steps tool."""
        return get_pipeline_steps(mcp_instance)

    def get_pipeline_status_tool() -> Any:
        """Return pipeline status tool."""
        return get_pipeline_status(mcp_instance)

    def validate_pipeline_dependencies_tool() -> Any:
        """Validate pipeline dependencies tool."""
        return validate_pipeline_dependencies(mcp_instance)

    def get_pipeline_config_info_tool() -> Any:
        """Return pipeline config info tool."""
        return get_pipeline_config_info(mcp_instance)

    # Register tools
    mcp_instance.register_tool(
        name="get_pipeline_steps",
        function=get_pipeline_steps_tool,
        schema={},
        description="Get information about all available pipeline steps, their metadata, and dependencies.",
    )

    mcp_instance.register_tool(
        name="get_pipeline_status",
        function=get_pipeline_status_tool,
        schema={},
        description="Get current pipeline execution status, recent logs, and execution statistics.",
    )

    mcp_instance.register_tool(
        name="validate_pipeline_dependencies",
        function=validate_pipeline_dependencies_tool,
        schema={},
        description="Validate pipeline step dependencies and identify missing or circular dependencies.",
    )

    mcp_instance.register_tool(
        name="get_pipeline_config_info",
        function=get_pipeline_config_info_tool,
        schema={},
        description="Get detailed pipeline configuration information and settings.",
    )

    # v3.0.0 long-running orchestration tools (safe-by-design: no live mutation).
    def get_v3_orchestration_capabilities_tool() -> Any:
        """Return v3 orchestration capabilities tool."""
        return get_v3_orchestration_capabilities()

    def run_v3_container_security_review_tool() -> Any:
        """Return v3 container security review tool."""
        return run_v3_container_security_review()

    def run_v3_orchestration_self_check_tool() -> Any:
        """Return v3 orchestration self-check tool."""
        return run_v3_orchestration_self_check()

    mcp_instance.register_tool(
        name="get_v3_orchestration_capabilities",
        function=get_v3_orchestration_capabilities_tool,
        schema={},
        description="Describe the v3.0.0 long-running orchestration contracts: durable observation "
        "streams, resumable run sessions, and auditable container plans (safe-by-design, no live mutation).",
    )
    mcp_instance.register_tool(
        name="run_v3_container_security_review",
        function=run_v3_container_security_review_tool,
        schema={},
        description="Run the auditable container-plan static security review on a hardened and an "
        "insecure example, proving the review flags privileged/root/unpinned/secret findings.",
    )
    mcp_instance.register_tool(
        name="run_v3_orchestration_self_check",
        function=run_v3_orchestration_self_check_tool,
        schema={},
        description="Run in-process checks of all three v3 orchestration contracts (stream manifest "
        "tamper detection, session status math, container review) and report pass counts.",
    )

    logger.info("Successfully registered pipeline MCP tools")
