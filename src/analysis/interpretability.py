"""Model-family interpretability summaries for acceptance evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from gnn.discovery import is_model_source_path
from gnn.pomdp_extractor import POMDPExtractor
from gnn.schema import parse_connections, parse_state_space, validate_matrix_dimensions

TRACE_KEYS = (
    "free_energy_trace",
    "belief_trace",
    "action_trace",
    "actions",
    "observations",
    "expected_free_energy",
)


def build_model_interpretability_summary(
    model_path: Path, pipeline_output_dir: Path | None = None
) -> Dict[str, Any]:
    """Summarize one GNN model from parser, POMDP, and artifact evidence."""
    content = model_path.read_text(encoding="utf-8")
    variables, variable_errors = parse_state_space(content, file_path=str(model_path))
    variable_names = {variable.name for variable in variables}
    connections, connection_errors = parse_connections(
        content,
        known_variables=variable_names,
        file_path=str(model_path),
    )
    matrix_errors = validate_matrix_dimensions(
        content, variables, file_path=str(model_path)
    )
    pomdp_space = POMDPExtractor(strict_validation=False).extract_from_gnn_content(
        content
    )
    matrices = pomdp_space.matrices if pomdp_space and pomdp_space.matrices else {}
    pipeline_output_dir = pipeline_output_dir or Path()
    telemetry_preview = _collect_trace_preview(pipeline_output_dir, model_path.stem)
    return {
        "model_name": _extract_model_name(content) or model_path.stem,
        "source_file": str(model_path),
        "variables": [
            {
                "name": variable.name,
                "dimensions": variable.dimensions,
                "dtype": variable.dtype,
            }
            for variable in variables
        ],
        "variable_count": len(variables),
        "connections": [
            {
                "source": edge.source,
                "target": edge.target,
                "directed": edge.directed,
                "label": edge.label,
            }
            for edge in connections
        ],
        "connection_count": len(connections),
        "matrix_shapes": {
            name: _shape_of(value) for name, value in sorted(matrices.items())
        },
        "validation_messages": [
            str(error)
            for error in [*variable_errors, *connection_errors, *matrix_errors]
        ],
        "pipeline_evidence": _collect_pipeline_evidence(pipeline_output_dir),
        "telemetry_present": bool(telemetry_preview),
        "telemetry_preview": telemetry_preview,
        "artifact_links": _collect_artifact_links(pipeline_output_dir, model_path.stem),
    }


def build_family_interpretability_summary(
    family_name: str, target_dir: Path, pipeline_output_dir: Path | None = None
) -> Dict[str, Any]:
    """Summarize all representative GNN files for one accepted model family."""
    model_paths = [
        path for path in sorted(target_dir.rglob("*.md")) if is_model_source_path(path)
    ]
    model_summaries = [
        build_model_interpretability_summary(path, pipeline_output_dir)
        for path in model_paths
    ]
    return {
        "schema": "gnn_model_family_interpretability_v1",
        "family": family_name,
        "target_dir": str(target_dir),
        "model_count": len(model_summaries),
        "models": model_summaries,
        "totals": {
            "variables": sum(model["variable_count"] for model in model_summaries),
            "connections": sum(model["connection_count"] for model in model_summaries),
            "validation_messages": sum(
                len(model["validation_messages"]) for model in model_summaries
            ),
            "models_with_telemetry": sum(
                1 for model in model_summaries if model["telemetry_present"]
            ),
        },
    }


def render_family_interpretability_markdown(summary: Dict[str, Any]) -> str:
    """Render a compact family interpretability Markdown report."""
    lines = [
        f"# Model Family Interpretability: {summary['family']}",
        "",
        f"- Target directory: {summary['target_dir']}",
        f"- Models: {summary['model_count']}",
        f"- Variables: {summary['totals']['variables']}",
        f"- Connections: {summary['totals']['connections']}",
        f"- Validation messages: {summary['totals']['validation_messages']}",
        f"- Models with telemetry: {summary['totals']['models_with_telemetry']}",
        "",
        "| Model | Variables | Connections | Matrices | Render | Execute | Telemetry | Artifacts |",
        "| --- | ---: | ---: | --- | --- | --- | --- | ---: |",
    ]
    for model in summary["models"]:
        matrix_names = ", ".join(sorted(model["matrix_shapes"])) or "none"
        evidence = model.get("pipeline_evidence", {})
        lines.append(
            "| {name} | {variables} | {connections} | {matrices} | {render} | {execute} | {telemetry} | {artifacts} |".format(
                name=model["model_name"],
                variables=model["variable_count"],
                connections=model["connection_count"],
                matrices=matrix_names,
                render=evidence.get("render_status", "unknown"),
                execute=evidence.get("execution_status", "unknown"),
                telemetry="present" if model.get("telemetry_present") else "missing",
                artifacts=len(model["artifact_links"]),
            )
        )
    return "\n".join(lines) + "\n"


def _extract_model_name(content: str) -> str | None:
    match = re.search(r"^## ModelName\s*\n(?P<name>.+?)\s*$", content, re.MULTILINE)
    return match.group("name").strip() if match else None


def _shape_of(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        if not value:
            return [0]
        return [len(value), *_shape_of(value[0])]
    return []


def _collect_trace_preview(output_dir: Path, model_stem: str) -> Dict[str, Any]:
    if not output_dir.exists():
        return {}
    previews: Dict[str, Any] = {}
    for json_path in _candidate_json_files(output_dir, model_stem):
        payload = _load_json(json_path)
        for key, value in _walk_trace_values(payload):
            if key not in previews:
                previews[key] = value[:5] if isinstance(value, list) else value
    return previews


def _collect_artifact_links(output_dir: Path, model_stem: str) -> List[str]:
    if not output_dir.exists():
        return []
    artifact_paths = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and model_stem.lower() in str(path).lower()
    ]
    return [str(path) for path in sorted(artifact_paths)[:25]]


def _collect_pipeline_evidence(output_dir: Path) -> Dict[str, Any]:
    summary = _load_pipeline_summary(output_dir)
    step_statuses = _extract_step_statuses(summary)
    execution_summary = _load_first_json(
        output_dir,
        ("execution_summary.json", "execute_summary.json"),
    )
    render_summary = _load_first_json(
        output_dir,
        ("render_processing_summary.json", "render_summary.json"),
    )
    return {
        "pipeline_summary_available": bool(summary),
        "render_status": step_statuses.get("11", "unknown"),
        "execution_status": step_statuses.get("12", "unknown"),
        "render_summary_available": bool(render_summary),
        "execution_summary_available": bool(execution_summary),
        "skip_or_failure_reason": _extract_skip_or_failure_reason(
            summary, render_summary, execution_summary
        ),
    }


def _candidate_json_files(output_dir: Path, model_stem: str) -> Iterable[Path]:
    for path in sorted(output_dir.rglob("*.json")):
        if model_stem.lower() in str(path).lower() or "summary" in path.name.lower():
            yield path


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_pipeline_summary(output_dir: Path) -> Dict[str, Any]:
    summary_path = (
        output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"
    )
    payload = _load_json(summary_path) if summary_path.exists() else {}
    return payload if isinstance(payload, dict) else {}


def _load_first_json(output_dir: Path, names: tuple[str, ...]) -> Dict[str, Any]:
    if not output_dir.exists():
        return {}
    for path in sorted(output_dir.rglob("*.json")):
        if path.name in names:
            payload = _load_json(path)
            return payload if isinstance(payload, dict) else {}
    return {}


def _extract_step_statuses(summary: Dict[str, Any]) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for raw_step in summary.get("steps", []):
        if not isinstance(raw_step, dict):
            continue
        script_name = str(raw_step.get("script_name", ""))
        match = re.match(r"(?P<number>\d+)_", script_name)
        if not match:
            continue
        statuses[match.group("number")] = _normalize_status(
            str(raw_step.get("status", ""))
        )
    return statuses


def _normalize_status(status: str) -> str:
    normalized = status.strip().upper().replace("-", "_")
    if normalized in {"SUCCESS", "PASSED", "PASS", "OK"}:
        return "passed"
    if "SKIP" in normalized:
        return "skipped"
    if "SUCCESS" in normalized and "PARTIAL" not in normalized:
        return "passed"
    return "failed" if normalized else "unknown"


def _extract_skip_or_failure_reason(
    pipeline_summary: Dict[str, Any],
    render_summary: Dict[str, Any],
    execution_summary: Dict[str, Any],
) -> str | None:
    render_reasons = [
        str(item.get("message"))
        for item in render_summary.get("failed_framework_renderings", [])
        if isinstance(item, dict) and item.get("message")
    ]
    if render_reasons:
        return "; ".join(render_reasons)
    execution_reasons = []
    for key in ("skip_reason", "skipped_reason", "failure_reason", "error", "message"):
        if execution_summary.get(key):
            execution_reasons.append(str(execution_summary[key]))
    for item in execution_summary.get("render_failures", []):
        if isinstance(item, dict) and item.get("message"):
            execution_reasons.append(str(item["message"]))
    if execution_reasons:
        return "; ".join(execution_reasons)
    for raw_step in pipeline_summary.get("steps", []):
        if not isinstance(raw_step, dict):
            continue
        status = _normalize_status(str(raw_step.get("status", "")))
        if status in {"failed", "skipped"}:
            return str(
                raw_step.get("error")
                or raw_step.get("description")
                or raw_step.get("script_name")
                or status
            )
    return None


def _walk_trace_values(payload: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in TRACE_KEYS and isinstance(value, list):
                yield key, value
            yield from _walk_trace_values(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _walk_trace_values(item)
