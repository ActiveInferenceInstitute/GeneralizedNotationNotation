#!/usr/bin/env python3
"""CLI-parity endpoints for the GNN FastAPI surfaces.

``register_parity_routes`` is called by BOTH app factories — ``gnn.api.app``
(``gnn serve``) and ``gnn.api.server`` (``python -m gnn.api.server``) — so the
two surfaces expose an identical parity surface. Every endpoint runs the same
backend logic as the corresponding CLI command, in-process (no subprocess, no
BackgroundTasks), with heavy backend modules imported lazily inside handlers
to keep app boot fast and import cycles impossible.

Exit-code → HTTP mapping (pinned contract; mirrors the strict
``pipeline_exit_succeeded`` policy in ``gnn.api.pipeline_runner``):

| CLI outcome                                   | HTTP | Envelope                                          |
|-----------------------------------------------|------|---------------------------------------------------|
| exit 0 (success)                              | 200  | success envelope, ``data.exit_code == 0``         |
| exit 2 (warnings: non-strict validate/parse   | 200  | success envelope, ``data.exit_code == 2``,        |
| errors, preflight warnings)                   |      | warnings/errors listed in ``data``                |
| would-be exit 2 under ``strict=True``         | 400  | error ``bad_request``, errors in ``error.details`` |
| missing file/directory                        | 404  | error ``not_found`` (mirrors ``_guard_input_file``) |
| path-boundary violation                       | 400  | error ``bad_request`` (``PathValidationError``)   |
| backend/import/operation failure              | 400  | error ``bad_request`` with sanitized detail       |

The ``APIEnvelope.status`` Literal stays ``{"success", "error"}`` — an exit-2
outcome is a SUCCESS envelope carrying ``data.exit_code == 2``, never a
"warning" status.

Handlers are plain ``def`` (FastAPI executes them in its worker threadpool)
because every parity command performs blocking in-process backend work.
POST operation endpoints and GET ``/api/v1/models`` carry ``data.exit_code``;
the template GETs are informational and omit it.
"""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    TypeVar,
)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from gnn.api.path_utils import (
    PathValidationError,
    get_repo_root,
    resolve_repo_path,
)
from gnn.api.responses import APIEnvelope, success_envelope

logger = logging.getLogger(__name__)

#: Frameworks accepted by ``POST /api/v1/render`` — the exact choice set of
#: the CLI ``gnn render --framework`` argument.
RenderFramework = Literal[
    "pymdp",
    "rxinfer",
    "activeinference_jl",
    "jax",
    "numpyro",
    "stan",
    "pytorch",
    "discopy",
    "bnlearn",
]

ParseOutputFormat = Literal["json", "yaml", "summary"]
GraphOutputFormat = Literal["mermaid", "text"]

#: CLI exit codes reachable on a 200 parity response (strict escalation and
#: hard failures return error envelopes instead).
ParityExitCode = Literal[0, 2]

#: Server-generated scratch directory for the model registry output. Not
#: client-supplied, hence deliberately outside request path resolution.
MODEL_REGISTRY_OUTPUT_DIR = "output/model_registry_api"


# ── Request models ───────────────────────────────────────────────────────────


class ValidateRequest(BaseModel):
    """Request body for ``POST /api/v1/validate``."""

    file_path: str = Field(min_length=1, description="Repository-local GNN file")
    strict: bool = Field(
        default=False,
        description="Treat validation errors as a 400 instead of exit-code 2",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "file_path": "input/gnn_files/discrete/simple_mdp.md",
                "strict": False,
            }
        },
    )


class ParseRequest(BaseModel):
    """Request body for ``POST /api/v1/parse``."""

    file_path: str = Field(min_length=1, description="Repository-local GNN file")
    format: ParseOutputFormat = Field(
        default="json", description="Output shape: json, yaml, or summary counts"
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {"file_path": "input/gnn_files/discrete/simple_mdp.md"}
        },
    )


class ExtractRequest(BaseModel):
    """Request body for ``POST /api/v1/extract``."""

    file_path: str = Field(min_length=1, description="Repository-local GNN file")
    strict: bool = Field(
        default=True, description="Enable strict POMDP validation in the extractor"
    )
    compact: bool = Field(
        default=False,
        description="Also return the raw compact JSON payload string",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {"file_path": "input/gnn_files/discrete/simple_mdp.md"}
        },
    )


class RenderRequest(BaseModel):
    """Request body for ``POST /api/v1/render``."""

    file_path: str = Field(min_length=1, description="Repository-local GNN file")
    framework: RenderFramework = Field(description="Renderer framework to run")
    output_dir: Optional[str] = Field(
        default=None,
        description=(
            "Repository-local render output directory "
            "(defaults to output/11_render_output/<stem>)"
        ),
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "file_path": "input/gnn_files/discrete/simple_mdp.md",
                "framework": "pymdp",
            }
        },
    )


class GraphRequest(BaseModel):
    """Request body for ``POST /api/v1/graph``."""

    file_path: str = Field(min_length=1, description="Repository-local GNN file")
    format: GraphOutputFormat = Field(
        default="mermaid", description="Graph rendering format"
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {"file_path": "input/gnn_files/discrete/simple_mdp.md"}
        },
    )


class PreflightRequest(BaseModel):
    """Request body for ``POST /api/v1/preflight``."""

    config: Optional[str] = Field(
        default=None,
        description="Optional repository-local config path to validate",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {}},
    )


class ReportRequest(BaseModel):
    """Request body for ``POST /api/v1/report``."""

    output_dir: str = Field(
        min_length=1, description="Existing repository-local pipeline output directory"
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"output_dir": "output"}},
    )


class PullRequest(BaseModel):
    """Request body for ``POST /api/v1/pull``.

    Mirrors the CLI ``gnn pull`` arguments: the template is copied by
    default, and ``dry_run=True`` returns the copy plan without writing.
    """

    name: str = Field(
        min_length=1, description="Name of the maintained template to pull"
    )
    output_dir: Optional[str] = Field(
        default=None,
        description=(
            "Repository-local directory to copy the template into "
            "(defaults to input/gnn_files)"
        ),
    )
    dry_run: bool = Field(
        default=False,
        description="Return the copy plan without writing any files",
    )
    overwrite: bool = Field(
        default=False,
        description="Replace an existing destination with a different checksum",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "name": "pomdp-gridworld-3x3",
                "output_dir": "input/gnn_files",
            }
        },
    )


# ── Response models ──────────────────────────────────────────────────────────


class ParityIssue(BaseModel):
    """One structured GNN parse/validation issue (CLI error serialization)."""

    code: str
    message: str
    line: Optional[int] = None
    file: Optional[str] = None


class ValidateResponse(BaseModel):
    """Payload for ``POST /api/v1/validate`` (CLI ``gnn validate`` parity)."""

    file: str
    valid: bool
    exit_code: ParityExitCode
    errors: List[ParityIssue] = Field(default_factory=list)
    variables_count: int
    connections_count: int
    semantic: Dict[str, Any]


class ParseVariable(BaseModel):
    """One parsed state-space variable declaration."""

    name: str
    dimensions: List[str]
    dtype: str = "float"
    default: Optional[str] = None


class ParseConnection(BaseModel):
    """One parsed connection/edge between state-space variables."""

    source: str
    target: str
    directed: bool
    label: Optional[str] = None
    line: Optional[int] = None


class ParseResponse(BaseModel):
    """Payload for ``POST /api/v1/parse`` with json/yaml format."""

    file: str
    format: ParseOutputFormat
    metadata: Dict[str, Any]
    variables: List[ParseVariable]
    connections: List[ParseConnection]
    warnings: List[str] = Field(default_factory=list)
    errors: List[ParityIssue] = Field(default_factory=list)
    exit_code: ParityExitCode
    pomdp: Optional[Dict[str, Any]] = None
    yaml: Optional[str] = None


class ParseSummary(BaseModel):
    """Payload for ``POST /api/v1/parse`` with summary format."""

    file: str
    variables_count: int
    connections_count: int
    metadata_keys: List[str]
    exit_code: ParityExitCode


class ExtractResponse(BaseModel):
    """Payload for ``POST /api/v1/extract`` (CLI ``gnn extract`` parity)."""

    file: str
    strict: bool
    compact: bool
    exit_code: ParityExitCode
    pomdp: Dict[str, Any]
    payload_json: Optional[str] = None


class RenderResponse(BaseModel):
    """Payload for ``POST /api/v1/render`` (CLI ``gnn render`` parity)."""

    file: str
    framework: str
    output_dir: str
    artifact: Optional[str] = None
    exit_code: ParityExitCode


class GraphResponse(BaseModel):
    """Payload for ``POST /api/v1/graph`` (CLI ``gnn graph`` parity)."""

    file: str
    format: GraphOutputFormat
    graph: str
    exit_code: ParityExitCode


class TemplateRecordModel(BaseModel):
    """One maintained template record with checksum metadata."""

    name: str
    description: str
    source: str
    filename: str
    sha256: str


class TemplatesResponse(BaseModel):
    """Payload for ``GET /api/v1/templates``."""

    templates: List[TemplateRecordModel]
    total: int


class TemplateResponse(BaseModel):
    """Payload for ``GET /api/v1/templates/{name}``."""

    template: TemplateRecordModel


class PullResponse(BaseModel):
    """Payload for ``POST /api/v1/pull`` (CLI ``gnn pull`` parity)."""

    template: str
    source: str
    destination: str
    sha256: str
    dry_run: bool
    overwritten: bool
    copied: bool
    message: Optional[str] = None
    existing_sha256: Optional[str] = None
    exit_code: ParityExitCode


class ModelsResponse(BaseModel):
    """Payload for ``GET /api/v1/models`` (CLI ``gnn models`` parity)."""

    total_models: int
    query_ontology: Optional[str] = None
    matching_models: List[str]
    exit_code: ParityExitCode


class PreflightIssueModel(BaseModel):
    """One preflight check result."""

    category: str
    severity: str
    message: str
    fix: Optional[str] = None


class PreflightResponse(BaseModel):
    """Payload for ``POST /api/v1/preflight`` (CLI ``gnn preflight`` parity)."""

    checks_passed: int
    checks_failed: int
    is_ok: bool
    exit_code: ParityExitCode
    issues: List[PreflightIssueModel] = Field(default_factory=list)


class ReportResponse(BaseModel):
    """Payload for ``POST /api/v1/report`` (CLI ``gnn report`` parity)."""

    output_dir: str
    report_path: str
    report_chars: int
    exit_code: ParityExitCode


# ── Shared plumbing ──────────────────────────────────────────────────────────

_ResponseT = TypeVar("_ResponseT", bound=BaseModel)


def _sanitize_detail(exc: BaseException) -> str:
    """Redact absolute repository/home paths from backend error text.

    Mirrors the RED_TEAM V-09 posture of the job manager: keep the diagnostic
    value of the message while removing internal absolute paths.
    """
    message = str(exc)
    for secret in (str(get_repo_root()), str(Path.home())):
        if secret and secret in message:
            message = message.replace(secret, "<redacted>")
    return message[:500]


def _run_backend(operation: Callable[[], _ResponseT], *, command: str) -> _ResponseT:
    """Run one lazy backend operation, mapping failures to 400 bad_request.

    Deliberately raised :class:`HTTPException` guards pass through unchanged;
    every other backend/import/operation failure becomes a 400 bad_request
    envelope with a sanitized detail (CLI exit 1 in the pinned mapping).
    """
    try:
        return operation()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Parity command %s failed: %s", command, exc, exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"{command} failed: {_sanitize_detail(exc)}",
        ) from exc


def _resolve_client_path(
    path_value: str, *, purpose: str, create: bool = False
) -> Path:
    """Resolve a client-supplied path, mapping boundary violations to 400.

    Wraps :func:`gnn.api.path_utils.resolve_repo_path` (repository-local
    boundary, symlink rejection). Existence is checked by the callers so a
    missing file/directory can map to 404 distinctly from a boundary 400.
    """
    try:
        return resolve_repo_path(
            path_value,
            purpose=purpose,
            must_exist=False,
            must_be_dir=False,
            create=create,
        )
    except PathValidationError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err


def _require_input_file(path_value: str, *, purpose: str = "GNN file") -> Path:
    """Resolve a client-supplied file path; a missing file maps to 404."""
    resolved = _resolve_client_path(path_value, purpose=purpose)
    if not resolved.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"{purpose} not found or not a regular file: {path_value}",
        )
    return resolved


def _require_input_dir(path_value: str, *, purpose: str) -> Path:
    """Resolve a client-supplied directory path; a missing dir maps to 404."""
    resolved = _resolve_client_path(path_value, purpose=purpose)
    if not resolved.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"{purpose} not found: {path_value}",
        )
    return resolved


# ── Backend adapters (one per CLI command) ───────────────────────────────────


def _validate_gnn(gnn_file: Path, *, strict: bool) -> ValidateResponse:
    """Mirror ``gnn.cli._cmd_validate`` section/state-space/matrix checks."""
    from gnn.schema import (
        GNNParseError,
        parse_connections,
        parse_state_space,
        validate_matrix_dimensions,
        validate_required_sections,
    )
    from gnn.validation import validate_content

    content = gnn_file.read_text(encoding="utf-8")
    file_name = str(gnn_file)

    errors: List[GNNParseError] = []
    errors.extend(validate_required_sections(content, file_path=file_name))
    variables, variable_errors = parse_state_space(content, file_path=file_name)
    errors.extend(variable_errors)
    connections, connection_errors = parse_connections(
        content,
        known_variables={variable.name for variable in variables},
        file_path=file_name,
    )
    errors.extend(connection_errors)
    errors.extend(validate_matrix_dimensions(content, variables, file_path=file_name))

    semantic = validate_content(content)
    known_messages = {error.message for error in errors}
    errors.extend(
        GNNParseError(code="GNN-SEMANTIC", message=message, file=file_name)
        for message in semantic["errors"]
        if message not in known_messages
    )

    issues = [
        ParityIssue(
            code=error.code, message=error.message, line=error.line, file=error.file
        )
        for error in errors
    ]
    if issues and strict:
        # Pinned mapping: strict=True escalates a would-be exit 2 to 400 with
        # the errors in error.details.
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"{len(issues)} error(s) found",
                "errors": [issue.model_dump(mode="json") for issue in issues],
            },
        )
    exit_code: ParityExitCode = 2 if issues else 0
    return ValidateResponse(
        file=file_name,
        valid=not issues,
        exit_code=exit_code,
        errors=issues,
        variables_count=len(variables),
        connections_count=len(connections),
        semantic=semantic,
    )


def _parse_gnn(
    gnn_file: Path, output_format: ParseOutputFormat
) -> "ParseResponse | ParseSummary":
    """Mirror ``gnn.cli._cmd_parse`` parsing, frontmatter, and POMDP probes."""
    from gnn.schema import parse_connections, parse_state_space

    content = gnn_file.read_text(encoding="utf-8")
    file_name = str(gnn_file)

    variables, variable_errors = parse_state_space(content, file_path=file_name)
    connections, connection_errors = parse_connections(
        content,
        known_variables={variable.name for variable in variables},
        file_path=file_name,
    )
    parse_errors = [*variable_errors, *connection_errors]

    metadata: Dict[str, Any] = {}
    try:
        from gnn.parsers.frontmatter import parse_frontmatter

        metadata, _ = parse_frontmatter(content)
    except ImportError as exc:
        logger.debug("Frontmatter parsing not available: %s", exc)

    # POMDP extraction is best-effort, exactly like the CLI: non-POMDP files
    # or extraction failures omit the "pomdp" key and add a warning note.
    pomdp_note: Optional[str] = None
    pomdp_data: Optional[Dict[str, Any]] = None
    if not variables:
        pomdp_note = "File declares no StateSpaceBlock variables; 'pomdp' key omitted"
    else:
        try:
            from gnn.extract.pomdp_extractor import extract_pomdp_from_file

            pomdp_result = extract_pomdp_from_file(gnn_file, strict_validation=False)
            # on_error="lenient" (default) returns Optional[POMDPStateSpace];
            # "collect" returns (spec, errors) — tolerate both shapes.
            pomdp_space = (
                pomdp_result[0] if isinstance(pomdp_result, tuple) else pomdp_result
            )
            if pomdp_space is None:
                pomdp_note = "File does not parse as a POMDP; 'pomdp' key omitted"
            else:
                pomdp_data = pomdp_space.to_dict()
        except Exception as exc:
            logger.debug("POMDP extraction unavailable: %s", exc)
            pomdp_note = f"POMDP extraction unavailable: {exc}"

    warnings_list = [str(error) for error in parse_errors]
    if pomdp_note is not None:
        warnings_list.append(pomdp_note)

    serialized_errors = [
        ParityIssue(
            code=error.code, message=error.message, line=error.line, file=error.file
        )
        for error in parse_errors
    ]
    exit_code: ParityExitCode = 2 if parse_errors else 0

    if output_format == "summary":
        return ParseSummary(
            file=file_name,
            variables_count=len(variables),
            connections_count=len(connections),
            metadata_keys=sorted(str(key) for key in metadata),
            exit_code=exit_code,
        )

    # CLI-shaped result dict; the yaml output is rendered from it verbatim.
    variables_payload: List[Dict[str, Any]] = [
        {
            "name": variable.name,
            "dimensions": variable.dimensions,
            "dtype": variable.dtype,
            "default": variable.default,
        }
        for variable in variables
    ]
    connections_payload: List[Dict[str, Any]] = [
        {
            "source": connection.source,
            "target": connection.target,
            "directed": connection.directed,
            "label": connection.label,
            "line": connection.line,
        }
        for connection in connections
    ]
    result: Dict[str, Any] = {
        "file": file_name,
        "metadata": metadata,
        "variables": variables_payload,
        "connections": connections_payload,
        "warnings": warnings_list,
        "errors": [issue.model_dump(mode="json") for issue in serialized_errors],
    }
    if pomdp_data is not None:
        result["pomdp"] = pomdp_data

    yaml_text: Optional[str] = None
    if output_format == "yaml":
        try:
            import yaml
        except ImportError:
            # PyYAML is an optional dependency: degrade to JSON, never crash.
            logger.warning("PyYAML not installed; emitting JSON instead")
        else:
            yaml_text = str(
                yaml.safe_dump(result, default_flow_style=False, sort_keys=False)
            )

    return ParseResponse(
        file=file_name,
        format=output_format,
        metadata=metadata,
        variables=[ParseVariable(**entry) for entry in variables_payload],
        connections=[ParseConnection(**entry) for entry in connections_payload],
        warnings=warnings_list,
        errors=serialized_errors,
        exit_code=exit_code,
        pomdp=pomdp_data,
        yaml=yaml_text,
    )


def _extract_gnn(gnn_file: Path, *, strict: bool, compact: bool) -> ExtractResponse:
    """Mirror ``gnn.cli._cmd_extract`` payload and empty-skeleton checks."""
    from gnn.extract import extract_to_json

    payload = extract_to_json(gnn_file, strict_validation=strict, compact=compact)
    try:
        payload_obj: Any = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"extract produced unparseable output: {_sanitize_detail(exc)}",
        ) from exc
    if not isinstance(payload_obj, dict):
        raise HTTPException(
            status_code=400,
            detail="extract produced a non-object payload",
        )
    if payload_obj.get("status") == "error":
        raise HTTPException(
            status_code=400,
            detail={"message": "extraction failed", "payload": payload_obj},
        )
    if (
        not payload_obj.get("state_variables")
        and not payload_obj.get("matrices")
        and payload_obj.get("model_name") is None
    ):
        # Lenient extraction fabricates a default skeleton for files with no
        # GNN state-space content at all; surface that as an error like the CLI.
        raise HTTPException(
            status_code=400,
            detail=f"no POMDP state-space content found in {gnn_file}",
        )
    return ExtractResponse(
        file=str(gnn_file),
        strict=strict,
        compact=compact,
        exit_code=0,
        pomdp=payload_obj,
        payload_json=payload if compact else None,
    )


_RENDER_ARTIFACT_SUFFIXES: Dict[str, FrozenSet[str]] = {
    "pymdp": frozenset({".py"}),
    "jax": frozenset({".py"}),
    "numpyro": frozenset({".py"}),
    "pytorch": frozenset({".py"}),
    "discopy": frozenset({".py"}),
    "bnlearn": frozenset({".py"}),
    "rxinfer": frozenset({".jl", ".toml"}),
    "activeinference_jl": frozenset({".jl"}),
    "stan": frozenset({".stan"}),
}

_RENDER_EXCLUDED_FILES: FrozenSet[str] = frozenset(
    {"README.md", "processing_summary.json", "render_processing_summary.json"}
)


def _is_render_artifact(path: Path, framework: str) -> bool:
    """Return whether ``path`` is a candidate artifact for ``framework``."""
    suffixes = _RENDER_ARTIFACT_SUFFIXES.get(framework, frozenset())
    return (
        path.is_file()
        and (not suffixes or path.suffix in suffixes)
        and path.name not in _RENDER_EXCLUDED_FILES
    )


def _find_render_artifact(render_dir: Path, framework: str) -> Optional[Path]:
    """Locate the primary artifact of a single-framework render.

    Mirrors the CLI ``gnn render`` artifact discovery: prefer outputs declared
    in the render processing summary, then fall back to a framework-scoped
    recursive scan. Returns ``None`` when nothing was produced (reported as
    ``artifact: null`` rather than guessed).
    """
    summary_path = render_dir / "render_processing_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for result in summary.get("file_results", {}).values():
                framework_result = result.get("framework_results", {}).get(framework)
                if framework_result:
                    for item in framework_result.get("output_files", []):
                        candidate = Path(item)
                        if candidate.exists() and _is_render_artifact(
                            candidate, framework
                        ):
                            return candidate
                for item in result.get("generated_files", []):
                    candidate = Path(item)
                    if (
                        candidate.exists()
                        and framework in str(candidate)
                        and _is_render_artifact(candidate, framework)
                    ):
                        return candidate
        except (json.JSONDecodeError, OSError, TypeError):
            logger.debug("Could not parse render summary at %s", summary_path)

    candidates = sorted(
        path
        for path in render_dir.rglob("*")
        if framework in str(path.parent) and _is_render_artifact(path, framework)
    )
    return candidates[0] if candidates else None


def _render_gnn(gnn_file: Path, framework: str, output_dir: Path) -> RenderResponse:
    """Mirror ``gnn.cli._cmd_render`` single-framework render invocation."""
    from gnn.render import process_render

    with tempfile.TemporaryDirectory(prefix="gnn-api-render-") as temp_dir:
        input_dir = Path(temp_dir) / "input"
        input_dir.mkdir()
        shutil.copy2(gnn_file, input_dir / gnn_file.name)
        ok = process_render(
            target_dir=input_dir,
            output_dir=output_dir,
            verbose=False,
            frameworks=[framework],
            strict_validation=False,
            strict_framework_success=True,
        )
    if ok not in (True, 0):
        raise HTTPException(
            status_code=400,
            detail=f"Render failed for {gnn_file} using framework {framework}",
        )
    artifact = _find_render_artifact(output_dir, framework)
    return RenderResponse(
        file=str(gnn_file),
        framework=framework,
        output_dir=str(output_dir),
        artifact=str(artifact) if artifact is not None else None,
        exit_code=0,
    )


def _graph_gnn(gnn_file: Path, output_format: GraphOutputFormat) -> GraphResponse:
    """Mirror ``gnn.cli._cmd_graph`` dependency-graph generation."""
    from gnn.multimodel.dep_graph import render_graph_from_file

    output = render_graph_from_file(str(gnn_file), output_format=output_format)
    return GraphResponse(
        file=str(gnn_file),
        format=output_format,
        graph=output,
        exit_code=0,
    )


def _preflight(config_path: Optional[Path]) -> PreflightResponse:
    """Mirror ``gnn.cli._cmd_preflight`` severity classification."""
    from gnn.pipeline.preflight import run_preflight

    report = run_preflight(config_path=config_path)
    issues = [
        PreflightIssueModel(
            category=issue.category,
            severity=issue.severity,
            message=issue.message,
            fix=issue.fix,
        )
        for issue in report.issues
    ]
    if any(issue.severity == "error" for issue in issues):
        # Pinned mapping: preflight errors (CLI exit 1) become a 400 carrying
        # the issues in error.details.
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Preflight checks reported errors",
                "issues": [issue.model_dump(mode="json") for issue in issues],
            },
        )
    exit_code: ParityExitCode = (
        2 if any(issue.severity == "warning" for issue in issues) else 0
    )
    return PreflightResponse(
        checks_passed=report.checks_passed,
        checks_failed=report.checks_failed,
        is_ok=report.is_ok,
        exit_code=exit_code,
        issues=issues,
    )


def _report(output_dir: Path) -> ReportResponse:
    """Mirror ``gnn.cli._cmd_report`` atomic PIPELINE_REPORT.md publish."""
    from gnn.report.pipeline_report import generate_pipeline_report

    report = generate_pipeline_report(output_dir)
    report_path = output_dir / "PIPELINE_REPORT.md"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=report_path.parent, delete=False
    ) as tmp_file:
        tmp_file.write(report)
    os.replace(tmp_file.name, str(report_path))
    return ReportResponse(
        output_dir=str(output_dir),
        report_path=str(report_path),
        report_chars=len(report),
        exit_code=0,
    )


def _model_registry(target_path: Path, query_ontology: Optional[str]) -> ModelsResponse:
    """Mirror ``gnn.cli._cmd_models`` registry processing and filtering."""
    from gnn.model_registry import process_model_registry

    registry_dir = _resolve_client_path(
        MODEL_REGISTRY_OUTPUT_DIR,
        purpose="Registry output directory",
        create=True,
    )
    results = process_model_registry(
        target_dir=target_path,
        output_dir=registry_dir,
        query_ontology=query_ontology,
    )
    return ModelsResponse(
        total_models=results["total_models"],
        query_ontology=query_ontology,
        matching_models=results["matching_models"],
        exit_code=0,
    )


def _pull(
    name: str,
    output_dir: Path,
    *,
    dry_run: bool,
    overwrite: bool,
) -> PullResponse:
    """Mirror ``gnn.cli.templates.pull_template`` collision-aware copy."""
    from gnn.cli.templates import pull_template as pull_template_backend

    try:
        result = pull_template_backend(
            name, output_dir, dry_run=dry_run, overwrite=overwrite
        )
    except KeyError as exc:
        # Mirrors the templates show route: unknown template names are 404.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileExistsError as exc:
        # The message already suggests passing overwrite to replace it.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return PullResponse(**result, exit_code=0)


# ── Route registration (both FastAPI surfaces) ───────────────────────────────


def register_parity_routes(app: FastAPI) -> None:
    """Register all CLI-parity routes on ``app``.

    Called by both ``gnn.api.app.create_app`` and ``gnn.api.server.create_app``
    so ``gnn serve`` and ``python -m gnn.api.server`` expose identical parity
    routes under the canonical ``{status, data, error, meta}`` envelope.
    """

    @app.post("/api/v1/validate", response_model=APIEnvelope, tags=["Validate"])
    def validate_gnn_file(request: ValidateRequest) -> APIEnvelope:
        """Validate one GNN file (CLI ``gnn validate`` parity)."""
        gnn_file = _require_input_file(request.file_path)
        response = _run_backend(
            lambda: _validate_gnn(gnn_file, strict=request.strict),
            command="validate",
        )
        return success_envelope(response.model_dump(mode="json"), endpoint="validate")

    @app.post("/api/v1/parse", response_model=APIEnvelope, tags=["Parse"])
    def parse_gnn_file(request: ParseRequest) -> APIEnvelope:
        """Parse one GNN file as JSON, YAML, or summary (CLI ``gnn parse`` parity)."""
        gnn_file = _require_input_file(request.file_path)
        response = _run_backend(
            lambda: _parse_gnn(gnn_file, request.format),
            command="parse",
        )
        return success_envelope(response.model_dump(mode="json"), endpoint="parse")

    @app.post("/api/v1/extract", response_model=APIEnvelope, tags=["Extract"])
    def extract_gnn_file(request: ExtractRequest) -> APIEnvelope:
        """Extract a POMDP state space (CLI ``gnn extract`` parity)."""
        gnn_file = _require_input_file(request.file_path)
        response = _run_backend(
            lambda: _extract_gnn(
                gnn_file, strict=request.strict, compact=request.compact
            ),
            command="extract",
        )
        return success_envelope(response.model_dump(mode="json"), endpoint="extract")

    @app.post("/api/v1/render", response_model=APIEnvelope, tags=["Render"])
    def render_gnn_file(request: RenderRequest) -> APIEnvelope:
        """Render one GNN file to framework code (CLI ``gnn render`` parity)."""
        gnn_file = _require_input_file(request.file_path)
        if request.output_dir is not None:
            output_dir = _resolve_client_path(
                request.output_dir,
                purpose="Output directory",
                create=True,
            )
        else:
            output_dir = _resolve_client_path(
                f"output/11_render_output/{gnn_file.stem}",
                purpose="Output directory",
                create=True,
            )
        response = _run_backend(
            lambda: _render_gnn(gnn_file, request.framework, output_dir),
            command="render",
        )
        return success_envelope(response.model_dump(mode="json"), endpoint="render")

    @app.post("/api/v1/graph", response_model=APIEnvelope, tags=["Graph"])
    def graph_gnn_file(request: GraphRequest) -> APIEnvelope:
        """Generate a dependency graph (CLI ``gnn graph`` parity)."""
        gnn_file = _require_input_file(request.file_path)
        response = _run_backend(
            lambda: _graph_gnn(gnn_file, request.format),
            command="graph",
        )
        return success_envelope(response.model_dump(mode="json"), endpoint="graph")

    @app.get("/api/v1/templates", response_model=APIEnvelope, tags=["Templates"])
    def list_gnn_templates() -> APIEnvelope:
        """List maintained templates (CLI ``gnn templates list`` parity)."""
        from gnn.cli.templates import list_templates

        def _operate() -> TemplatesResponse:
            records = [TemplateRecordModel(**entry) for entry in list_templates()]
            return TemplatesResponse(templates=records, total=len(records))

        response = _run_backend(_operate, command="templates")
        return success_envelope(response.model_dump(mode="json"), endpoint="templates")

    @app.get("/api/v1/templates/{name}", response_model=APIEnvelope, tags=["Templates"])
    def show_gnn_template(name: str) -> APIEnvelope:
        """Show one maintained template (CLI ``gnn templates show`` parity)."""
        from gnn.cli.templates import show_template

        try:
            record = show_template(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        def _operate() -> TemplateResponse:
            return TemplateResponse(template=TemplateRecordModel(**record))

        response = _run_backend(_operate, command="templates")
        return success_envelope(response.model_dump(mode="json"), endpoint="templates")

    @app.post("/api/v1/pull", response_model=APIEnvelope, tags=["Templates"])
    def pull_gnn_template(request: PullRequest) -> APIEnvelope:
        """Pull a maintained template into an output directory (CLI ``gnn pull`` parity)."""
        output_dir = _resolve_client_path(
            request.output_dir or "input/gnn_files",
            purpose="Pull output directory",
        )
        response = _run_backend(
            lambda: _pull(
                request.name,
                output_dir,
                dry_run=request.dry_run,
                overwrite=request.overwrite,
            ),
            command="pull",
        )
        return success_envelope(response.model_dump(mode="json"), endpoint="pull")

    @app.get("/api/v1/models", response_model=APIEnvelope, tags=["Models"])
    def list_models(
        target_dir: str = "input/gnn_files",
        query_ontology: Optional[str] = None,
    ) -> APIEnvelope:
        """Query the model registry (CLI ``gnn models`` parity)."""
        target_path = _require_input_dir(target_dir, purpose="Target directory")
        response = _run_backend(
            lambda: _model_registry(target_path, query_ontology),
            command="models",
        )
        return success_envelope(response.model_dump(mode="json"), endpoint="models")

    @app.post("/api/v1/preflight", response_model=APIEnvelope, tags=["Preflight"])
    def run_preflight_checks(request: PreflightRequest) -> APIEnvelope:
        """Run environment & config checks (CLI ``gnn preflight`` parity)."""
        config_path: Optional[Path] = None
        if request.config is not None:
            config_path = _require_input_file(request.config, purpose="Config file")
        response = _run_backend(
            lambda: _preflight(config_path),
            command="preflight",
        )
        return success_envelope(response.model_dump(mode="json"), endpoint="preflight")

    @app.post("/api/v1/report", response_model=APIEnvelope, tags=["Report"])
    def generate_report(request: ReportRequest) -> APIEnvelope:
        """Generate PIPELINE_REPORT.md (CLI ``gnn report`` parity)."""
        output_dir = _require_input_dir(request.output_dir, purpose="Output directory")
        response = _run_backend(
            lambda: _report(output_dir),
            command="report",
        )
        return success_envelope(response.model_dump(mode="json"), endpoint="report")


__all__ = ["register_parity_routes"]
