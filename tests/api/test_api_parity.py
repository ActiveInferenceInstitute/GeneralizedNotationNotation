#!/usr/bin/env python3
"""Behavior tests for the CLI-parity API endpoints (``src/gnn/api/parity.py``).

Every behavioral test runs against BOTH FastAPI surfaces via the parametrized
``client`` fixture — ``gnn.api.app.create_app`` (the ``gnn serve`` surface)
and ``gnn.api.server.create_app`` (the ``python -m gnn.api.server`` surface) —
and a parity-presence test asserts the two surfaces register an identical set
of parity routes.

Pinned exit-code → HTTP mapping (see the module docstring of
``src/gnn/api/parity.py``; mirrors the strict ``pipeline_exit_succeeded``
policy in ``gnn.api.pipeline_runner``):

- CLI exit 0                        → 200 success envelope, data.exit_code == 0
- CLI exit 2 (warnings)             → 200 success envelope, data.exit_code == 2,
                                      warnings/errors listed in data
- would-be exit 2 under strict=True → 400 bad_request, errors in error.details
- missing file/directory            → 404 not_found
- path-boundary violation           → 400 bad_request
- backend/import/operation failure  → 400 bad_request with sanitized detail
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

import pytest
from fastapi.testclient import TestClient

from gnn.api.app import create_app as create_run_app
from gnn.api.server import app as server_module_app
from gnn.api.server import create_app as create_job_app

#: Committed exemplar with a full valid GNN model (11 variables, 10
#: connections) — positive-case input for validate/parse/extract/render/graph.
VALID_EXEMPLAR = "input/gnn_files/discrete/simple_mdp.md"

#: Committed fixture missing required sections → validate exit 2 / strict 400.
INVALID_EXEMPLAR = "tests/api/fixtures/invalid_gnn_missing_sections.md"

#: Committed fixture with all sections but an undeclared connection target
#: (`A>Z`) → parse exit 2 and validate exit 2.
PARSE_WARNING_EXEMPLAR = "tests/api/fixtures/parse_warning_unknown_variable.md"

#: The exact parity route surface that must exist on BOTH FastAPI factories.
PARITY_ROUTES: Dict[str, Set[str]] = {
    "/api/v1/validate": {"POST"},
    "/api/v1/parse": {"POST"},
    "/api/v1/extract": {"POST"},
    "/api/v1/render": {"POST"},
    "/api/v1/graph": {"POST"},
    "/api/v1/templates": {"GET"},
    "/api/v1/templates/{name}": {"GET"},
    "/api/v1/models": {"GET"},
    "/api/v1/preflight": {"POST"},
    "/api/v1/report": {"POST"},
}


def _assert_envelope(payload: Any, status: str = "success") -> Dict[str, Any]:
    """Assert and return the canonical API response envelope."""
    assert set(payload) == {"status", "data", "error", "meta"}
    assert isinstance(payload, dict)
    assert payload["status"] == status
    assert isinstance(payload["data"], dict)
    assert isinstance(payload["meta"], dict)
    assert "timestamp" in payload["meta"]
    if status == "success":
        assert payload["error"] is None
    else:
        assert isinstance(payload["error"], dict)
        assert payload["error"]["code"]
        assert payload["error"]["message"]
    return payload


@pytest.fixture(autouse=True)
def disable_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep parity checks independent from global limiter history."""
    monkeypatch.setenv("GNN_RATE_LIMIT", "0")


@pytest.fixture(params=["app_factory", "server_factory"])
def client(request: pytest.FixtureRequest) -> Iterator[TestClient]:
    """Serve every behavior test against both FastAPI surfaces."""
    factory: Callable[[], Any] = (
        create_run_app if request.param == "app_factory" else create_job_app
    )
    yield TestClient(factory())


# ── Route parity across surfaces ─────────────────────────────────────────────


def _route_table(app: Any) -> Dict[str, Set[str]]:
    """Collect path → method-set for every HTTP route on ``app``."""
    table: Dict[str, Set[str]] = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path and methods:
            table.setdefault(path, set()).update(methods)
    return table


@pytest.mark.unit
def test_parity_routes_present_on_both_surfaces() -> None:
    """Every parity route must exist with identical methods on both surfaces."""
    run_app_table = _route_table(create_run_app())
    job_app_table = _route_table(create_job_app())
    module_app_table = _route_table(server_module_app)
    for path, methods in PARITY_ROUTES.items():
        for surface, table in (
            ("gnn.api.app", run_app_table),
            ("gnn.api.server factory", job_app_table),
            ("gnn.api.server module app", module_app_table),
        ):
            assert table.get(path) == methods, f"{path} missing or wrong on {surface}"


# ── POST /api/v1/validate ────────────────────────────────────────────────────


@pytest.mark.unit
def test_validate_happy_path_returns_exit_code_0(client: TestClient) -> None:
    """A fully valid exemplar validates cleanly: 200, exit_code 0, counts."""
    response = client.post("/api/v1/validate", json={"file_path": VALID_EXEMPLAR})
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    assert envelope["data"]["exit_code"] == 0
    assert envelope["data"]["valid"] is True
    assert envelope["data"]["errors"] == []
    assert envelope["data"]["variables_count"] == 11
    assert envelope["data"]["connections_count"] == 10


@pytest.mark.unit
def test_validate_non_strict_warnings_map_to_exit_code_2(
    client: TestClient,
) -> None:
    """Missing required sections: 200 success envelope carrying exit_code 2."""
    response = client.post("/api/v1/validate", json={"file_path": INVALID_EXEMPLAR})
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    assert envelope["data"]["exit_code"] == 2
    assert envelope["data"]["valid"] is False
    errors = envelope["data"]["errors"]
    assert errors
    assert all("code" in error and "message" in error for error in errors)
    assert any("Missing required section" in error["message"] for error in errors)


@pytest.mark.unit
def test_validate_strict_escalates_warnings_to_400(client: TestClient) -> None:
    """strict=True escalates a would-be exit 2 to a 400 bad_request envelope."""
    response = client.post(
        "/api/v1/validate",
        json={"file_path": INVALID_EXEMPLAR, "strict": True},
    )
    assert response.status_code == 400
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "bad_request"
    details = envelope["error"]["details"]
    assert isinstance(details, dict)
    assert details["errors"]
    assert all("message" in error for error in details["errors"])


@pytest.mark.unit
def test_validate_missing_file_maps_to_404(client: TestClient) -> None:
    """A nonexistent repository-local file mirrors _guard_input_file as 404."""
    response = client.post(
        "/api/v1/validate",
        json={"file_path": "input/gnn_files/no_such_model.md"},
    )
    assert response.status_code == 404
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "not_found"
    assert "not found" in envelope["error"]["message"]


@pytest.mark.unit
def test_validate_out_of_repo_path_maps_to_400(client: TestClient) -> None:
    """Client paths outside the repository boundary are rejected as 400."""
    response = client.post("/api/v1/validate", json={"file_path": "/etc/passwd"})
    assert response.status_code == 400
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "bad_request"
    assert "repository root" in envelope["error"]["message"]


@pytest.mark.unit
def test_validate_rejects_unknown_request_fields(client: TestClient) -> None:
    """Request models forbid extra fields explicitly (no silent ignoring)."""
    response = client.post(
        "/api/v1/validate",
        json={"file_path": VALID_EXEMPLAR, "unexpected": True},
    )
    assert response.status_code == 422
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "validation_error"


# ── POST /api/v1/parse ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_json_happy_path_returns_structure(client: TestClient) -> None:
    """Parsing a valid exemplar returns variables, connections, exit_code 0."""
    response = client.post(
        "/api/v1/parse", json={"file_path": VALID_EXEMPLAR, "format": "json"}
    )
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["exit_code"] == 0
    assert data["format"] == "json"
    assert len(data["variables"]) == 11
    assert len(data["connections"]) == 10
    assert data["variables"][0]["name"]
    assert data["connections"][0]["source"]


@pytest.mark.unit
def test_parse_summary_format_returns_counts(client: TestClient) -> None:
    """Summary format mirrors the CLI's File/Variables/Connections report."""
    response = client.post(
        "/api/v1/parse", json={"file_path": VALID_EXEMPLAR, "format": "summary"}
    )
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["exit_code"] == 0
    assert data["variables_count"] == 11
    assert data["connections_count"] == 10
    assert isinstance(data["metadata_keys"], list)


@pytest.mark.unit
def test_parse_yaml_format_embeds_yaml_text(client: TestClient) -> None:
    """YAML format carries the structured payload plus rendered YAML text."""
    response = client.post(
        "/api/v1/parse", json={"file_path": VALID_EXEMPLAR, "format": "yaml"}
    )
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["format"] == "yaml"
    if importlib.util.find_spec("yaml") is not None:
        assert data["yaml"] is not None
        assert "variables:" in data["yaml"]
    else:
        # PyYAML absent: the endpoint degrades to the structured payload
        # exactly like the CLI; yaml_text stays None (the envelope stays
        # structured either way — see parity.py).
        assert data["yaml"] is None


@pytest.mark.unit
def test_parse_unknown_connection_maps_to_exit_code_2(client: TestClient) -> None:
    """An undeclared connection target is a warning outcome: 200, exit 2."""
    response = client.post("/api/v1/parse", json={"file_path": PARSE_WARNING_EXEMPLAR})
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["exit_code"] == 2
    assert data["warnings"]
    assert any("Z" in warning for warning in data["warnings"])
    assert data["errors"]


@pytest.mark.unit
def test_parse_missing_file_maps_to_404(client: TestClient) -> None:
    """A nonexistent file maps to 404 before any parsing happens."""
    response = client.post(
        "/api/v1/parse", json={"file_path": "input/gnn_files/no_such_model.md"}
    )
    assert response.status_code == 404
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "not_found"


# ── POST /api/v1/extract ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_extract_happy_path_returns_pomdp_payload(client: TestClient) -> None:
    """Extracting the exemplar returns the POMDP payload with exit_code 0."""
    response = client.post("/api/v1/extract", json={"file_path": VALID_EXEMPLAR})
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["exit_code"] == 0
    assert data["strict"] is True
    assert data["pomdp"]["model_name"]


@pytest.mark.unit
def test_extract_contentless_file_maps_to_400(client: TestClient) -> None:
    """A file with no POMDP content fails explicitly (CLI exit 1 → 400)."""
    response = client.post(
        "/api/v1/extract", json={"file_path": "input/gnn_files/INDEX.md"}
    )
    assert response.status_code == 400
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "bad_request"
    assert "no POMDP state-space content" in envelope["error"]["message"]


# ── POST /api/v1/render ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_render_happy_path_produces_artifact(client: TestClient) -> None:
    """Rendering the exemplar with pymdp yields 200 and a real artifact."""
    output_dir = "output/api_parity_render"
    shutil.rmtree(Path(output_dir), ignore_errors=True)
    try:
        response = client.post(
            "/api/v1/render",
            json={
                "file_path": VALID_EXEMPLAR,
                "framework": "pymdp",
                "output_dir": output_dir,
            },
        )
        assert response.status_code == 200
        envelope = _assert_envelope(response.json())
        data = envelope["data"]
        assert data["exit_code"] == 0
        assert data["framework"] == "pymdp"
        assert data["artifact"] is not None
        assert data["artifact"].endswith(".py")
        assert Path(data["artifact"]).is_file()
    finally:
        shutil.rmtree(Path(output_dir), ignore_errors=True)


@pytest.mark.unit
def test_render_rejects_unknown_framework_with_422(client: TestClient) -> None:
    """Frameworks outside the CLI choice set fail request validation."""
    response = client.post(
        "/api/v1/render",
        json={"file_path": VALID_EXEMPLAR, "framework": "matplotlib"},
    )
    assert response.status_code == 422
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "validation_error"


# ── POST /api/v1/graph ───────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("output_format", ["mermaid", "text"])
def test_graph_happy_path_returns_graph(client: TestClient, output_format: str) -> None:
    """Both graph formats return the rendered graph with exit_code 0."""
    response = client.post(
        "/api/v1/graph",
        json={"file_path": VALID_EXEMPLAR, "format": output_format},
    )
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["exit_code"] == 0
    assert data["format"] == output_format
    assert data["graph"]
    if output_format == "mermaid":
        assert data["graph"].startswith("graph TD")


# ── GET /api/v1/templates ────────────────────────────────────────────────────


@pytest.mark.unit
def test_templates_list_returns_records(client: TestClient) -> None:
    """Listing templates returns checksummed records."""
    response = client.get("/api/v1/templates")
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["total"] == len(data["templates"])
    assert data["total"] >= 3
    first = data["templates"][0]
    assert first["name"] and len(first["sha256"]) == 64


@pytest.mark.unit
def test_templates_show_known_name(client: TestClient) -> None:
    """Showing a maintained template returns its full record."""
    response = client.get("/api/v1/templates/actinf-pomdp-2state")
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    template = envelope["data"]["template"]
    assert template["name"] == "actinf-pomdp-2state"
    assert len(template["sha256"]) == 64


@pytest.mark.unit
def test_templates_show_unknown_name_maps_to_404(client: TestClient) -> None:
    """An unknown template name mirrors the CLI KeyError as 404."""
    response = client.get("/api/v1/templates/no-such-template")
    assert response.status_code == 404
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "not_found"
    assert "Unknown template" in envelope["error"]["message"]


# ── GET /api/v1/models ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_models_happy_path_returns_registry_counts(client: TestClient) -> None:
    """Registry processing over a committed directory returns model counts."""
    response = client.get(
        "/api/v1/models", params={"target_dir": "input/gnn_files/discrete"}
    )
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["exit_code"] == 0
    assert data["total_models"] >= 1
    assert isinstance(data["matching_models"], list)


@pytest.mark.unit
def test_models_ontology_query_filters_matches(client: TestClient) -> None:
    """An ontology query with no hits returns an empty matching list."""
    response = client.get(
        "/api/v1/models",
        params={
            "target_dir": "input/gnn_files/discrete",
            "query_ontology": "no-such-ontology-term-xyz",
        },
    )
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    assert envelope["data"]["matching_models"] == []
    assert envelope["data"]["query_ontology"] == "no-such-ontology-term-xyz"


@pytest.mark.unit
def test_models_missing_target_dir_maps_to_404(client: TestClient) -> None:
    """A nonexistent target directory maps to 404 before processing."""
    response = client.get(
        "/api/v1/models", params={"target_dir": "output/no_such_target_dir"}
    )
    assert response.status_code == 404
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "not_found"


# ── POST /api/v1/preflight ───────────────────────────────────────────────────


def _fake_preflight_report(severities: List[str]) -> Callable[[Optional[Path]], Any]:
    """Build a run_preflight stub returning issues with the given severities."""
    from gnn.pipeline.preflight import PreflightIssue, PreflightReport

    def fake_run_preflight(config_path: Optional[Path] = None) -> Any:
        return PreflightReport(
            issues=[
                PreflightIssue(
                    category="dependency",
                    severity=severity,
                    message=f"synthetic {severity} issue",
                )
                for severity in severities
            ],
            checks_passed=5,
            checks_failed=sum(severity == "error" for severity in severities),
        )

    return fake_run_preflight


@pytest.mark.unit
def test_preflight_warnings_map_to_exit_code_2(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warning-severity preflight issues: 200 success envelope, exit_code 2."""
    monkeypatch.setattr(
        "gnn.pipeline.preflight.run_preflight",
        _fake_preflight_report(["warning", "info"]),
    )
    response = client.post("/api/v1/preflight", json={})
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["exit_code"] == 2
    assert data["is_ok"] is True
    assert any(issue["severity"] == "warning" for issue in data["issues"])


@pytest.mark.unit
def test_preflight_errors_map_to_400_with_issue_details(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Error-severity preflight issues (CLI exit 1) become a 400 envelope."""
    monkeypatch.setattr(
        "gnn.pipeline.preflight.run_preflight",
        _fake_preflight_report(["error", "warning"]),
    )
    response = client.post("/api/v1/preflight", json={})
    assert response.status_code == 400
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "bad_request"
    details = envelope["error"]["details"]
    assert isinstance(details, dict)
    assert any(issue["severity"] == "error" for issue in details["issues"])


@pytest.mark.unit
def test_preflight_real_call_returns_canonical_envelope(
    client: TestClient,
) -> None:
    """Wiring smoke against the real backend.

    The real environment's dependency state decides the branch (missing
    optional deps here produce error issues → 400; a fully provisioned env
    returns 200 with exit_code 0 or 2) — either way the canonical envelope
    contract must hold.
    """
    response = client.post("/api/v1/preflight", json={})
    envelope = _assert_envelope(
        response.json(), status="success" if response.status_code == 200 else "error"
    )
    if response.status_code == 200:
        assert envelope["data"]["exit_code"] in (0, 2)
    else:
        assert response.status_code == 400
        assert envelope["error"]["code"] == "bad_request"


# ── POST /api/v1/report ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_report_happy_path_writes_pipeline_report(client: TestClient) -> None:
    """Reporting over an existing directory writes PIPELINE_REPORT.md."""
    output_dir = Path("output/api_parity_report")
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    try:
        response = client.post("/api/v1/report", json={"output_dir": str(output_dir)})
        assert response.status_code == 200
        envelope = _assert_envelope(response.json())
        data = envelope["data"]
        assert data["exit_code"] == 0
        assert data["report_path"].endswith("PIPELINE_REPORT.md")
        assert data["report_chars"] > 0
        assert Path(data["report_path"]).is_file()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


@pytest.mark.unit
def test_report_missing_dir_maps_to_404(client: TestClient) -> None:
    """A nonexistent output directory maps to 404 (CLI exit 1 → not_found)."""
    response = client.post(
        "/api/v1/report",
        json={"output_dir": "output/no_such_report_dir"},
    )
    assert response.status_code == 404
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "not_found"
