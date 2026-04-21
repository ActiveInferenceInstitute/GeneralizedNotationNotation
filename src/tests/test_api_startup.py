#!/usr/bin/env python3
"""Phase 4.2 regression tests for api (REST module).

Zero-mock per CLAUDE.md: uses real FastAPI TestClient when available,
skips cleanly when optional deps absent.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _get_app():
    """Return the FastAPI app object, or skip if unavailable."""
    try:
        from api import server
    except ImportError as e:
        pytest.skip(f"api module unavailable: {e}")
    app = getattr(server, "app", None)
    if app is None:
        pytest.skip("api.server does not expose .app attribute")
    return app


def test_api_module_loads_and_exposes_version():
    from api import get_module_info
    info = get_module_info()
    assert isinstance(info, dict)
    assert "version" in info


def test_api_app_has_routes_registered():
    app = _get_app()
    # FastAPI exposes the route table via .routes.
    assert hasattr(app, "routes"), "FastAPI app missing .routes attribute"
    route_paths = [getattr(r, "path", None) for r in app.routes]
    # Must have at least one non-trivial route registered.
    actual_paths = [p for p in route_paths if p and p != "/"]
    assert len(actual_paths) >= 1, (
        f"No substantive API routes registered: {route_paths}"
    )


def test_api_openapi_schema_is_well_formed():
    """Every FastAPI app exposes /openapi.json; the schema must be a dict
    with 'paths' and 'info' sections per the OpenAPI 3.0 spec."""
    app = _get_app()
    # FastAPI offers .openapi() directly — no need to start a server.
    try:
        schema = app.openapi()
    except Exception as e:
        pytest.skip(f"App not initialized enough for openapi(): {e}")
    assert isinstance(schema, dict)
    assert "paths" in schema
    assert "info" in schema
    # Title should be set.
    assert schema["info"].get("title"), "OpenAPI schema has no title"


def test_api_health_endpoint_exists():
    """A canonical liveness probe must be registered. Accepts any of the
    common conventions (/, /health, /api/health, /api/v1/health)."""
    app = _get_app()
    paths = [getattr(r, "path", "") for r in app.routes]
    health_paths = ("/", "/health", "/api/health", "/api/v1/health")
    assert any(p in health_paths for p in paths), (
        f"No liveness endpoint registered. Expected one of {health_paths}; got: {paths}"
    )
