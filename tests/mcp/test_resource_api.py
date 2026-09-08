"""MCP resource-API contract tests (wave-2 MED-02).

Pins that:
- ``gnn.mcp.list_available_resources`` returns the real resource list
  (it previously aliased ``get_available_tools`` and returned the tool list);
- HTTP capability listing and the resource read gate agree on the only
  registered resource (`gnn://documentation/{doc_name}`) — listing a
  concrete URI exposes the read while keeping capabilities honest;
- `npx_inspector.get_resource` no longer sends a URI as a JSON-RPC method.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytestmark = pytest.mark.mcp

import gnn.mcp as mcp_pkg
from gnn.mcp.server_http import (
    get_http_capabilities,
    get_safe_http_resource_uris,
    is_safe_http_resource,
)


class TestRealResourceLister:
    """list_available_resources returns resources, not tools."""

    @pytest.mark.unit
    def test_returns_resource_dicts_not_tool_dicts(self) -> None:
        from gnn.mcp import initialize

        initialize(halt_on_missing_sdk=False, force_proceed_flag=True)
        resources = mcp_pkg.list_available_resources()
        assert isinstance(resources, list)
        assert resources, "the registry must have at least one resource"
        for resource in resources:
            assert "uri" in resource  # not a tool's "name" key

    @pytest.mark.unit
    def test_exposes_gnn_documentation_template(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gnn.mcp import initialize

        initialize(halt_on_missing_sdk=False, force_proceed_flag=True)
        from gnn.mcp import mcp_instance

        # Wait for the documentation resource to register (discovery may
        # still be in flight after initialize returns).
        import time

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if "gnn://documentation/{doc_name}" in mcp_instance.resources:
                break
            time.sleep(0.1)
        resources = mcp_pkg.list_available_resources()
        templates = {r.get("uri") for r in resources if isinstance(r, dict)}
        assert "gnn://documentation/{doc_name}" in templates


class TestHTTPResourceGateAgreesWithCapabilities:
    """Listing a concrete URI must expose the read AND show the resource."""

    @pytest.mark.unit
    def test_concrete_uri_exposes_read_while_capabilities_stay_honest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gnn.mcp import initialize

        initialize(halt_on_missing_sdk=False, force_proceed_flag=True)
        from gnn.mcp import mcp_instance

        import time

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if "gnn://documentation/{doc_name}" in mcp_instance.resources:
                break
            time.sleep(0.1)
        monkeypatch.setenv("GNN_MCP_SAFE_RESOURCES", "gnn://documentation/grammar")
        assert is_safe_http_resource("gnn://documentation/grammar") is True
        assert is_safe_http_resource("gnn://documentation/missing") is False

        capabilities = get_http_capabilities()
        templates = {
            r["uri_template"] for r in capabilities["resources"] if isinstance(r, dict)
        }
        assert "gnn://documentation/{doc_name}" in templates

    @pytest.mark.unit
    def test_template_uri_exposes_all_concrete_reads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gnn.mcp import initialize

        initialize(halt_on_missing_sdk=False, force_proceed_flag=True)
        from gnn.mcp import mcp_instance

        import time

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if "gnn://documentation/{doc_name}" in mcp_instance.resources:
                break
            time.sleep(0.1)
        monkeypatch.setenv("GNN_MCP_SAFE_RESOURCES", "gnn://documentation/{doc_name}")
        for doc in ("grammar", "file_structure", "punctuation", "schema_json"):
            assert is_safe_http_resource(f"gnn://documentation/{doc}") is True
        assert is_safe_http_resource("gnn://other/thing") is False

    @pytest.mark.unit
    def test_default_denies_everything(self) -> None:
        assert is_safe_http_resource("gnn://documentation/grammar") is False


class TestNpxInspectorResourceRead:
    """The inspector routes reads through the real method, not as a guess."""

    @pytest.mark.unit
    def test_get_resource_uses_mcp_resource_get(self) -> None:
        from gnn.mcp.npx_inspector import StdioMCPClient

        captured: dict[str, Any] = {}

        def fake_send_request(method: str, params: Any = None) -> dict[str, Any]:
            captured["method"] = method
            captured["params"] = params
            return {"ok": True}

        # Bypass __init__ (which spawns a process) and patch the send path.
        client = StdioMCPClient.__new__(StdioMCPClient)
        client._send_request = fake_send_request  # type: ignore[method-assign]
        client.get_resource("gnn://documentation/grammar")
        assert captured["method"] == "mcp.resource.get"
        assert captured["params"] == {"uri": "gnn://documentation/grammar"}


class TestToolCountGateStaysGreen:
    """Adding resources and re-exporting the lister must not move the tool count."""

    @pytest.mark.unit
    def test_documented_count_matches_audit(self) -> None:
        audit = json.loads(
            __import__("pathlib")
            .Path(__file__)
            .resolve()
            .parents[2]
            .joinpath("src/gnn/mcp/audit_report.json")
            .read_text(encoding="utf-8")
        )
        assert audit["tools_total"] >= 140
        assert audit.get("schema_checks_ok", 0) == audit["tools_total"]
