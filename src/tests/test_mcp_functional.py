#!/usr/bin/env python3
"""
Functional tests for the MCP (Model Context Protocol) module.

Tests MCP tool registration, MCPServer lifecycle, tool discovery,
tool execution, resource management, and server status.
"""

import pytest
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.mcp import (
    MCP,
    MCPTool,
    MCPResource,
    MCPServer,
    MCPPerformanceMetrics,
    MCPModuleInfo,
    create_mcp_server,
)
from mcp.exceptions import (
    MCPToolNotFoundError,
    MCPInvalidParamsError,
    MCPToolExecutionError,
    MCPValidationError,
)


@pytest.fixture
def mcp_instance():
    """Create a fresh MCP instance for each test."""
    return MCP(enable_caching=False, enable_rate_limiting=False)


@pytest.fixture
def sample_tool_func():
    """A simple callable for tool registration."""
    def add(a: int, b: int) -> dict:
        return {"result": a + b}
    return add


class TestMCPToolRegistration:
    """Test registering tools with the MCP server."""

    @pytest.mark.unit
    def test_register_basic_tool(self, mcp_instance, sample_tool_func):
        """Should register a tool and make it discoverable."""
        mcp_instance.register_tool(
            name="test_add",
            func=sample_tool_func,
            schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            description="Add two numbers",
        )
        assert "test_add" in mcp_instance.tools
        assert mcp_instance.tools["test_add"].description == "Add two numbers"

    @pytest.mark.unit
    def test_register_tool_with_metadata(self, mcp_instance, sample_tool_func):
        """Should store category, version, and tags."""
        mcp_instance.register_tool(
            name="fancy_tool",
            func=sample_tool_func,
            schema={},
            description="A fancy tool",
            category="math",
            version="2.0.0",
            tags=["experimental"],
        )
        tool = mcp_instance.tools["fancy_tool"]
        assert tool.category == "math"
        assert tool.version == "2.0.0"
        assert "experimental" in tool.tags

    @pytest.mark.unit
    def test_register_tool_overwrites(self, mcp_instance, sample_tool_func):
        """Registering a tool with the same name should overwrite."""
        mcp_instance.register_tool(name="dup", func=sample_tool_func, schema={}, description="v1")
        mcp_instance.register_tool(name="dup", func=sample_tool_func, schema={}, description="v2")
        assert mcp_instance.tools["dup"].description == "v2"

    @pytest.mark.unit
    def test_register_tool_requires_name(self, mcp_instance, sample_tool_func):
        """Should raise error if name is empty."""
        with pytest.raises(MCPInvalidParamsError):
            mcp_instance.register_tool(name="", func=sample_tool_func, schema={}, description="test")

    @pytest.mark.unit
    def test_register_tool_requires_callable(self, mcp_instance):
        """Should raise error if func is not callable."""
        with pytest.raises(MCPInvalidParamsError):
            mcp_instance.register_tool(name="bad", func="not_a_function", schema={}, description="test")


class TestMCPToolExecution:
    """Test executing registered tools."""

    @pytest.mark.unit
    def test_execute_tool(self, mcp_instance, sample_tool_func):
        """Should execute a registered tool and return its result."""
        mcp_instance.register_tool(
            name="add",
            func=sample_tool_func,
            schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            description="Add numbers",
        )
        result = mcp_instance.execute_tool("add", {"a": 3, "b": 4})
        assert result == {"result": 7}

    @pytest.mark.unit
    def test_execute_nonexistent_tool(self, mcp_instance):
        """Should raise MCPToolNotFoundError for unknown tool."""
        with pytest.raises(MCPToolNotFoundError):
            mcp_instance.execute_tool("nonexistent", {})

    @pytest.mark.unit
    def test_execute_tool_with_missing_required_param(self, mcp_instance, sample_tool_func):
        """Should raise error when required parameter is missing."""
        mcp_instance.register_tool(
            name="strict_add",
            func=sample_tool_func,
            schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            description="Strict add",
        )
        with pytest.raises(MCPInvalidParamsError):
            mcp_instance.execute_tool("strict_add", {"a": 1})


class TestMCPResourceRegistration:
    """Test resource registration and retrieval."""

    @pytest.mark.unit
    def test_register_resource(self, mcp_instance):
        """Should register a resource with URI template."""
        retriever = lambda uri: {"data": "test"}
        mcp_instance.register_resource(
            uri_template="gnn://models/{model_name}",
            retriever=retriever,
            description="GNN model resource",
        )
        assert "gnn://models/{model_name}" in mcp_instance.resources

    @pytest.mark.unit
    def test_resource_metadata(self, mcp_instance):
        """Should store resource metadata correctly."""
        retriever = lambda uri: {}
        mcp_instance.register_resource(
            uri_template="test://resource",
            retriever=retriever,
            description="Test resource",
            mime_type="text/plain",
            category="testing",
        )
        resource = mcp_instance.resources["test://resource"]
        assert resource.mime_type == "text/plain"
        assert resource.category == "testing"


class TestMCPServerLifecycle:
    """Test MCPServer start/stop and request handling."""

    @pytest.mark.unit
    def test_server_creation(self, mcp_instance):
        """Should create a server with the given MCP instance."""
        server = MCPServer(mcp_instance)
        assert server.running is False
        assert server.mcp is mcp_instance

    @pytest.mark.unit
    def test_server_start_stop(self, mcp_instance):
        """Should start and stop cleanly."""
        server = MCPServer(mcp_instance)
        assert server.start() is True
        assert server.running is True
        assert server.stop() is True
        assert server.running is False

    @pytest.mark.unit
    def test_server_double_start(self, mcp_instance):
        """Starting an already-running server should return False."""
        server = MCPServer(mcp_instance)
        server.start()
        assert server.start() is False
        server.stop()

    @pytest.mark.unit
    def test_server_double_stop(self, mcp_instance):
        """Stopping an already-stopped server should return False."""
        server = MCPServer(mcp_instance)
        assert server.stop() is False

    @pytest.mark.unit
    def test_handle_initialize_request(self, mcp_instance):
        """Should handle JSON-RPC initialize request."""
        server = MCPServer(mcp_instance)
        response = server.handle_request({
            "jsonrpc": "2.0",
            "method": "initialize",
            "id": 1,
        })
        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert "protocolVersion" in response["result"]

    @pytest.mark.unit
    def test_handle_tools_list_request(self, mcp_instance, sample_tool_func):
        """Should list registered tools via JSON-RPC."""
        mcp_instance.register_tool(
            name="my_tool",
            func=sample_tool_func,
            schema={},
            description="A tool",
        )
        server = MCPServer(mcp_instance)
        response = server.handle_request({
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 2,
        })
        assert "result" in response
        tools = response["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        assert "my_tool" in tool_names

    @pytest.mark.unit
    def test_handle_invalid_method(self, mcp_instance):
        """Should return error for unknown method."""
        server = MCPServer(mcp_instance)
        response = server.handle_request({
            "jsonrpc": "2.0",
            "method": "unknown/method",
            "id": 3,
        })
        assert "error" in response
        assert response["error"]["code"] == -32601

    @pytest.mark.unit
    def test_handle_invalid_jsonrpc(self, mcp_instance):
        """Should return error for invalid jsonrpc version."""
        server = MCPServer(mcp_instance)
        response = server.handle_request({
            "jsonrpc": "1.0",
            "method": "initialize",
            "id": 4,
        })
        assert "error" in response


class TestMCPCapabilities:
    """Test capabilities reporting."""

    @pytest.mark.unit
    def test_get_capabilities_structure(self, mcp_instance):
        """Should return tools, resources, and server info."""
        caps = mcp_instance.get_capabilities()
        assert "tools" in caps
        assert "resources" in caps
        assert "server" in caps
        assert caps["server"]["name"] == "GNN MCP Server"

    @pytest.mark.unit
    def test_get_server_status(self, mcp_instance):
        """Should return status with uptime and counts."""
        status = mcp_instance.get_server_status()
        assert "uptime" in status
        assert "tools_count" in status
        assert "resources_count" in status
        assert status["tools_count"] == 0


class TestMCPToolDataclass:
    """Test the MCPTool dataclass validation."""

    @pytest.mark.unit
    def test_tool_creation(self):
        """Should create a valid MCPTool."""
        tool = MCPTool(
            name="test",
            func=lambda: None,
            schema={},
            description="test tool",
        )
        assert tool.name == "test"
        assert tool.use_count == 0

    @pytest.mark.unit
    def test_tool_mark_used(self):
        """mark_used should increment use count."""
        tool = MCPTool(name="t", func=lambda: None, schema={}, description="t")
        tool.mark_used()
        tool.mark_used()
        assert tool.use_count == 2
        assert tool.last_used is not None

    @pytest.mark.unit
    def test_tool_empty_name_raises(self):
        """Should reject empty tool name."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            MCPTool(name="", func=lambda: None, schema={}, description="test")

    @pytest.mark.unit
    def test_tool_non_callable_raises(self):
        """Should reject non-callable func."""
        with pytest.raises(ValueError, match="callable"):
            MCPTool(name="bad", func="not_callable", schema={}, description="test")
