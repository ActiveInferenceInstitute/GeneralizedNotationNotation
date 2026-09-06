#!/usr/bin/env python3
"""Pure JSON-RPC 2.0 envelope helpers shared by the MCP server transports.

Extracted from ``server_core`` / ``server_stdio`` / ``server_http`` so all three
build identical wire envelopes from one implementation. The builders are pure
functions: they never touch queues, sockets, or the MCP registry, so they can
be unit-tested and reused by any future transport.

ID policy: responses echo the request ``id``, including explicit null IDs.
Transports suppress responses to notifications before serialization. The
``omit_id_when_none`` option remains for compatibility callers only; all three
maintained transports use the default envelope.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "JSONRPC_VERSION",
    "PARSE_ERROR",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "INVALID_PARAMS",
    "INTERNAL_ERROR",
    "jsonrpc_result",
    "jsonrpc_error",
]

JSONRPC_VERSION = "2.0"

# Standard JSON-RPC 2.0 / MCP error codes used across the transports.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def validate_request(request: Any) -> dict[str, Any] | None:
    """Reject invalid envelopes; supported MCP methods require object params."""
    if (
        not isinstance(request, dict)
        or request.get("jsonrpc") != JSONRPC_VERSION
        or not isinstance(request.get("method"), str)
        or not request["method"]
        or (
            request.get("id") is not None
            and type(request["id"]) not in (str, int, float)
        )
    ):
        return jsonrpc_error(None, INVALID_REQUEST, "Invalid Request")
    if not isinstance(request.get("params", {}), dict):
        return jsonrpc_error(
            request.get("id"), INVALID_PARAMS, "Params must be an object"
        )
    return None


def jsonrpc_result(
    request_id: Any,
    result: Any,
    *,
    omit_id_when_none: bool = False,
) -> dict[str, Any]:
    """Build a successful JSON-RPC 2.0 response envelope.

    Args:
        request_id: Request id echoed back to the caller.
        result: Successful result payload (any JSON-serialisable value).
        omit_id_when_none: When True, drop the ``id`` key entirely if
            ``request_id`` is ``None`` instead of emitting ``"id": null``.
    """
    response: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "result": result}
    if request_id is not None or not omit_id_when_none:
        response["id"] = request_id
    return response


def jsonrpc_error(
    request_id: Any,
    code: int,
    message: str,
    data: Any = None,
    *,
    omit_id_when_none: bool = False,
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error response envelope.

    Args:
        request_id: Request id echoed back to the caller.
        code: Integer error code (JSON-RPC reserved range or server-defined).
        message: Human-readable error summary.
        data: Optional structured error detail; omitted from the payload when
            ``None``.
        omit_id_when_none: When True, drop the ``id`` key entirely if
            ``request_id`` is ``None`` instead of emitting ``"id": null``.
    """
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    response: dict[str, Any] = {
        "jsonrpc": JSONRPC_VERSION,
        "error": error,
    }
    if request_id is not None or not omit_id_when_none:
        response["id"] = request_id
    return response
