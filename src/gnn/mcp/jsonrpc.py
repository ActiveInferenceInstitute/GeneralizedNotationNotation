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

import datetime
import json
import math
import os
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
    "sanitize_json_value",
    "serialize_response",
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_EMBED_CHARS",
    "TRUNCATION_NOTICE",
    "truncate_embedded_text",
    "tag_non_json_values",
]

JSONRPC_VERSION = "2.0"

# Standard JSON-RPC 2.0 / MCP error codes used across the transports.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# --- Request/response size policy (wave-2 MED-04) --------------------------------------
#
# Unbounded reads let a garbage Content-Length header abort the HTTP
# connection uncaught and let a hostile line hang the stdio reader, so both
# transports enforce one shared cap. Responses larger than
# MAX_RESPONSE_EMBED_CHARS are truncated to the policy documented below
# instead of being streamed raw.
MAX_REQUEST_BYTES = 32 * 1024 * 1024  # 32 MiB request body cap, both transports
MAX_RESPONSE_EMBED_CHARS = 1_000_000  # 1M chars per embedded response text

TRUNCATION_NOTICE = (
    "... [truncated by GNN MCP server response-size policy "
    "(MAX_RESPONSE_EMBED_CHARS); the full result exceeded the embed limit]"
)

# Maximum container nesting the wire walkers descend into. A cyclic or
# pathologically nested payload raises ValueError here — a normal exception
# with stack room left for the caller's fallback path — instead of a
# RecursionError at the interpreter limit, whose handler itself can fail.
_MAX_SANITIZE_DEPTH = 100

def validate_request(request: Any) -> dict[str, Any] | None:
    """Reject invalid envelopes; supported MCP methods require object params.

    Single-request contract (MIN-01): this server intentionally accepts only
    ONE JSON-RPC object per message. JSON-RPC 2.0 batch arrays (``[req, ...]``)
    are rejected with -32600 — GNN MCP tools execute pipeline steps with
    ordering/locking side effects that a batch would execute concurrently, so
    the array form is refused rather than silently flattened. Clients with
    several calls MUST issue them as separate requests on the same transport.
    A non-dict ``params`` member is rejected -32602 the same way as any other
    method payload.
    """
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


# --- Wire serialization ----------------------------------------------------------------
#
# Tool results and error payloads are produced by arbitrary callables, so the
# transports must never assume a payload is JSON-serializable: a set, a
# datetime, a numpy scalar, or a NaN float would otherwise crash the stdio
# writer thread (silently hanging the client), abort the HTTP response
# mid-write, or emit bare NaN/Infinity tokens that strict JSON parsers reject
# (RFC 8259 §6 requires interoperable implementations to never emit them).
# Every transport serializes outgoing envelopes through serialize_response.


def _non_finite_float_token(value: float) -> str:
    """Canonical JSON-safe token for a non-finite float."""
    if value != value:
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def sanitize_json_value(value: Any, _depth: int = 0) -> Any:
    """Return a JSON-serializable deep copy of ``value``.

    Non-finite floats become their canonical string tokens; sets and
    frozensets become deterministically ordered lists; bytes decode with a
    lossless backslash escape; datetimes use ISO format; paths render via
    ``os.fspath``; numpy-style scalars unwrap via ``item()``; every other
    non-serializable object degrades to ``str``.

    Nesting deeper than ``_MAX_SANITIZE_DEPTH`` (e.g. a cyclic structure)
    raises ``ValueError``.
    """
    if _depth > _MAX_SANITIZE_DEPTH:
        raise ValueError(
            f"JSON payload nesting exceeds maximum depth {_MAX_SANITIZE_DEPTH}"
        )
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _non_finite_float_token(value)
    if isinstance(value, dict):
        return {
            k
            if isinstance(k, (str, int, bool)) or k is None
            else str(k): sanitize_json_value(v, _depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item, _depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        sanitized = [sanitize_json_value(item, _depth + 1) for item in value]
        sanitized.sort(key=repr)
        return sanitized
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="backslashreplace")
    if isinstance(value, (datetime.date, datetime.time, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    to_item = getattr(value, "item", None)  # numpy scalars and 0-d arrays
    if callable(to_item):
        try:
            return sanitize_json_value(to_item(), _depth + 1)
        except Exception:  # noqa: BLE001 — degrade to str() below
            pass
    return str(value)


def serialize_response(
    payload: Any,
    *,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    ensure_ascii: bool = False,
) -> str:
    """Serialize an outgoing payload to protocol-valid JSON text.

    All transports must route results AND error envelopes through this one
    helper so a payload produced by an arbitrary tool callable can never
    crash a writer thread or emit invalid JSON. Raises ``ValueError`` for
    payloads deeper than ``_MAX_SANITIZE_DEPTH`` (callers convert that into a
    string-only -32603 fallback envelope, which this helper serializes
    trivially).

    Fast path: try ``json.dumps`` directly first (the common case — payloads
    that are already JSON-serializable). Fall back to ``sanitize_json_value``
    only when the fast path raises (non-serializable types, NaN/Inf, deep
    nesting), so the deep walk is skipped for every well-formed payload.
    """
    try:
        return json.dumps(
            payload,
            indent=indent,
            separators=separators,
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return json.dumps(
            sanitize_json_value(payload),
            indent=indent,
            separators=separators,
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        )


def tag_non_json_values(value: Any, _depth: int = 0) -> Any:
    """Deep-copy ``value`` with non-JSON-native values replaced by typed tags.

    Used for result-cache keys so structurally distinct params can never alias
    to one key the way plain ``json.dumps(..., default=str)`` allowed (the set
    ``{1}`` and the string ``"{1}"`` both stringify to ``"{1}"``).
    """
    if _depth > _MAX_SANITIZE_DEPTH:
        raise ValueError(
            f"JSON payload nesting exceeds maximum depth {_MAX_SANITIZE_DEPTH}"
        )
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"__json_type__": "float", "repr": repr(value)}
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return {
            k if isinstance(k, str) else str(k): tag_non_json_values(v, _depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [tag_non_json_values(item, _depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        return {
            "__json_type__": "set",
            "items": sorted(repr(item) for item in value),
        }
    return {"__json_type__": type(value).__name__, "repr": repr(value)}


def truncate_embedded_text(text: str, limit: int = MAX_RESPONSE_EMBED_CHARS) -> str:
    """Apply the documented response-size policy to an embedded text payload.

    Tools that return large matrices serialize their results into a single
    text field (``tools/call`` embeds ``serialize_response(result)``). To
    keep responses bounded on every transport without changing wire shape,
    an oversized payload is TRUNCATED (never dropped and never turned into a
    hard error): the head ``limit`` characters survive and a named notice
    documents the cut, so clients always receive parseable JSON with an
    explicit marker instead of a silent partial payload.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + TRUNCATION_NOTICE
