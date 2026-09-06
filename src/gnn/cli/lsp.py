"""GNN Language Server helpers for the ``gnn lsp`` subcommand.

Pure JSON-RPC framing and request/response handlers plus the serve loop.
Transport streams are injectable (``read_message``/``write_message``
default to ``sys.stdin``/``sys.stdout`` at call time), so every layer is
testable without a real stdio session.

Public functions: read_message, write_message, handle_initialize,
handle_hover, diagnose_text, publish_diagnostics, run_lsp_loop, start_lsp.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Callable, Optional, TextIO

logger = logging.getLogger(__name__)


def read_message(stream: Optional[TextIO] = None) -> Any:
    """Read one Content-Length-framed JSON-RPC message from ``stream``.

    Defaults to ``sys.stdin``. Returns None on EOF or malformed framing.
    """
    if stream is None:
        stream = sys.stdin
    line = stream.readline()
    if not line:
        return None

    if not line.startswith("Content-Length: "):
        return None

    content_length = int(line[16:].strip())

    # Read the empty line
    stream.readline()

    # Read the body
    body = stream.read(content_length)
    return json.loads(body)


def write_message(msg: Any, stream: Optional[TextIO] = None) -> None:
    """Write a Content-Length-framed JSON-RPC message to ``stream``.

    Defaults to ``sys.stdout`` (resolved at call time so test harnesses
    that swap stdout are honored).
    """
    if stream is None:
        stream = sys.stdout
    body = json.dumps(msg)
    stream.write(f"Content-Length: {len(body)}\r\n\r\n{body}")
    stream.flush()


def handle_initialize(msg_id: Any) -> dict[str, Any]:
    """Handle the initialize request."""
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "capabilities": {
                "textDocumentSync": 1,  # Full sync
                "completionProvider": {
                    "resolveProvider": False,
                    "triggerCharacters": ["."],
                },
                "hoverProvider": True,
            },
            "serverInfo": {"name": "gnn-lsp", "version": "1.0.0"},
        },
    }


def handle_hover(msg_id: Any, params: Any) -> dict[str, Any]:
    """Handle the textDocument/hover request."""
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "contents": {
                "kind": "markdown",
                "value": "**GNN Identifier**\n\nGeneralized Notation Notation construct.",
            }
        },
    }


def diagnose_text(text: str) -> list[dict[str, Any]]:
    """Return LSP diagnostic dicts for basic GNN text issues.

    Pure function: no I/O, deterministic output for a given document.
    """
    diagnostics: list[dict[str, Any]] = []

    # Simple syntax check: look for missing closing braces
    if "{" in text and "}" not in text:
        diagnostics.append(
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 100},
                },
                "severity": 1,  # Error
                "message": "Missing closing brace '}'",
            }
        )

    return diagnostics


def publish_diagnostics(uri: Any, text: Any, stream: Optional[TextIO] = None) -> None:
    """Run basic validation and publish a diagnostics notification."""
    write_message(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {"uri": uri, "diagnostics": diagnose_text(text)},
        },
        stream,
    )


def run_lsp_loop(
    reader: Callable[[], Any],
    writer: Callable[..., Any],
) -> None:
    """Serve one LSP session over injected transport callables.

    Reads framed requests via ``reader()`` until EOF or ``exit``, dispatches
    each method, and answers via ``writer``. Unhandled requests receive a
    ``Method not found`` error response; unhandled notifications are ignored.
    An exception while handling one message logs and terminates the loop.
    """
    while True:
        try:
            msg = reader()
            if not msg:
                break

            logger.info("Received: %s", msg.get("method"))

            method = msg.get("method")
            msg_id = msg.get("id")

            if method == "initialize":
                writer(handle_initialize(msg_id))
            elif method == "initialized":
                logger.debug("Client initialized notification received")
            elif method == "textDocument/hover":
                writer(handle_hover(msg_id, msg.get("params")))
            elif method == "textDocument/didOpen":
                params = msg.get("params", {})
                doc = params.get("textDocument", {})
                uri = doc.get("uri", "")
                text = doc.get("text", "")
                if uri and text:
                    publish_diagnostics(uri, text)
            elif method == "textDocument/didChange":
                params = msg.get("params", {})
                doc = params.get("textDocument", {})
                uri = doc.get("uri", "")
                changes = params.get("contentChanges", [])
                if uri and changes:
                    # Sync sends full text
                    text = changes[0].get("text", "")
                    publish_diagnostics(uri, text)
            elif method == "shutdown":
                writer({"jsonrpc": "2.0", "id": msg_id, "result": None})
            elif method == "exit":
                break
            else:
                # Ignore unhandled notifications
                if msg_id is not None:
                    # Return method not found if it is a request
                    writer(
                        {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "error": {"code": -32601, "message": "Method not found"},
                        }
                    )
        except Exception as e:
            logger.error("Error handling message: %s", e)
            break


def start_lsp(log_path: Optional[str] = "gnn-lsp.log") -> None:
    """Start the Language Server Protocol loop on stdin/stdout.

    Logs to ``log_path`` (default ``gnn-lsp.log`` in the working directory)
    so stdout stays a clean JSON-RPC transport; pass ``None`` to disable
    file logging.
    """
    if log_path is not None:
        logging.basicConfig(filename=log_path, level=logging.INFO)
    logger.info("Starting GNN LSP Server...")
    run_lsp_loop(read_message, write_message)
    logger.info("GNN LSP Server shutting down.")
