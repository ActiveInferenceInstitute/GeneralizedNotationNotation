#!/usr/bin/env python3
"""
Covers the CLI LSP helper handlers in ``src/cli/lsp.py``.

Exercises the pure request/response handlers and the JSON-RPC framing helpers
(write_message to stdout, read_message from stdin). These are lightweight and
deterministic — no editor, network, or server loop involved.

Test Coverage:
- handle_initialize() capabilities advertisement
- handle_hover() markdown hover payload shape
- write_message() emits Content-Length-framed JSON to stdout
- read_message() parses a real Content-Length-framed stdin message
- publish_diagnostics() flags a missing closing brace / stays silent on clean text
"""

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cli import lsp as cli_lsp


class TestCLILSPHelpers:
    """Tests for the CLI LSP protocol helpers."""

    @pytest.mark.unit
    def test_handle_initialize_advertises_capabilities(self) -> None:
        """Initialize should advertise hover + completion capabilities."""
        response = cli_lsp.handle_initialize(42)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 42
        caps = response["result"]["capabilities"]
        assert caps["hoverProvider"] is True
        assert "textDocumentSync" in caps
        assert response["result"]["serverInfo"]["name"] == "gnn-lsp"

    @pytest.mark.unit
    def test_handle_hover_returns_markdown(self) -> None:
        """Hover should return a markdown payload."""
        response = cli_lsp.handle_hover(7, {})
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 7
        assert response["result"]["contents"]["kind"] == "markdown"
        assert "GNN" in response["result"]["contents"]["value"]

    @pytest.mark.unit
    def test_write_message_frames_with_content_length(self, capsys: Any) -> None:
        """write_message should emit a Content-Length-framed JSON body."""
        cli_lsp.write_message({"jsonrpc": "2.0", "id": 1})
        captured = capsys.readouterr().out
        assert captured.startswith("Content-Length: ")
        header, body = captured.split("\r\n\r\n", 1)
        expected_len = int(header.split(":", 1)[1].strip())
        assert expected_len == len(body)
        # Body must round-trip as JSON.
        assert json.loads(body)["id"] == 1

    @pytest.mark.unit
    def test_read_message_parses_framed_stdin(self, monkeypatch: Any) -> None:
        """read_message should parse a Content-Length-framed stdin message."""
        body = '{"jsonrpc":"2.0","id":1,"method":"initialize"}'
        framed = f"Content-Length: {len(body)}\r\n\r\n{body}"
        monkeypatch.setattr(sys, "stdin", io.StringIO(framed))
        msg = cli_lsp.read_message()
        assert msg is not None
        assert msg["method"] == "initialize"
        assert msg["id"] == 1

    @pytest.mark.unit
    def test_read_message_empty_stdin_returns_none(self, monkeypatch: Any) -> None:
        """An empty stdin line should yield None (EOF)."""
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        assert cli_lsp.read_message() is None

    @pytest.mark.unit
    def test_publish_diagnostics_flags_missing_brace(self, capsys: Any) -> None:
        """A document with an unclosed brace should emit an error diagnostic."""
        cli_lsp.publish_diagnostics("file:///model.md", "model { unreachable")
        captured = capsys.readouterr().out
        assert "textDocument/publishDiagnostics" in captured
        payload = json.loads(captured.split("\r\n\r\n", 1)[1])
        diagnostics = payload["params"]["diagnostics"]
        assert diagnostics, "Expected a diagnostic for the unclosed brace"
        assert diagnostics[0]["severity"] == 1  # Error
        assert "closing brace" in diagnostics[0]["message"].lower()

    @pytest.mark.unit
    def test_publish_diagnostics_clean_text_silent(self, capsys: Any) -> None:
        """Balanced braces should produce no diagnostics."""
        cli_lsp.publish_diagnostics("file:///clean.md", "model { balanced }")
        captured = capsys.readouterr().out
        payload = json.loads(captured.split("\r\n\r\n", 1)[1])
        assert payload["params"]["diagnostics"] == []