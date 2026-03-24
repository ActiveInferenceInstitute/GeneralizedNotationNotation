#!/usr/bin/env python3
"""
GNN Language Server — Minimal LSP server for GNN file diagnostics.

Provides:
  - textDocument/didOpen + didSave → diagnostics (section validation, parse errors)
  - textDocument/hover → variable info (dimensions, type)

Requires `pygls` package. Falls back gracefully when not installed.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check for pygls availability
try:
    from lsprotocol.types import (
        TEXT_DOCUMENT_DID_OPEN,
        TEXT_DOCUMENT_DID_SAVE,
        TEXT_DOCUMENT_HOVER,
        Diagnostic,
        DiagnosticSeverity,
        DidOpenTextDocumentParams,
        DidSaveTextDocumentParams,
        Hover,
        HoverParams,
        MarkupContent,
        MarkupKind,
        Position,
        Range,
    )
    from pygls.server import LanguageServer
    PYGLS_AVAILABLE = True
except ImportError:
    PYGLS_AVAILABLE = False
    logger.debug("pygls not installed — LSP server unavailable. Install with: pip install pygls")


def create_server() -> Any:
    """
    Create and configure the GNN language server.

    Returns:
        Configured LanguageServer instance, or None if pygls is not installed.
    """
    if not PYGLS_AVAILABLE:
        logger.warning("Cannot create LSP server: pygls is not installed")
        return None

    server = LanguageServer("gnn-lsp", "0.1.0")

    @server.feature(TEXT_DOCUMENT_DID_OPEN)
    def did_open(params: DidOpenTextDocumentParams):
        """Publish diagnostics when a GNN file is opened."""
        _publish_diagnostics(server, params.text_document.uri, params.text_document.text)

    @server.feature(TEXT_DOCUMENT_DID_SAVE)
    def did_save(params: DidSaveTextDocumentParams):
        """Re-publish diagnostics on save."""
        doc = server.workspace.get_text_document(params.text_document.uri)
        _publish_diagnostics(server, params.text_document.uri, doc.source)

    @server.feature(TEXT_DOCUMENT_HOVER)
    def hover(params: HoverParams):
        """Show variable info on hover."""
        doc = server.workspace.get_text_document(params.text_document.uri)
        return _get_hover(doc.source, params.position)

    return server


def _publish_diagnostics(server: Any, uri: str, content: str) -> None:
    """Run GNN validation and publish diagnostics."""
    import sys
    src_dir = str(Path(__file__).parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    diagnostics: List[Diagnostic] = []

    try:
        from gnn.schema import (
            parse_connections,
            parse_state_space,
            validate_required_sections,
        )

        file_path = uri.replace("file://", "")

        # Section validation
        section_errors = validate_required_sections(content, file_path=file_path)
        for err in section_errors:
            line = _extract_line(err)
            diagnostics.append(Diagnostic(
                range=Range(
                    start=Position(line=max(0, line - 1), character=0),
                    end=Position(line=max(0, line - 1), character=100),
                ),
                message=str(err),
                severity=DiagnosticSeverity.Error,
                source="gnn-lsp",
            ))

        # Parse errors
        variables, var_errors = parse_state_space(content, file_path=file_path)
        for err in var_errors:
            line = _extract_line(err)
            diagnostics.append(Diagnostic(
                range=Range(
                    start=Position(line=max(0, line - 1), character=0),
                    end=Position(line=max(0, line - 1), character=100),
                ),
                message=str(err),
                severity=DiagnosticSeverity.Warning,
                source="gnn-lsp",
            ))

        var_names = {v.name for v in variables}
        _, conn_errors = parse_connections(content, known_variables=var_names, file_path=file_path)
        for err in conn_errors:
            line = _extract_line(err)
            diagnostics.append(Diagnostic(
                range=Range(
                    start=Position(line=max(0, line - 1), character=0),
                    end=Position(line=max(0, line - 1), character=100),
                ),
                message=str(err),
                severity=DiagnosticSeverity.Warning,
                source="gnn-lsp",
            ))

    except Exception as e:
        diagnostics.append(Diagnostic(
            range=Range(
                start=Position(line=0, character=0),
                end=Position(line=0, character=100),
            ),
            message=f"LSP analysis error: {e}",
            severity=DiagnosticSeverity.Information,
            source="gnn-lsp",
        ))

    server.publish_diagnostics(uri, diagnostics)
    logger.debug(f"Published {len(diagnostics)} diagnostics for {uri}")


def _get_hover(content: str, position: Any) -> Optional[Any]:
    """Generate hover info for a variable at the given position."""
    if not PYGLS_AVAILABLE:
        return None

    import sys
    src_dir = str(Path(__file__).parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    try:
        from gnn.schema import parse_state_space

        lines = content.split("\n")
        if position.line >= len(lines):
            return None

        line = lines[position.line]
        # Extract word at cursor position
        word = _word_at_position(line, position.character)
        if not word:
            return None

        # Look up variable
        variables, _ = parse_state_space(content)
        for v in variables:
            if v.name == word:
                info = (
                    f"**{v.name}**\n\n"
                    f"- **Dimensions**: {v.dimensions}\n"
                    f"- **Type**: {v.dtype}\n"
                )
                if v.default:
                    info += f"- **Default**: {v.default}\n"
                return Hover(
                    contents=MarkupContent(kind=MarkupKind.Markdown, value=info)
                )
    except (ImportError, ValueError, AttributeError) as e:
        logger.debug(f"Hover info unavailable: {e}")

    return None


def _word_at_position(line: str, char: int) -> Optional[str]:
    """Extract the word at a given character position in a line."""
    if char >= len(line) or not (line[char].isalnum() or line[char] == "_"):
        return None
    # Find word boundaries
    start = char
    while start > 0 and (line[start - 1].isalnum() or line[start - 1] == "_"):
        start -= 1
    end = char
    while end < len(line) and (line[end].isalnum() or line[end] == "_"):
        end += 1
    word = line[start:end]
    return word if word else None


def _extract_line(error) -> int:
    """Extract line number from a GNNParseError or string."""
    if hasattr(error, "line") and error.line:
        return error.line
    # Try to extract from string representation
    m = re.search(r":(\d+)", str(error))
    return int(m.group(1)) if m else 1


def start_server():
    """Start the LSP server on stdio."""
    server = create_server()
    if server:
        logger.info("Starting GNN LSP server on stdio...")
        server.start_io()
    else:
        logger.error("LSP server could not be created (missing pygls)")
