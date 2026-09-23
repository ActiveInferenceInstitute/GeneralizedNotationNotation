#!/usr/bin/env python3
"""
GNN Language Server — Minimal LSP server for GNN file diagnostics.

Provides:
  - textDocument/didOpen + didChange + didSave → diagnostics (section
    validation, parse errors, matrix-dimension cross-validation, unknown
    section headers)
  - textDocument/hover → variable info (dimensions, type)
  - textDocument/completion → GNN vocabulary completions (shared pygls-free
    vocabulary module: gnn.lsp.completions)

Requires `pygls` package. Falls back gracefully when not installed.
"""

from typing import Any, cast

from gnn import __version__
from gnn.schemas.section_contract import CANONICAL_GNN_SECTIONS

FEATURES: dict[str, Any] = {
    "diagnostics": True,
    "did_change": True,
    "hover_info": True,
    "completion": True,
    "gnn_language_support": True,
}


import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check for pygls availability
try:
    from lsprotocol.types import (
        TEXT_DOCUMENT_COMPLETION,
        TEXT_DOCUMENT_DID_CHANGE,
        TEXT_DOCUMENT_DID_OPEN,
        TEXT_DOCUMENT_DID_SAVE,
        TEXT_DOCUMENT_HOVER,
        CompletionItem,
        CompletionItemKind,
        CompletionParams,
        Diagnostic,
        DiagnosticSeverity,
        DidChangeTextDocumentParams,
        DidOpenTextDocumentParams,
        DidSaveTextDocumentParams,
        Hover,
        HoverParams,
        MarkupContent,
        MarkupKind,
        Position,
        PublishDiagnosticsParams,
        Range,
    )

    try:
        # pygls exposes LanguageServer on pygls.server (older) or pygls.lsp.server (newer).
        # Mypy only sees the venv's pygls type declarations; suppress the
        # attr-defined error here — the try/except handles both locations at runtime.
        from pygls.server import (  # type: ignore[attr-defined]
            LanguageServer,
        )
    except ImportError:
        from pygls.lsp.server import LanguageServer

    PYGLS_AVAILABLE = True
except ImportError:
    PYGLS_AVAILABLE = False
    logger.debug(
        "pygls not installed — LSP server unavailable. Install with: uv sync --group dev"
    )


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
    def did_open(params: DidOpenTextDocumentParams) -> Any:
        """Publish diagnostics when a GNN file is opened."""
        _publish_diagnostics(
            server, params.text_document.uri, params.text_document.text
        )

    @server.feature(TEXT_DOCUMENT_DID_SAVE)
    def did_save(params: DidSaveTextDocumentParams) -> Any:
        """Re-publish diagnostics on save."""
        doc = server.workspace.get_text_document(params.text_document.uri)
        _publish_diagnostics(server, params.text_document.uri, doc.source)

    @server.feature(TEXT_DOCUMENT_DID_CHANGE)
    def did_change(params: DidChangeTextDocumentParams) -> Any:
        """Republish diagnostics when the document changes.

        pygls applies the content changes (full or incremental) to the workspace
        document before dispatching this handler, so the stored source is current.
        """
        doc = server.workspace.get_text_document(params.text_document.uri)
        _publish_diagnostics(server, params.text_document.uri, doc.source)

    @server.feature(TEXT_DOCUMENT_HOVER)
    def hover(params: HoverParams) -> Any:
        """Show variable info on hover."""
        doc = server.workspace.get_text_document(params.text_document.uri)
        return _get_hover(doc.source, params.position)

    @server.feature(TEXT_DOCUMENT_COMPLETION)
    def completion(params: CompletionParams) -> Any:
        """Offer GNN vocabulary completions at the cursor position."""
        doc = server.workspace.get_text_document(params.text_document.uri)
        return _get_completions(doc.source, params.position)

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
            validate_matrix_dimensions,
            validate_required_sections,
        )

        file_path = uri.replace("file://", "")

        # Section validation
        section_errors = validate_required_sections(content, file_path=file_path)
        for err in section_errors:
            line = _extract_line(err)
            diagnostics.append(
                Diagnostic(
                    range=Range(
                        start=Position(line=max(0, line - 1), character=0),
                        end=Position(line=max(0, line - 1), character=100),
                    ),
                    message=str(err),
                    severity=DiagnosticSeverity.Error,
                    source="gnn",
                )
            )

        # Parse errors
        variables, var_errors = parse_state_space(content, file_path=file_path)
        for err in var_errors:
            line = _extract_line(err)
            diagnostics.append(
                Diagnostic(
                    range=Range(
                        start=Position(line=max(0, line - 1), character=0),
                        end=Position(line=max(0, line - 1), character=100),
                    ),
                    message=str(err),
                    severity=DiagnosticSeverity.Warning,
                    source="gnn",
                )
            )

        var_names = {v.name for v in variables}
        _, conn_errors = parse_connections(
            content, known_variables=var_names, file_path=file_path
        )
        for err in conn_errors:
            line = _extract_line(err)
            diagnostics.append(
                Diagnostic(
                    range=Range(
                        start=Position(line=max(0, line - 1), character=0),
                        end=Position(line=max(0, line - 1), character=100),
                    ),
                    message=str(err),
                    severity=DiagnosticSeverity.Warning,
                    source="gnn",
                )
            )

        # Matrix-dimension cross-validation (GNN-E002 errors and GNN-W003
        # undeclared-parameterization warnings, by the error's severity).
        for err in validate_matrix_dimensions(content, variables, file_path=file_path):
            line = _extract_line(err)
            severity = (
                DiagnosticSeverity.Warning
                if getattr(err, "severity", "error") == "warning"
                else DiagnosticSeverity.Error
            )
            diagnostics.append(
                Diagnostic(
                    range=Range(
                        start=Position(line=max(0, line - 1), character=0),
                        end=Position(line=max(0, line - 1), character=100),
                    ),
                    message=str(err),
                    severity=severity,
                    source="gnn",
                )
            )

    except Exception as e:
        diagnostics.append(
            Diagnostic(
                range=Range(
                    start=Position(line=0, character=0),
                    end=Position(line=0, character=100),
                ),
                message=f"LSP analysis error: {e}",
                severity=DiagnosticSeverity.Information,
                source="gnn-lsp",
            )
        )

    # Unknown `## ` section headers — header normalization mirrors
    # MarkdownGNNParser._split_into_sections (strip -> "## " -> [3:].strip()).
    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped.startswith("## "):
            continue
        header = stripped[3:].strip()
        if header in CANONICAL_GNN_SECTIONS:
            continue
        diagnostics.append(
            Diagnostic(
                range=Range(
                    start=Position(line=line_no - 1, character=0),
                    end=Position(line=line_no - 1, character=len(stripped)),
                ),
                message=f"Unknown section header '{header}'",
                severity=DiagnosticSeverity.Warning,
                source="gnn",
            )
        )

    _publish_to_server(server, uri, diagnostics)
    logger.debug(f"Published {len(diagnostics)} diagnostics for {uri}")


def _publish_to_server(server: Any, uri: str, diagnostics: List[Any]) -> None:
    """Publish diagnostics across pygls generations.

    pygls 1.x exposes ``publish_diagnostics(uri, diagnostics)``; pygls 2.x
    renamed it to ``text_document_publish_diagnostics(PublishDiagnosticsParams)``
    and removed the 1.x name, so probe for the 2.x API first.
    """
    publish_v2 = getattr(server, "text_document_publish_diagnostics", None)
    if publish_v2 is not None:
        publish_v2(PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics))
    else:
        server.publish_diagnostics(uri, diagnostics)


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


def _get_completions(content: str, position: Any) -> Optional[List[Any]]:
    """Build completion items for the GNN vocabulary at a cursor position.

    Thin adapter over the shared pygls-free vocabulary module
    ``gnn.lsp.completions`` — the same source the CLI server serves.
    """
    if not PYGLS_AVAILABLE:
        return None

    try:
        from gnn.lsp.completions import completion_context, context_completions

        line_prefix, in_model_parameters, in_gnn_section = completion_context(
            content, position.line, position.character
        )
        items = context_completions(
            line_prefix,
            in_model_parameters=in_model_parameters,
            in_gnn_section=in_gnn_section,
        )
        return [
            CompletionItem(
                label=item["label"],
                kind=CompletionItemKind(item["kind"]),
                detail=item["detail"],
                insert_text=item["insert_text"],
            )
            for item in items
        ]
    except (ImportError, ValueError, AttributeError) as e:
        logger.debug(f"Completion info unavailable: {e}")
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


def _extract_line(error: Any) -> int:
    """Extract line number from a GNNParseError or string."""
    if hasattr(error, "line") and error.line:
        return cast("int", error.line)
    # Try to extract from string representation
    m = re.search(r":(\d+)", str(error))
    return int(m.group(1)) if m else 1


def start_server() -> Any:
    """Start the LSP server on stdio."""
    server = create_server()
    if server:
        logger.info("Starting GNN LSP server on stdio...")
        server.start_io()
    else:
        logger.error("LSP server could not be created (missing pygls)")


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "lsp",
        "version": __version__,
        "description": "GNN Language Server Protocol implementation",
    }


__all__ = [
    "create_server",
    "start_server",
    "get_module_info",
    "__version__",
]
