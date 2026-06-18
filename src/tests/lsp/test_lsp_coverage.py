#!/usr/bin/env python3
"""Phase 4.2 regression tests for lsp (Step 22 support).

Uses real GNN syntax to exercise completion, diagnostic, and hover paths.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_lsp_module_exports_expected_surface() -> Any:
    import lsp

    # Core expected names per src/lsp/__init__.py.
    assert hasattr(lsp, "create_server")
    assert hasattr(lsp, "start_server")
    assert hasattr(lsp, "get_module_info")


def test_get_module_info_returns_populated_dict() -> Any:
    from lsp import get_module_info

    info = get_module_info()
    assert isinstance(info, dict)
    assert "name" in info
    assert info["name"].lower() in {"lsp", "gnn-lsp"}


def test_word_at_position_extracts_token() -> Any:
    from lsp import _word_at_position

    # Cursor inside "StateSpaceBlock"
    word = _word_at_position("  StateSpaceBlock", 4)
    assert word is not None
    assert "State" in word or "StateSpace" in word or word == "StateSpaceBlock"


def test_word_at_position_whitespace_returns_none_or_empty() -> Any:
    from lsp import _word_at_position

    word = _word_at_position("   ", 1)
    # Accept None or empty string — both signal "no token here".
    assert word is None or word == ""


def test_extract_line_from_attribute_error() -> Any:
    from types import SimpleNamespace

    from lsp import _extract_line

    # _extract_line first checks for an .line attribute on the error.
    err = SimpleNamespace(line=5)
    assert _extract_line(err) == 5


def test_extract_line_from_string_representation() -> Any:
    from lsp import _extract_line

    # When no .line attribute, falls back to regex extraction from str(error).
    line = _extract_line("error at myfile.gnn:42: bad section")
    assert line == 42


def test_extract_line_defaults_to_1_when_nothing_matches() -> Any:
    from lsp import _extract_line

    # Regression: arbitrary shapes should return the safe default (1), not raise.
    line = _extract_line("some error with no numbers")
    assert line == 1


def test_create_server_returns_server_or_none_without_uncaught_error() -> Any:
    """create_server returns a server when pygls is installed and None otherwise."""
    import lsp

    try:
        server = lsp.create_server()
    except Exception as e:
        pytest.fail(f"create_server() raised unexpected {type(e).__name__}: {e}")
    if lsp.PYGLS_AVAILABLE:
        assert server is not None
    else:
        assert server is None
