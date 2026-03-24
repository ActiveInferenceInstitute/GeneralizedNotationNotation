"""Tests for the GNN Language Server."""

from unittest.mock import MagicMock, patch

import pytest

# Import the LSP server logic
try:
    from lsp import PYGLS_AVAILABLE, _extract_line, _word_at_position, create_server
except ImportError:
    try:
        from src.lsp import (
            PYGLS_AVAILABLE,
            _extract_line,
            _word_at_position,
            create_server,
        )
    except ImportError:
        PYGLS_AVAILABLE = False

def test_word_at_position() -> None:
    """Test extracting the word under the cursor."""
    line = "variable_name and more"
    assert _word_at_position(line, 0) == "variable_name"
    assert _word_at_position(line, 5) == "variable_name"
    assert _word_at_position(line, 12) == "variable_name"
    assert _word_at_position(line, 13) is None  # Space
    assert _word_at_position(line, 14) == "and"
    assert _word_at_position(line, 21) == "more"
    assert _word_at_position(line, 100) is None

def test_extract_line() -> None:
    """Test extracting line number from error objects or strings."""
    # From object with .line
    mock_err = MagicMock()
    mock_err.line = 42
    assert _extract_line(mock_err) == 42
    
    # From string with line info
    assert _extract_line("Parse error at :15") == 15
    assert _extract_line("No line info here") == 1
    assert _extract_line("Another :1234 error") == 1234

def test_create_server_graceful() -> None:
    """Test that create_server handles missing pygls gracefully."""
    with patch("lsp.PYGLS_AVAILABLE", False):
        server = create_server()
        assert server is None

@pytest.mark.skipif(not PYGLS_AVAILABLE, reason="pygls not installed")
def test_create_server_success() -> None:
    """Test that create_server returns a LanguageServer instance if available."""
    server = create_server()
    assert server is not None
    # Check if a few expected features are registered (internal to pygls)
    assert hasattr(server, "feature")

def test_lsp_availability_flag() -> None:
    """Verify that the availability flag matches reality."""
    try:
        import pygls.server
        expected = True
    except ImportError:
        expected = False
    
    assert PYGLS_AVAILABLE == expected
