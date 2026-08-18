"""Tests for lsp module's public API surface not covered by existing tests.

Covers: start_server behavior, get_module_info (extended), FEATURES,
__version__, _word_at_position edge cases, _extract_line edge cases.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestLSPConstants:
    """Test module-level constants."""

    def test_features_dict(self) -> None:
        import lsp

        assert hasattr(lsp, "FEATURES")
        assert isinstance(lsp.FEATURES, dict)
        for key in ("diagnostics", "hover_info", "completion", "gnn_language_support"):
            assert key in lsp.FEATURES

    def test_version(self) -> None:
        import lsp

        assert hasattr(lsp, "__version__")
        assert isinstance(lsp.__version__, str)

    def test_get_module_info_extended(self) -> None:
        from lsp import get_module_info

        info = get_module_info()
        assert isinstance(info, dict)
        assert info["name"] == "lsp"
        assert "version" in info
        assert "description" in info
        # Description should reference Language Server Protocol
        assert "Language Server" in info["description"] or "LSP" in info["description"]


class TestWordAtPositionEdgeCases:
    """Edge cases for _word_at_position."""

    def test_word_at_start_of_line(self) -> None:
        from lsp import _word_at_position

        assert _word_at_position("StateSpaceBlock", 0) == "StateSpaceBlock"

    def test_word_ending_at_line_end(self) -> None:
        from lsp import _word_at_position

        assert _word_at_position("abc", 2) == "abc"

    def test_empty_line(self) -> None:
        from lsp import _word_at_position

        assert _word_at_position("", 0) is None

    def test_underscore_joined_word(self) -> None:
        from lsp import _word_at_position

        assert _word_at_position("my_var_name", 4) == "my_var_name"

    def test_negative_position(self) -> None:
        from lsp import _word_at_position

        # Negative index wraps in Python — since abc[-1] = 'c', it finds 'c'
        word = _word_at_position("abc", -1)
        # At minimum, should not crash; return value is implementation-specific
        assert word is None or isinstance(word, str)

    def test_single_char_word(self) -> None:
        from lsp import _word_at_position

        assert _word_at_position("a b", 0) == "a"


class TestExtractLineEdgeCases:
    """Edge cases for _extract_line."""

    def test_extract_line_from_object_with_line_zero(self) -> None:
        from types import SimpleNamespace

        from lsp import _extract_line

        # .line = 0 is falsy — should fall through to default 1
        err = SimpleNamespace(line=0)
        assert _extract_line(err) == 1

    def test_extract_line_multiple_colons(self) -> None:
        from lsp import _extract_line

        # Should grab the first number after a colon
        assert _extract_line("error at :7: more detail") == 7

    def test_extract_line_trailing_colon(self) -> None:
        from lsp import _extract_line

        assert _extract_line("error:") == 1

    def test_extract_line_large_number(self) -> None:
        from lsp import _extract_line

        assert _extract_line("line :9999") == 9999


class TestStartServer:
    """Test start_server function."""

    def test_start_server_handles_missing_pygls(self, monkeypatch: Any) -> None:
        import lsp

        original_flag = lsp.PYGLS_AVAILABLE
        try:
            lsp.PYGLS_AVAILABLE = False
            # Should not raise; logs error and returns None
            assert lsp.start_server() is None
        finally:
            lsp.PYGLS_AVAILABLE = original_flag

    def test_start_server_with_pygls_present(self) -> None:
        import lsp

        assert lsp.PYGLS_AVAILABLE, "pygls must be installed in the dev environment"

        # start_server would block on stdio; monkeypatch create_server to return None
        original = lsp.create_server
        try:
            lsp.create_server = lambda: None
            assert lsp.start_server() is None
        finally:
            lsp.create_server = original
