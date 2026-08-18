#!/usr/bin/env python3
"""Tests for the bounded literal_eval wrapper (utils.safe_eval)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.safe_eval import safe_literal_eval


class TestSafeLiteralEval:
    """Bounds enforcement for untrusted GNN parameter strings."""

    def test_parses_plain_literals(self) -> None:
        assert safe_literal_eval("[1, 2, 3]") == [1, 2, 3]
        assert safe_literal_eval("(1.0, 0.0)") == (1.0, 0.0)
        assert safe_literal_eval("42") == 42

    def test_rejects_overlong_literal(self) -> None:
        with pytest.raises(ValueError):
            safe_literal_eval("[" * 5_001 + "]" * 5_001)

    def test_rejects_deep_nesting(self) -> None:
        with pytest.raises(ValueError):
            safe_literal_eval("[" * 100 + "]" * 100)

    def test_passes_through_non_strings(self) -> None:
        assert safe_literal_eval([1, 2, 3]) == [1, 2, 3]
        assert safe_literal_eval(7) == 7

    def test_raises_syntax_error_on_garbage(self) -> None:
        with pytest.raises(SyntaxError):
            safe_literal_eval("not a literal !!")
