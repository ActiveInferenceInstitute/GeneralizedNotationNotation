"""Regression for CVE-2026-4539: AdlLexer GUID regex catastrophic backtracking."""

from __future__ import annotations

import time

import pytest


@pytest.mark.timeout(10)
def test_adllexer_guid_redos_mitigated() -> None:
    from pygments import lex
    from pygments.lexers import AdlLexer

    malicious_input = "A" * 10000 + "-"
    lexer = AdlLexer()
    start = time.perf_counter()
    list(lex(malicious_input, lexer))
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"possible ReDoS regression: {elapsed:.2f}s"
