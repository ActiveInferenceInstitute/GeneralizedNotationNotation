"""Regression tests for scripts/check_external_links.py URL capture.

The checker is a standalone tool (not imported by src/), so tests load it by
path and exercise the pure helpers: ``_normalize_url`` (trailing-punctuation,
backtick, and unbalanced-paren handling) and ``_should_skip_url``
(local/template targets are not links).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.helpers import load_module_from_path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_external_links.py"


def _load() -> Any:
    return load_module_from_path("check_external_links", SCRIPT)


@pytest.fixture(scope="module")
def checker() -> Any:
    return _load()


class TestNormalizeUrl:
    def test_strips_trailing_punctuation(self, checker: Any) -> None:
        assert (
            checker._normalize_url("https://example.org/a.md.")
            == "https://example.org/a.md"
        )
        assert (
            checker._normalize_url("https://example.org/a.md,")
            == "https://example.org/a.md"
        )

    def test_strips_trailing_backtick(self, checker: Any) -> None:
        # storage.googleapis.com/jax-releases/*.html URLs sit in backticks
        assert (
            checker._normalize_url(
                "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`"
            )
            == "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        )

    def test_rebalances_cutoff_paren(self, checker: Any) -> None:
        # The URL regex stops at ')', truncating learn.microsoft links
        assert (
            checker._normalize_url(
                "https://learn.microsoft.com/en-us/previous-versions/ms256108(v=vs.85"
            )
            == "https://learn.microsoft.com/en-us/previous-versions/ms256108(v=vs.85)"
        )

    def test_balanced_parens_untouched(self, checker: Any) -> None:
        url = "https://en.wikipedia.org/wiki/APL_(programming_language)"
        assert checker._normalize_url(url) == url

    def test_noop_on_clean_url(self, checker: Any) -> None:
        url = "https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation"
        assert checker._normalize_url(url) == url


class TestShouldSkipUrl:
    def test_localhost_skipped(self, checker: Any) -> None:
        assert checker._should_skip_url("http://localhost:11434")
        assert checker._should_skip_url("http://127.0.0.1:8000/health")

    def test_example_dot_com_skipped(self, checker: Any) -> None:
        assert checker._should_skip_url("https://example.com/logo.png")
        assert checker._should_skip_url("https://docs.example.com/")
        assert checker._should_skip_url("https://icons.example.com/server.svg")

    def test_template_markers_skipped(self, checker: Any) -> None:
        assert checker._should_skip_url("https://github.com/<your-username>/repo.git")
        # the collector's regex truncates at '>'; the truncated form must skip too
        assert checker._should_skip_url("https://github.com/<your-username")
        assert checker._should_skip_url("http://server:port/api/search")
        assert checker._should_skip_url(
            "https://raw.githubusercontent.com/org/repo/main/spec/{0}.md"
        )

    def test_real_url_not_skipped(self, checker: Any) -> None:
        assert not checker._should_skip_url(
            "https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation"
        )
