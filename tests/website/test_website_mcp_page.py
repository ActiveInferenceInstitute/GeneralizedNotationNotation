"""Tests for the get_website_page MCP tool / read_website_page shared implementation.

Pins the page-content read contract shared by the Python API and the MCP
tool: catalogue-validated page keys, real content reads from a generated
site, character-cap truncation with an explicit marker, and graceful
errors. Page names are always derived from the one page catalogue
(``gnn.website.pages.page_names()``) — never hardcoded.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

TRUNCATION_MARKER = "\n\n… [truncated]"


def _build_site(tmp_path: Any) -> Path:
    """Build a real website from an empty input dir into ``tmp_path``."""
    from gnn.website import WebsiteGenerator

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    out: Path = tmp_path / "site"
    result = WebsiteGenerator().generate_website(
        {"input_dir": str(input_dir), "output_dir": str(out)}
    )
    assert result["success"] is True
    return out


class TestGetWebsitePage:
    @pytest.mark.unit
    def test_read_page_returns_catalogue_title_and_content(self, tmp_path: Any) -> None:
        from gnn.website.inspection import read_website_page
        from gnn.website.mcp import get_website_page_mcp
        from gnn.website.pages import SITE_PAGES, page_names

        site = _build_site(tmp_path)
        page_name = page_names()[0]
        spec = next(page for page in SITE_PAGES if page.name == page_name)

        # A generous explicit cap keeps the non-truncated assertions
        # deterministic regardless of how large the generated page is.
        result = get_website_page_mcp(str(site), page_name, max_chars=1_000_000)
        assert result["success"] is True
        assert result["page"] == page_name
        assert result["filename"] == spec.filename
        assert result["path"] == str(site / spec.filename)
        assert result["size_bytes"] > 0
        assert result["total_chars"] > 0
        assert result["truncated"] is False
        assert "<title>" in result["content"]
        assert spec.title in result["content"]

        # The Python API shares one implementation with the MCP tool.
        assert read_website_page(site, page_name, max_chars=1_000_000) == result

    @pytest.mark.unit
    def test_max_chars_caps_content_with_marker(self, tmp_path: Any) -> None:
        from gnn.website.mcp import get_website_page_mcp
        from gnn.website.pages import page_names

        site = _build_site(tmp_path)
        page_name = page_names()[0]

        result = get_website_page_mcp(str(site), page_name, max_chars=50)
        assert result["success"] is True
        assert result["truncated"] is True
        assert result["total_chars"] > 50
        assert result["content"].endswith(TRUNCATION_MARKER)
        assert len(result["content"]) == 50 + len(TRUNCATION_MARKER)

    @pytest.mark.unit
    def test_unknown_page_name_reports_error(self, tmp_path: Any) -> None:
        from gnn.website.mcp import get_website_page_mcp

        site = _build_site(tmp_path)
        result = get_website_page_mcp(str(site), "definitely_not_a_catalogue_page")
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.unit
    def test_missing_directory_reports_error(self, tmp_path: Any) -> None:
        from gnn.website.mcp import get_website_page_mcp
        from gnn.website.pages import page_names

        result = get_website_page_mcp(str(tmp_path / "missing"), page_names()[0])
        assert result["success"] is False
        assert "Directory not found" in result["error"]

    @pytest.mark.unit
    def test_missing_page_file_reports_error(self, tmp_path: Any) -> None:
        from gnn.website.inspection import read_website_page
        from gnn.website.pages import page_names

        empty_site = tmp_path / "partial_site"
        empty_site.mkdir()
        result = read_website_page(empty_site, page_names()[0])
        assert result["success"] is False
        assert f"{page_names()[0]}.html" in result["error"]
