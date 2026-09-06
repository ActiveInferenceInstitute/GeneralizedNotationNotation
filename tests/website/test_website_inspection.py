"""Tests for website.inspection — pure queries over a generated site.

Pins the ``inspect_website`` / ``list_website_pages`` contracts shared by
the Python API and the MCP tools, including their behavior on complete,
partial, and missing websites. Deterministic and filesystem-only.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _build_site(tmp_path: Any) -> Path:
    """Build a real 7-page website from an empty input dir."""
    from gnn.website import WebsiteGenerator

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    out: Path = tmp_path / "site"
    result = WebsiteGenerator(mcp_tools_provider=list).generate_website(
        {
            "input_dir": str(input_dir),
            "output_dir": str(out),
            "pipeline_output_root": str(tmp_path),
        }
    )
    assert result["success"] is True
    assert result["pages_created"] == 7
    return out


class TestKeyPages:
    @pytest.mark.unit
    def test_key_pages_cover_the_seven_built_pages_in_order(self) -> None:
        from gnn.website.inspection import KEY_PAGES

        assert KEY_PAGES == (
            "index.html",
            "pipeline.html",
            "gnn_files.html",
            "analysis.html",
            "visualization.html",
            "reports.html",
            "mcp.html",
        )


class TestInspectWebsite:
    @pytest.mark.unit
    def test_missing_directory_reports_error(self, tmp_path: Any) -> None:
        from gnn.website import inspect_website

        result = inspect_website(tmp_path / "missing")
        assert result["success"] is False
        assert "Directory not found" in result["error"]

    @pytest.mark.unit
    def test_complete_site_reports_all_key_pages(self, tmp_path: Any) -> None:
        from gnn.website import inspect_website

        site = _build_site(tmp_path)
        result = inspect_website(site)

        assert result["success"] is True
        assert result["directory"] == str(site)
        assert result["pages_count"] == 7
        assert result["all_key_pages_present"] is True
        assert set(result["completeness"]) == set(result["pages"])
        assert result["total_size_bytes"] > 0
        assert "assets_count" in result

    @pytest.mark.unit
    def test_partial_site_flags_missing_key_pages(self, tmp_path: Any) -> None:
        from gnn.website import inspect_website

        site = _build_site(tmp_path)
        (site / "analysis.html").unlink()

        result = inspect_website(site)

        assert result["success"] is True
        assert result["pages_count"] == 6
        assert result["completeness"]["analysis.html"] is False
        assert result["completeness"]["index.html"] is True
        assert result["all_key_pages_present"] is False


class TestListWebsitePages:
    @pytest.mark.unit
    def test_missing_directory_reports_error(self, tmp_path: Any) -> None:
        from gnn.website import list_website_pages

        result = list_website_pages(tmp_path / "missing")
        assert result["success"] is False
        assert "Directory not found" in result["error"]

    @pytest.mark.unit
    def test_lists_every_page_with_metadata(self, tmp_path: Any) -> None:
        from gnn.website import list_website_pages

        site = _build_site(tmp_path)
        result = list_website_pages(site)

        assert result["success"] is True
        assert result["total_pages"] == 7
        names = [p["name"] for p in result["pages"]]
        assert names == sorted(names)  # deterministic order
        for page in result["pages"]:
            assert page["size_bytes"] > 0
            # modified must round-trip as ISO-8601
            datetime.fromisoformat(page["modified"])

    @pytest.mark.unit
    def test_empty_directory_lists_zero_pages(self, tmp_path: Any) -> None:
        from gnn.website import list_website_pages

        empty = tmp_path / "empty"
        empty.mkdir()
        result = list_website_pages(empty)

        assert result["success"] is True
        assert result["pages"] == []
        assert result["total_pages"] == 0
