"""Tests for gnn.website.pages — the one site page catalogue.

Pins that every page inventory derives from ``SITE_PAGES`` instead of a
private hardcoded list: the generator builders map, the inspection
key-page tuple, the module-info page/tool lists, and the generated site
shape. Also pins the stable receipt hook (``page_count``) used by
downstream dashboard lanes.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


EXPECTED_PAGE_NAMES: tuple[str, ...] = (
    "index",
    "pipeline",
    "gnn_files",
    "analysis",
    "visualization",
    "reports",
    "mcp",
)


class TestSitePagesCatalogue:
    """SITE_PAGES is the frozen, ordered description of the site's pages."""

    @pytest.mark.unit
    def test_catalogue_covers_seven_pages_in_order(self) -> None:
        from gnn.website.pages import SITE_PAGES, page_names

        assert page_names() == EXPECTED_PAGE_NAMES
        assert isinstance(SITE_PAGES, tuple)
        assert all(isinstance(page.filename, str) for page in SITE_PAGES)

    @pytest.mark.unit
    def test_specs_are_frozen_and_complete(self) -> None:
        from gnn.website.pages import SITE_PAGES

        for page in SITE_PAGES:
            assert page.name and page.title and page.builder
            assert page.description and page.icon
            assert page.filename == f"{page.name}.html"
            with pytest.raises((AttributeError, TypeError)):
                page.title = "mutated"  # type: ignore[misc]

    @pytest.mark.unit
    def test_helpers_and_receipt_hook(self) -> None:
        from gnn.website.pages import is_valid_page, page_count, page_names

        assert is_valid_page("index") is True
        assert is_valid_page("index.html") is False
        assert is_valid_page("no_such_page") is False
        assert page_count() == len(EXPECTED_PAGE_NAMES)
        assert page_names() == EXPECTED_PAGE_NAMES

    @pytest.mark.unit
    def test_every_builder_name_exists_on_generator(self) -> None:
        from gnn.website import WebsiteGenerator
        from gnn.website.pages import SITE_PAGES

        generator_methods = dir(WebsiteGenerator)
        for page in SITE_PAGES:
            assert page.builder in generator_methods, page.name


class TestSingleSourceDerivation:
    """The three historical inventories derive from SITE_PAGES."""

    @pytest.mark.unit
    def test_key_pages_derive_from_site_pages(self) -> None:
        from gnn.website.inspection import KEY_PAGES
        from gnn.website.pages import page_filenames

        assert KEY_PAGES == page_filenames()
        assert KEY_PAGES == tuple(f"{name}.html" for name in EXPECTED_PAGE_NAMES)

    @pytest.mark.unit
    def test_generator_builders_cover_site_pages(self) -> None:
        from gnn.website import WebsiteGenerator
        from gnn.website.pages import SITE_PAGES, page_filenames

        builders = WebsiteGenerator()._page_builders()
        assert set(builders) == set(page_filenames())
        for page in SITE_PAGES:
            bound = builders[page.filename]
            assert bound.__func__ is getattr(WebsiteGenerator, page.builder)

    @pytest.mark.unit
    def test_module_info_pages_derive_from_site_pages(self) -> None:
        from gnn.website.mcp import get_website_module_info_mcp
        from gnn.website.pages import page_names

        result = get_website_module_info_mcp()
        assert result["success"] is True
        assert result["pages"] == list(page_names())

    @pytest.mark.unit
    def test_module_info_tools_match_registration_order(
        self, test_mcp_tools: Any
    ) -> None:
        from gnn.website.mcp import get_website_module_info_mcp, register_tools

        register_tools(test_mcp_tools)
        result = get_website_module_info_mcp()
        assert result["success"] is True
        assert result["mcp_tools"] == list(test_mcp_tools.tools)

    @pytest.mark.unit
    def test_generated_site_matches_catalogue(self, tmp_path: Any) -> None:
        from gnn.website import WebsiteGenerator
        from gnn.website.pages import page_filenames

        result = WebsiteGenerator().generate_website(
            {"output_dir": str(tmp_path / "site"), "input_dir": str(tmp_path)}
        )
        assert result["success"] is True
        assert result["pages_created"] == len(EXPECTED_PAGE_NAMES)
        assert set(result["pages"]) == set(page_filenames())

    @pytest.mark.unit
    def test_pipeline_page_description_tracks_registry(self) -> None:
        from gnn.pipeline.step_registry import STEPS
        from gnn.website.pages import SITE_PAGES

        pipeline_page = next(page for page in SITE_PAGES if page.name == "pipeline")
        assert f"Full {len(STEPS)}-step pipeline status table." == (
            pipeline_page.description
        )
