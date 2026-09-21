"""Unit tests for the website↔GUI cross-links (F9).

Pins: ``collect_website_data`` reporting ``gui_navigation`` only when
``22_gui_output/navigation.html`` exists, the visualization page rendering the
GUI card only when that flag is set, and ``generate_html_navigation`` adding a
reciprocal footer link to the website only when
``20_website_output/index.html`` exists. All three behaviors fail pre-fix: no
flag is collected, no card is rendered, and no reciprocal link is written.
Deterministic and filesystem-only — no network, no gradio.
"""

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _collect(tmp_path: Any) -> dict[str, Any]:
    from gnn.website import collect_website_data

    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(exist_ok=True)
    return collect_website_data(
        tmp_path, input_dir, assets_dir, output_dir=tmp_path / "site"
    )


class TestCollectGuiNavigationFlag:
    """collect_website_data reports gui_navigation from the step-22 artifact."""

    def test_absent_navigation_html_is_false(self, tmp_path: Any) -> None:
        assert _collect(tmp_path)["gui_navigation"] is False

    def test_present_navigation_html_is_true(self, tmp_path: Any) -> None:
        gui_dir = tmp_path / "22_gui_output"
        gui_dir.mkdir()
        (gui_dir / "navigation.html").write_text("<html></html>")

        assert _collect(tmp_path)["gui_navigation"] is True


class TestVisualizationPageGuiCard:
    """_page_visualization renders the GUI card only when gui_navigation is set."""

    @staticmethod
    def _render(gui_navigation: bool) -> str:
        from gnn.website import WebsiteGenerator

        data: dict[str, Any] = {
            "visualizations": [],
            "step_statuses": {},
            "gui_navigation": gui_navigation,
            "processed_files": 0,
            "mcp_summary": {},
        }
        return WebsiteGenerator()._page_visualization(data)

    def test_card_present_when_gui_navigation(self) -> None:
        html = self._render(True)
        assert "../22_gui_output/navigation.html" in html
        assert "Interactive GUI navigation" in html

    def test_no_card_without_gui_navigation(self) -> None:
        html = self._render(False)
        assert "../22_gui_output/navigation.html" not in html
        assert "Interactive GUI navigation" not in html


class TestNavigationReciprocalWebsiteLink:
    """generate_html_navigation links back to the website only when it exists."""

    @staticmethod
    def _render(tmp_path: Any) -> str:
        from gnn.gui import generate_html_navigation

        output_dir = tmp_path / "22_gui_output"
        assert generate_html_navigation(
            tmp_path, output_dir, logging.getLogger("test_crosslink")
        )
        return (output_dir / "navigation.html").read_text()

    def test_no_website_link_without_site(self, tmp_path: Any) -> None:
        assert "20_website_output" not in self._render(tmp_path)

    def test_website_link_with_site(self, tmp_path: Any) -> None:
        site = tmp_path / "20_website_output"
        site.mkdir()
        (site / "index.html").write_text("<html></html>")

        content = self._render(tmp_path)
        assert "../20_website_output/index.html" in content
        assert "Open Website Dashboard" in content
        # the pre-existing report link stays untouched
        assert "23_report_output/comprehensive_analysis_report.html" in content
