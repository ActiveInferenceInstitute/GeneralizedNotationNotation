"""Tests for the gnn.parsers public API surface.

Covers: has_frontmatter, parse_frontmatter, get_parse_tree_visualization,
MarkdownGNNParser.parse_string for small deterministic GNN inputs.
"""

import sys
from pathlib import Path

FRONTMATTER_DOC = "---\ntitle: Demo\nversion: 1.0\n---\n## Body\nx"

GNN_SNIPPET = (
    "## ModelName\n"
    "Tiny Agent\n"
    "\n"
    "## StateSpaceBlock\n"
    "s_f0[2,1,type=int]\n"
    "o_m0[2,1,type=int]\n"
    "\n"
    "## Connections\n"
    "s_f0>o_m0\n"
)


class TestFrontmatterHelpers:
    """Test gnn.parsers.frontmatter string helpers."""

    def test_frontmatter_helpers(self) -> None:
        from gnn.parsers import has_frontmatter, parse_frontmatter

        assert has_frontmatter(FRONTMATTER_DOC) is True
        assert has_frontmatter(GNN_SNIPPET) is False

        metadata, remaining = parse_frontmatter(FRONTMATTER_DOC)
        assert metadata["title"] == "Demo"
        assert metadata["version"] == 1.0
        assert remaining == "## Body\nx"

        metadata_only, content_only = parse_frontmatter(GNN_SNIPPET)
        assert metadata_only == {}
        assert content_only == GNN_SNIPPET


class TestMarkdownParsing:
    """Test markdown parsing through the public facade."""

    def test_parse_string_markdown_extracts_model(self) -> None:
        from gnn.parsers import MarkdownGNNParser

        result = MarkdownGNNParser().parse_string(GNN_SNIPPET)
        assert result.success is True
        assert result.errors == []
        assert result.model.model_name == "Tiny Agent"
        names = {variable.name for variable in result.model.variables}
        assert names == {"s_f0", "o_m0"}
        connection = result.model.connections[0]
        assert connection.source_variables == ["s_f0"]
        assert connection.target_variables == ["o_m0"]

    def test_parse_string_empty_content_reports_error(self) -> None:
        from gnn.parsers import MarkdownGNNParser

        result = MarkdownGNNParser().parse_string("")
        assert result.success is False
        assert result.has_errors() is True
        assert "Empty content" in result.errors

    def test_parse_tree_visualization_outline(self) -> None:
        from gnn.parsers import get_parse_tree_visualization

        outline = get_parse_tree_visualization(GNN_SNIPPET)
        assert "Sections: 3" in outline
        for section in ("ModelName", "StateSpaceBlock", "Connections"):
            assert f"- {section}" in outline
