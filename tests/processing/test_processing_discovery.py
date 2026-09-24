"""Tests for processing discovery: corpus filtering and GNN file discovery.

Covers: is_model_source_path, NON_MODEL_MARKDOWN_FILENAMES,
FileDiscoveryStrategy.discover, and discover_gnn_files.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestModelSourceFiltering:
    """Test the corpus-filter predicate behind discovery."""

    def test_is_model_source_path_filters_non_model_markdown(self) -> None:
        from gnn.processing import NON_MODEL_MARKDOWN_FILENAMES, is_model_source_path

        assert "readme.md" in NON_MODEL_MARKDOWN_FILENAMES
        assert is_model_source_path(Path("README.md")) is False
        assert is_model_source_path(Path("AGENTS.md")) is False
        assert is_model_source_path(Path("notes.template.md")) is False
        assert is_model_source_path(Path("demo.example.md")) is False
        assert is_model_source_path(Path("model.md")) is True


class TestFileDiscovery:
    """Test GNN file discovery over a small directory."""

    def test_file_discovery_finds_gnn_files_only(self, tmp_path: Any) -> None:
        from gnn.processing import FileDiscoveryStrategy, discover_gnn_files

        model = tmp_path / "model.md"
        model.write_text(
            "## ModelName\n\nDemo\n\n"
            "## StateSpaceBlock\n\ns: state\n\n"
            "## Connections\n\ns -> s\n"
        )
        (tmp_path / "README.md").write_text("Just a readme.\n")
        (tmp_path / "notes.template.md").write_text("Template doc.\n")

        strategy = FileDiscoveryStrategy()
        strategy.configure(target_extensions=[".md"])
        discovered = strategy.discover(tmp_path)
        assert sorted(p.name for p in discovered) == ["model.md"]

        listed = discover_gnn_files(tmp_path, recursive=False)
        assert sorted(p.name for p in listed) == ["model.md"]
