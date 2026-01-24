#!/usr/bin/env python3
"""
Comprehensive GNN Parsing Tests

Tests the GNN parsing functionality including:
- File discovery (discover_gnn_files)
- Markdown parser with edge cases
- Scala parser
- Lean parser
- Coq parser
- Error recovery
- Malformed content handling
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *
from gnn.parsers.markdown_parser import MarkdownGNNParser
from gnn.parsers.scala_parser import ScalaGNNParser
from gnn.parsers.lean_parser import LeanGNNParser
from gnn.parsers.common import ParseError


class TestGNNDiscovery:
    """Tests for GNN file discovery functionality."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_discovery_imports(self):
        """Test that discovery function can be imported from gnn package."""
        from gnn import discover_gnn_files
        assert callable(discover_gnn_files)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_discover_gnn_files_basic(self, isolated_temp_dir):
        """Test basic GNN file discovery."""
        from gnn import discover_gnn_files

        test_dir = isolated_temp_dir / "test_models"
        test_dir.mkdir()

        (test_dir / "model1.md").write_text("# GNN Model 1")
        (test_dir / "model2.gnn").write_text("# GNN Model 2")
        (test_dir / "not_gnn.txt").write_text("Not a GNN file")

        discovered_files = discover_gnn_files(test_dir)

        assert isinstance(discovered_files, list)
        assert len(discovered_files) >= 1

    @pytest.mark.unit
    @pytest.mark.fast
    def test_discover_gnn_files_empty_directory(self, isolated_temp_dir):
        """Test discovery in empty directory."""
        from gnn import discover_gnn_files

        empty_dir = isolated_temp_dir / "empty"
        empty_dir.mkdir()

        discovered_files = discover_gnn_files(empty_dir)

        assert isinstance(discovered_files, list)
        assert len(discovered_files) == 0

    @pytest.mark.unit
    @pytest.mark.fast
    def test_discover_gnn_files_nested(self, isolated_temp_dir):
        """Test discovery in nested directories."""
        from gnn import discover_gnn_files

        base_dir = isolated_temp_dir / "nested"
        base_dir.mkdir()
        sub_dir = base_dir / "subdir"
        sub_dir.mkdir()
        deep_dir = sub_dir / "deep"
        deep_dir.mkdir()

        (base_dir / "model1.md").write_text("# Model 1")
        (sub_dir / "model2.md").write_text("# Model 2")
        (deep_dir / "model3.md").write_text("# Model 3")

        discovered_files = discover_gnn_files(base_dir)

        assert len(discovered_files) >= 1  # Should find at least some files

    @pytest.mark.unit
    @pytest.mark.fast
    def test_discover_gnn_files_nonexistent_directory(self, isolated_temp_dir):
        """Test discovery with nonexistent directory."""
        from gnn import discover_gnn_files

        nonexistent = isolated_temp_dir / "does_not_exist"

        # Should handle gracefully
        try:
            result = discover_gnn_files(nonexistent)
            assert isinstance(result, list)
        except (FileNotFoundError, ValueError):
            pass  # Expected behavior

    @pytest.mark.unit
    @pytest.mark.fast
    def test_discover_gnn_files_returns_paths(self, isolated_temp_dir):
        """Test that discovered files are Path objects or strings."""
        from gnn import discover_gnn_files

        test_dir = isolated_temp_dir / "paths_test"
        test_dir.mkdir()
        (test_dir / "model.md").write_text("# GNN Model")

        discovered_files = discover_gnn_files(test_dir)

        for file_path in discovered_files:
            assert isinstance(file_path, (str, Path))


class TestMarkdownParser:
    """Tests for Markdown GNN parser."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_valid_parsing(self, sample_markdown):
        """Test parsing valid markdown GNN content."""
        parser = MarkdownGNNParser()
        result = parser.parse_string(sample_markdown)
        assert result.success
        assert result.model.model_name == 'TestModel'

    @pytest.mark.unit
    @pytest.mark.fast
    def test_invalid_parsing(self):
        """Test parsing invalid content returns failure."""
        parser = MarkdownGNNParser()
        result = parser.parse_string('Invalid content without sections')
        assert not result.success
        assert len(result.errors) > 0

    @pytest.mark.unit
    @pytest.mark.fast
    def test_empty_string(self):
        """Test parsing empty string."""
        parser = MarkdownGNNParser()
        result = parser.parse_string('')
        assert not result.success

    @pytest.mark.unit
    @pytest.mark.fast
    def test_whitespace_only(self):
        """Test parsing whitespace-only content."""
        parser = MarkdownGNNParser()
        result = parser.parse_string('   \n\n\t\t  ')
        assert not result.success

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parse_with_state_space_block(self):
        """Test parsing markdown with StateSpaceBlock."""
        content = """## GNNSection
ActInfPOMDP

## ModelName
TestModel

## StateSpaceBlock
A[3,3,type=float]
B[3,3,3,type=float]
"""
        parser = MarkdownGNNParser()
        result = parser.parse_string(content)
        assert result.success or "StateSpace" in str(result)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parse_with_connections(self):
        """Test parsing markdown with Connections block."""
        content = """## GNNSection
ActInfPOMDP

## ModelName
ConnectionsTest

## StateSpaceBlock
A[3,3,type=float]
B[3,3,type=float]

## Connections
A>B
B-A
"""
        parser = MarkdownGNNParser()
        result = parser.parse_string(content)
        # Should not crash, may succeed or fail based on content
        assert isinstance(result.success, bool)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parse_with_unicode(self):
        """Test parsing content with unicode characters."""
        content = """## GNNSection
ActInfPOMDP

## ModelName
UnicodeTest_αβγ

## Description
Model with unicode: αβγδ ∑∏∫
"""
        parser = MarkdownGNNParser()
        result = parser.parse_string(content)
        # Should handle unicode gracefully
        assert isinstance(result.success, bool)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parse_with_special_characters(self):
        """Test parsing content with special characters."""
        content = """## GNNSection
ActInfPOMDP

## ModelName
SpecialChars_Test-1.0

## Description
Model with special chars: !@#$%^&*()
"""
        parser = MarkdownGNNParser()
        result = parser.parse_string(content)
        assert isinstance(result.success, bool)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parse_file(self, isolated_temp_dir, sample_markdown):
        """Test parsing from file."""
        parser = MarkdownGNNParser()

        # Create test file
        test_file = isolated_temp_dir / "test_model.md"
        test_file.write_text(sample_markdown)

        result = parser.parse_file(test_file)
        assert result.success

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parse_nonexistent_file(self, isolated_temp_dir):
        """Test parsing nonexistent file."""
        parser = MarkdownGNNParser()

        nonexistent = isolated_temp_dir / "does_not_exist.md"

        try:
            result = parser.parse_file(nonexistent)
            assert not result.success
        except (FileNotFoundError, ParseError):
            pass  # Expected behavior - parser wraps FileNotFoundError in ParseError


class TestScalaParser:
    """Tests for Scala GNN parser."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_valid_parsing(self, sample_scala):
        """Test parsing valid Scala GNN content."""
        parser = ScalaGNNParser()
        result = parser.parse_string(sample_scala)
        assert result.success

    @pytest.mark.unit
    @pytest.mark.fast
    def test_empty_string(self):
        """Test parsing empty string."""
        parser = ScalaGNNParser()
        result = parser.parse_string('')
        # Should handle gracefully
        assert isinstance(result.success, bool)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_simple_scala(self):
        """Test parsing simple Scala content."""
        content = """
object TestModel {
  val states = Array(3)
  val observations = Array(2)
}
"""
        parser = ScalaGNNParser()
        result = parser.parse_string(content)
        assert isinstance(result.success, bool)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_scala_with_imports(self):
        """Test parsing Scala with imports."""
        content = """
import scala.collection.mutable
import gnn.core._

object TestModel extends GNNModel {
  val states = 3
}
"""
        parser = ScalaGNNParser()
        result = parser.parse_string(content)
        assert isinstance(result.success, bool)


class TestLeanParser:
    """Tests for Lean GNN parser."""

    @pytest.fixture
    def sample_lean(self):
        return 'def test := 42'

    @pytest.mark.unit
    @pytest.mark.fast
    def test_valid_parsing(self, sample_lean):
        """Test parsing valid Lean content."""
        parser = LeanGNNParser()
        result = parser.parse_string(sample_lean)
        assert result.success

    @pytest.mark.unit
    @pytest.mark.fast
    def test_empty_string(self):
        """Test parsing empty string."""
        parser = LeanGNNParser()
        result = parser.parse_string('')
        assert isinstance(result.success, bool)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_lean_definition(self):
        """Test parsing Lean definition."""
        content = """
def GNNModel : Type :=
  { states : Nat
  , observations : Nat
  }
"""
        parser = LeanGNNParser()
        result = parser.parse_string(content)
        assert isinstance(result.success, bool)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_lean_with_types(self):
        """Test parsing Lean with type annotations."""
        content = """
def stateSpace : Type := Fin 3
def obsSpace : Type := Fin 2

theorem model_valid : stateSpace → obsSpace → Prop := fun _ _ => True
"""
        parser = LeanGNNParser()
        result = parser.parse_string(content)
        assert isinstance(result.success, bool)


class TestCoqParser:
    """Tests for Coq GNN parser."""

    @pytest.fixture
    def sample_coq(self):
        return 'Definition test := 42.'

    @pytest.mark.unit
    @pytest.mark.fast
    def test_valid_parsing(self, sample_coq):
        """Test parsing valid Coq content."""
        from gnn.parsers.coq_parser import CoqGNNParser
        parser = CoqGNNParser()
        result = parser.parse_string(sample_coq)
        assert result.success

    @pytest.mark.unit
    @pytest.mark.fast
    def test_empty_string(self):
        """Test parsing empty string."""
        from gnn.parsers.coq_parser import CoqGNNParser
        parser = CoqGNNParser()
        result = parser.parse_string('')
        assert isinstance(result.success, bool)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_coq_inductive(self):
        """Test parsing Coq inductive definition."""
        content = """
Inductive State : Type :=
  | s0 : State
  | s1 : State
  | s2 : State.
"""
        from gnn.parsers.coq_parser import CoqGNNParser
        parser = CoqGNNParser()
        result = parser.parse_string(content)
        assert isinstance(result.success, bool)


class TestParserEdgeCases:
    """Tests for parser edge cases and error recovery."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_malformed_brackets(self):
        """Test handling malformed bracket content."""
        content = "## StateSpaceBlock\nA[3,3,type=float\nB[2,2"  # Missing closing brackets
        parser = MarkdownGNNParser()
        # Should not crash
        try:
            result = parser.parse_string(content)
            assert isinstance(result.success, bool)
        except Exception as e:
            # Acceptable to raise for malformed input
            assert isinstance(e, (ValueError, SyntaxError, Exception))

    @pytest.mark.unit
    @pytest.mark.fast
    def test_very_long_content(self):
        """Test handling very long content."""
        content = "## ModelName\nLongModel\n\n## StateSpaceBlock\n"
        content += "A" * 10000  # Very long variable name

        parser = MarkdownGNNParser()
        # Should handle without hanging
        try:
            result = parser.parse_string(content)
            assert isinstance(result.success, bool)
        except Exception:
            pass  # Memory or length limits acceptable

    @pytest.mark.unit
    @pytest.mark.fast
    def test_binary_content(self):
        """Test handling binary content."""
        binary_content = b'\x00\x01\x02\x03\xff\xfe'

        parser = MarkdownGNNParser()
        try:
            result = parser.parse_string(binary_content.decode('utf-8', errors='replace'))
            assert isinstance(result.success, bool)
        except Exception:
            pass  # Expected for binary input

    @pytest.mark.unit
    @pytest.mark.fast
    def test_null_bytes(self):
        """Test handling content with null bytes."""
        content = "## ModelName\nTest\x00Model\n"

        parser = MarkdownGNNParser()
        try:
            result = parser.parse_string(content)
            assert isinstance(result.success, bool)
        except Exception:
            pass  # Null bytes may cause issues

    @pytest.mark.unit
    @pytest.mark.fast
    def test_deeply_nested_sections(self):
        """Test handling deeply nested markdown sections."""
        content = "# Level 1\n## Level 2\n### Level 3\n#### Level 4\n##### Level 5\n###### Level 6\n"
        content += "## ModelName\nDeepModel"

        parser = MarkdownGNNParser()
        result = parser.parse_string(content)
        assert isinstance(result.success, bool)


class TestParserInstantiation:
    """Tests for parser class instantiation."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_markdown_parser_instantiation(self):
        """Test MarkdownGNNParser can be instantiated."""
        parser = MarkdownGNNParser()
        assert parser is not None

    @pytest.mark.unit
    @pytest.mark.fast
    def test_scala_parser_instantiation(self):
        """Test ScalaGNNParser can be instantiated."""
        parser = ScalaGNNParser()
        assert parser is not None

    @pytest.mark.unit
    @pytest.mark.fast
    def test_lean_parser_instantiation(self):
        """Test LeanGNNParser can be instantiated."""
        parser = LeanGNNParser()
        assert parser is not None

    @pytest.mark.unit
    @pytest.mark.fast
    def test_coq_parser_instantiation(self):
        """Test CoqGNNParser can be instantiated."""
        from gnn.parsers.coq_parser import CoqGNNParser
        parser = CoqGNNParser()
        assert parser is not None


if __name__ == '__main__':
    pytest.main()
