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

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.parsers.common import ParseError
from gnn.parsers.lean_parser import LeanGNNParser
from gnn.parsers.markdown_parser import MarkdownGNNParser
from gnn.parsers.scala_parser import ScalaGNNParser


class TestGNNDiscovery:
    """Tests for GNN file discovery functionality."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_discovery_imports(self) -> None:
        """Test that discovery function can be imported from gnn package."""
        from gnn import discover_gnn_files

        assert callable(discover_gnn_files)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_discover_gnn_files_basic(self, isolated_temp_dir: Any) -> None:
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
    def test_discover_gnn_files_empty_directory(self, isolated_temp_dir: Any) -> None:
        """Test discovery in empty directory."""
        from gnn import discover_gnn_files

        empty_dir = isolated_temp_dir / "empty"
        empty_dir.mkdir()

        discovered_files = discover_gnn_files(empty_dir)

        assert isinstance(discovered_files, list)
        assert len(discovered_files) == 0

    @pytest.mark.unit
    @pytest.mark.fast
    def test_discover_gnn_files_nested(self, isolated_temp_dir: Any) -> None:
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
    def test_discover_gnn_files_nonexistent_directory(
        self, isolated_temp_dir: Any
    ) -> None:
        """Test discovery with nonexistent directory."""
        from gnn import discover_gnn_files

        nonexistent = isolated_temp_dir / "does_not_exist"

        # A nonexistent directory yields no models rather than raising.
        result = discover_gnn_files(nonexistent)
        assert result == []

    @pytest.mark.unit
    @pytest.mark.fast
    def test_discover_gnn_files_returns_paths(self, isolated_temp_dir: Any) -> None:
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
    def test_valid_parsing(self, sample_markdown: str) -> None:
        """Test parsing valid markdown GNN content."""
        parser = MarkdownGNNParser()
        result = parser.parse_string(sample_markdown)
        assert result.success, "Markdown GNN parsing should succeed"
        assert result.model.model_name == "TestModel"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_invalid_parsing(self) -> None:
        """Test parsing invalid content returns failure."""
        parser = MarkdownGNNParser()
        result = parser.parse_string("Invalid content without sections")
        assert not result.success
        assert len(result.errors) > 0

    @pytest.mark.unit
    @pytest.mark.fast
    def test_empty_string(self) -> Any:
        """Test parsing empty string."""
        parser = MarkdownGNNParser()
        result = parser.parse_string("")
        assert not result.success

    @pytest.mark.unit
    @pytest.mark.fast
    def test_whitespace_only(self) -> None:
        """Test parsing whitespace-only content."""
        parser = MarkdownGNNParser()
        result = parser.parse_string("   \n\n\t\t  ")
        assert not result.success

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parse_with_state_space_block(self) -> None:
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
        assert result.success, "StateSpaceBlock-only document must parse"
        assert [v.name for v in result.model.variables] == ["A", "B"]

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parse_with_connections(self) -> None:
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
        assert result.success, "Connections document must parse"
        assert [
            (c.source_variables, c.target_variables) for c in result.model.connections
        ] == [
            (["A"], ["B"]),
            (["B"], ["A"]),
        ]

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parse_annotated_edges_v11(self) -> None:
        """Test v1.1 ':annotation' suffix on edges (gnn_syntax.md section 3).

        An annotated directed edge ``A>B:label`` must resolve target ``A``
        and target ``B`` as variables, carry the annotation as a label,
        and NOT warn about an unknown target variable ``B:label``.
        """
        content = """## GNNSection
TestAnnotated

## GNNVersionAndFlags
GNN v1.1

## ModelName
Annotated Edge Test

## StateSpaceBlock
A[2,2,type=float]
B[2,2,2,type=float]
C[2,type=float]
D[2,type=float]
E[2,type=float]
pi[2,type=float]
u[2,type=float]
s[2,1,type=float]
o[2,1,type=float]

## Connections
D>s:prior_initialization
s-B
B>s
A-o:observation_mapping
E>u:select_action
"""
        parser = MarkdownGNNParser()
        result = parser.parse_string(content)
        assert result.success, "Annotated-edge document must parse"
        model = result.model

        by_pair = {
            (tuple(c.source_variables), tuple(c.target_variables)): c
            for c in model.connections
        }
        annotated = by_pair[("D",), ("s",)]
        assert annotated.annotation == "prior_initialization"
        assert annotated.target_variables == ["s"]
        assert by_pair[("A",), ("o",)].annotation == "observation_mapping"
        assert by_pair[("E",), ("u",)].annotation == "select_action"
        # Unannotated edges stay None.
        assert by_pair[("s",), ("B",)].annotation is None

        # The structural consequence of the fix: no unknown-target
        # warnings for annotated edges.
        declared = {v.name for v in model.variables}
        unknown = [
            t
            for c in model.connections
            for t in c.target_variables
            if t not in declared
        ]
        assert unknown == [], f"unknown targets after annotation strip: {unknown}"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_annotated_edge_serialization_round_trip(self) -> None:
        """An annotation survives a parse -> serialize -> parse round trip."""
        from gnn.parsers.markdown_serializer import MarkdownSerializer

        content = """## GNNSection
TestAnnRoundTrip

## GNNVersionAndFlags
GNN v1.1

## ModelName
Annotation Round Trip

## StateSpaceBlock
D[2,type=float]
s[2,type=float]

## Connections
D>s:prior_initialization
"""
        parser = MarkdownGNNParser()
        result = parser.parse_string(content)
        assert result.success
        serialized = MarkdownSerializer().serialize(result.model)
        assert "D>s:prior_initialization" in serialized
        reparsed = parser.parse_string(serialized)
        assert reparsed.success
        conn = reparsed.model.connections[0]
        assert conn.annotation == "prior_initialization"
        assert conn.target_variables == ["s"]

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parse_with_unicode(self) -> None:
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
        assert result.success, "Unicode content must parse"
        assert result.model.model_name == "UnicodeTest_αβγ"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parse_with_special_characters(self) -> None:
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
        assert result.success, "Special-character content must parse"
        assert result.model.model_name == "SpecialChars_Test-1.0"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parse_file(self, isolated_temp_dir: Any, sample_markdown: str) -> None:
        """Test parsing from file."""
        parser = MarkdownGNNParser()

        # Create test file
        test_file = isolated_temp_dir / "test_model.md"
        test_file.write_text(sample_markdown)

        result = parser.parse_file(test_file)
        assert result.success, "Markdown file parsing should succeed"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parse_nonexistent_file(self, isolated_temp_dir: Any) -> None:
        """Test parsing nonexistent file."""
        parser = MarkdownGNNParser()

        nonexistent = isolated_temp_dir / "does_not_exist.md"

        with pytest.raises(ParseError, match="Failed to read file"):
            parser.parse_file(nonexistent)


class TestScalaParser:
    """Tests for Scala GNN parser."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_valid_parsing(self, sample_scala: str) -> None:
        """Test parsing valid Scala GNN content."""
        parser = ScalaGNNParser()
        result = parser.parse_string(sample_scala)
        assert result.success, "Scala GNN parsing should succeed"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_empty_string(self) -> Any:
        """Test parsing empty string."""
        parser = ScalaGNNParser()
        result = parser.parse_string("")
        # Should handle gracefully
        assert result.success, "Empty content must yield a successful empty parse"
        assert result.model.variables == []

    @pytest.mark.unit
    @pytest.mark.fast
    def test_simple_scala(self) -> None:
        """Test parsing simple Scala content."""
        content = """
object TestModel {
  val states = Array(3)
  val observations = Array(2)
}
"""
        parser = ScalaGNNParser()
        result = parser.parse_string(content)
        assert result.success, "Simple Scala content must parse"
        assert result.model.model_name == "TestModel"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_scala_with_imports(self) -> None:
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
        assert result.success, "Scala content with imports must parse"
        assert result.model.model_name == "TestModel"


class TestLeanParser:
    """Tests for Lean GNN parser."""

    @pytest.fixture
    def sample_lean(self) -> str:
        return "def test := 42"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_valid_parsing(self, sample_lean: str) -> None:
        """Test parsing valid Lean content."""
        parser = LeanGNNParser()
        result = parser.parse_string(sample_lean)
        assert result.success, "Lean GNN parsing should succeed"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_empty_string(self) -> Any:
        """Test parsing empty string."""
        parser = LeanGNNParser()
        result = parser.parse_string("")
        assert result.success, "Empty content must yield a successful empty parse"
        assert result.model.variables == []

    @pytest.mark.unit
    @pytest.mark.fast
    def test_lean_definition(self) -> None:
        """Test parsing Lean definition."""
        content = """
def GNNModel : Type :=
  { states : Nat
  , observations : Nat
  }
"""
        parser = LeanGNNParser()
        result = parser.parse_string(content)
        assert result.success, "Lean definition must parse"
        assert result.model.model_name == "LeanGNNModel"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_lean_with_types(self) -> None:
        """Test parsing Lean with type annotations."""
        content = """
def stateSpace : Type := Fin 3
def obsSpace : Type := Fin 2

theorem model_valid : stateSpace → obsSpace → Prop := fun _ _ => True
"""
        parser = LeanGNNParser()
        result = parser.parse_string(content)
        assert result.success, "Lean type annotations must parse"
        assert [v.name for v in result.model.variables] == ["stateSpace", "obsSpace"]


class TestCoqParser:
    """Tests for Coq GNN parser."""

    @pytest.fixture
    def sample_coq(self) -> str:
        return "Definition test := 42."

    @pytest.mark.unit
    @pytest.mark.fast
    def test_valid_parsing(self, sample_coq: str) -> None:
        """Test parsing valid Coq content."""
        from gnn.parsers.coq_parser import CoqGNNParser

        parser = CoqGNNParser()
        result = parser.parse_string(sample_coq)
        assert result.success, "Coq GNN parsing should succeed"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_empty_string(self) -> Any:
        """Test parsing empty string."""
        from gnn.parsers.coq_parser import CoqGNNParser

        parser = CoqGNNParser()
        result = parser.parse_string("")
        assert result.success, "Empty content must yield a successful empty parse"
        assert result.model.variables == []

    @pytest.mark.unit
    @pytest.mark.fast
    def test_coq_inductive(self) -> None:
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
        assert result.success, "Coq inductive definition must parse"
        assert result.model.model_name == "CoqGNNModel"


class TestParserEdgeCases:
    """Tests for parser edge cases and error recovery."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_malformed_brackets(self) -> None:
        """Test handling malformed bracket content."""
        content = (
            "## StateSpaceBlock\nA[3,3,type=float\nB[2,2"  # Missing closing brackets
        )
        parser = MarkdownGNNParser()
        # The parser recovers from unclosed brackets instead of raising.
        result = parser.parse_string(content)
        assert result.success, "Unclosed brackets must not crash the parser"
        assert result.errors == [], "Recovery must not report errors"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_very_long_content(self) -> None:
        """Test handling very long content."""
        content = "## ModelName\nLongModel\n\n## StateSpaceBlock\n"
        content += "A" * 10000  # Very long variable name

        parser = MarkdownGNNParser()
        # Must complete without hanging or raising.
        result = parser.parse_string(content)
        assert result.success, "Very long content must parse"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_binary_content(self) -> None:
        """Test handling binary content."""
        binary_content = b"\x00\x01\x02\x03\xff\xfe"

        parser = MarkdownGNNParser()
        result = parser.parse_string(binary_content.decode("utf-8", errors="replace"))
        assert not result.success, "Binary soup must be rejected, not accepted"
        assert result.errors, "Failure must report errors"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_null_bytes(self) -> None:
        """Test handling content with null bytes."""
        content = "## ModelName\nTest\x00Model\n"

        parser = MarkdownGNNParser()
        result = parser.parse_string(content)
        assert result.success, "Null bytes must not crash the parser"
        assert result.model.model_name == "Test\x00Model"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_deeply_nested_sections(self) -> None:
        """Test handling deeply nested markdown sections."""
        content = "# Level 1\n## Level 2\n### Level 3\n#### Level 4\n##### Level 5\n###### Level 6\n"
        content += "## ModelName\nDeepModel"

        parser = MarkdownGNNParser()
        result = parser.parse_string(content)
        assert result.success, "Nested markdown sections must parse"
        assert result.model.model_name == "DeepModel"


class TestParserInstantiation:
    """Tests for parser class instantiation."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_markdown_parser_instantiation(self) -> None:
        """Test MarkdownGNNParser can be instantiated."""
        parser = MarkdownGNNParser()
        assert parser is not None

    @pytest.mark.unit
    @pytest.mark.fast
    def test_scala_parser_instantiation(self) -> None:
        """Test ScalaGNNParser can be instantiated."""
        parser = ScalaGNNParser()
        assert parser is not None

    @pytest.mark.unit
    @pytest.mark.fast
    def test_lean_parser_instantiation(self) -> None:
        """Test LeanGNNParser can be instantiated."""
        parser = LeanGNNParser()
        assert parser is not None

    @pytest.mark.unit
    @pytest.mark.fast
    def test_coq_parser_instantiation(self) -> None:
        """Test CoqGNNParser can be instantiated."""
        from gnn.parsers.coq_parser import CoqGNNParser

        parser = CoqGNNParser()
        assert parser is not None


class TestParameterParsing:
    """Regression tests for parameter value parsing and comment handling.

    Pins the two parser-invariant fixes:
    - an inline ``#`` comment on a braced matrix must not push the value into
      the Python-set literal path (sets are unordered and not JSON-serializable)
    - an inline ``#`` comment on a non-matrix token must be stripped from the
      value instead of leaking into it
    """

    @pytest.mark.unit
    @pytest.mark.fast
    def test_matrix_with_trailing_comment_stays_a_matrix(self) -> None:
        """A braced matrix with a trailing inline comment parses to a row list,
        never a Python set (which would break JSON export)."""
        value = (
            "A = { (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0) }  # I wrote this"
        )
        parser = MarkdownGNNParser()
        param = parser._parse_parameter_assignment(value)
        assert param is not None
        assert isinstance(param.value, list)
        assert param.value == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert param.description == "I wrote this"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_inline_comment_stripped_from_token_value(self) -> None:
        """A ``#`` comment after a bare token is removed from the parsed value."""
        parser = MarkdownGNNParser()
        param = parser._parse_parameter_assignment("alpha = scaling  # rise correction")
        assert param is not None
        assert param.value == "scaling"
        assert param.description == "rise correction"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_hash_inside_quoted_string_is_preserved(self) -> None:
        """A ``#`` inside a quoted string is data, not a comment delimiter."""
        parser = MarkdownGNNParser()
        param = parser._parse_parameter_assignment(
            'notes = "wave # 3"  # recording label'
        )
        assert param is not None
        assert param.value == "wave # 3"
        assert param.description == "recording label"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_hash_inside_matrix_row_is_preserved(self) -> None:
        """A ``#`` inside a ``{...}`` matrix block is a value-side comment yet the
        whole matrix still parses to rows, not a set."""
        parser = MarkdownGNNParser()
        value = "P = { (1.0, 0.0),  # first row\n          (0.0, 1.0) }  # done"
        param = parser._parse_parameter_assignment(value)
        assert param is not None
        assert isinstance(param.value, list)
        assert param.value == [[1.0, 0.0], [0.0, 1.0]]
        assert param.description == "done"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_json_round_trip_of_matrix_parameter(self) -> None:
        """A model whose parameters are matrices and inline-commented tokens must
        serialize through the JSON serializer without raising."""
        from gnn.parsers.json_serializer import JSONSerializer

        content = """## GNNSection
actinf

## ModelName
RT Model

## StateSpaceBlock
s[2]
o[3]

## Connections
s>o

## InitialParameterization
A = { (0.9, 0.1, 0.0), (0.0, 1.0, 0.0) }  # transition
rate = 0.1  # learning rate
"""
        parser = MarkdownGNNParser()
        result = parser.parse_string(content)
        assert result.success
        out = JSONSerializer().serialize(result.model)
        data = json.loads(out)
        params = {p["name"]: p["value"] for p in data["parameters"]}
        assert isinstance(params["A"], list)
        assert params["rate"] == 0.1


if __name__ == "__main__":
    pytest.main()
