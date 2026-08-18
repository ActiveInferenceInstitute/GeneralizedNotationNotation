"""Tests for ontology module's public API surface not covered by existing tests.

Covers: process_ontology, process_gnn_ontology, generate_ontology_report_for_file,
validate_annotations, validate_ontology_terms (extended), get_mcp_interface,
FEATURES, __version__.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestOntologyConstants:
    """Test module-level constants."""

    def test_features_dict(self) -> None:
        import ontology

        assert hasattr(ontology, "FEATURES")
        assert isinstance(ontology.FEATURES, dict)
        for key in (
            "parsing",
            "validation",
            "reporting",
            "basic_processing",
            "mcp_integration",
        ):
            assert key in ontology.FEATURES

    def test_version(self) -> None:
        import ontology

        assert hasattr(ontology, "__version__")
        assert isinstance(ontology.__version__, str)


class TestProcessOntology:
    """Test process_ontology function."""

    def test_process_empty_dir(self, tmp_path: Path) -> None:
        from ontology import process_ontology

        target = tmp_path / "empty"
        target.mkdir()
        out = tmp_path / "out"
        result = process_ontology(target_dir=target, output_dir=out)
        assert isinstance(result, bool)

    def test_process_nonexistent_dir(self, tmp_path: Path) -> None:
        from ontology import process_ontology

        out = tmp_path / "out"
        result = process_ontology(
            target_dir=tmp_path / "nonexistent",
            output_dir=out,
        )
        assert isinstance(result, bool)

    def test_process_with_sample_file(self, sample_gnn_file: Any) -> None:
        from ontology import process_ontology

        target_dir = sample_gnn_file.parent
        output_dir = target_dir / "ontology_out"
        result = process_ontology(target_dir=target_dir, output_dir=output_dir)
        assert isinstance(result, bool)

    def test_process_with_strict_validation(self, sample_gnn_file: Any) -> None:
        from ontology import process_ontology

        output_dir = sample_gnn_file.parent / "strict_out"
        result = process_ontology(
            target_dir=sample_gnn_file.parent,
            output_dir=output_dir,
            strict_validation=True,
        )
        assert isinstance(result, bool)


class TestProcessGnnOntology:
    """Test process_gnn_ontology function."""

    def test_process_gnn_ontology_with_real_file(self, sample_gnn_file: Any) -> None:
        from ontology import process_gnn_ontology

        result = process_gnn_ontology(str(sample_gnn_file))
        assert isinstance(result, dict)
        # Should contain success or result key
        assert len(result) > 0

    def test_process_gnn_ontology_nonexistent_file(self) -> None:
        from ontology import process_gnn_ontology

        result = process_gnn_ontology("/nonexistent/file.md")
        assert isinstance(result, dict)


class TestGenerateOntologyReport:
    """Test generate_ontology_report_for_file."""

    def test_generate_report_with_sample(
        self, sample_gnn_file: Any, tmp_path: Path
    ) -> None:
        from ontology import generate_ontology_report_for_file

        result = generate_ontology_report_for_file(sample_gnn_file, tmp_path)
        assert isinstance(result, dict)

    def test_generate_report_nonexistent(self, tmp_path: Path) -> None:
        from ontology import generate_ontology_report_for_file

        result = generate_ontology_report_for_file(
            tmp_path / "nonexistent.md", tmp_path / "out"
        )
        assert isinstance(result, dict)


class TestValidateAnnotationsExtended:
    """Extended tests for validate_annotations."""

    def test_validate_annotations_empty_list(self) -> None:
        from ontology import validate_annotations

        result = validate_annotations([])
        assert isinstance(result, dict)

    def test_validate_annotations_valid_terms(self) -> None:
        from ontology import validate_annotations

        result = validate_annotations(["s=HiddenState", "o=Observation"])
        assert isinstance(result, dict)

    def test_validate_annotations_with_explicit_terms(self) -> None:
        from ontology import validate_annotations

        terms: dict[str, Any] = {"HiddenState": {}, "Observation": {}}
        result = validate_annotations(["s=HiddenState"], terms)
        assert isinstance(result, dict)

    def test_validate_ontology_terms_with_list(self) -> None:
        from ontology import validate_ontology_terms

        result = validate_ontology_terms(["HiddenState", "Observation"])
        assert isinstance(result, bool)

    def test_validate_ontology_terms_with_string(self) -> None:
        from ontology import validate_ontology_terms

        result = validate_ontology_terms("HiddenState")
        assert isinstance(result, bool)

    def test_validate_ontology_terms_none(self) -> None:
        from ontology import validate_ontology_terms

        result = validate_ontology_terms()
        assert result is True


class TestGetMcpInterface:
    """Test get_mcp_interface function."""

    def test_get_mcp_interface_returns_dict(self) -> None:
        from ontology import get_mcp_interface

        result = get_mcp_interface()
        assert isinstance(result, dict)
