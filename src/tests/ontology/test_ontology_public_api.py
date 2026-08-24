"""Tests for ontology module's public API surface not covered by existing tests.

Covers: process_ontology, process_gnn_ontology, generate_ontology_report_for_file,
validate_annotations, validate_ontology_terms (extended), get_mcp_interface,
FEATURES, __version__.
"""

import json
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

    def test_strict_validation_fails_on_unknown_annotation(
        self, tmp_path: Path
    ) -> None:
        from ontology import process_ontology

        target = tmp_path / "input"
        target.mkdir()
        (target / "invalid.md").write_text(
            "## ActInfOntologyAnnotation\ns=DefinitelyNotAnOntologyTerm\n",
            encoding="utf-8",
        )

        assert (
            process_ontology(target, tmp_path / "strict", strict_validation=True)
            is False
        )
        assert process_ontology(target, tmp_path / "lenient") is True

    def test_custom_ontology_file_is_authoritative_for_strict_validation(
        self, tmp_path: Path
    ) -> None:
        from ontology import process_ontology

        target = tmp_path / "input"
        target.mkdir()
        (target / "custom.md").write_text(
            "## ActInfOntologyAnnotation\nx=NovelLatentState\n",
            encoding="utf-8",
        )
        terms_file = tmp_path / "terms.json"
        terms_file.write_text(
            json.dumps({"custom": ["NovelLatentState"]}), encoding="utf-8"
        )
        output = tmp_path / "output"

        assert process_ontology(
            target,
            output,
            strict_validation=True,
            ontology_terms_file=terms_file,
        )
        receipt = json.loads(
            (output / "ontology_results.json").read_text(encoding="utf-8")
        )
        report = json.loads(Path(receipt["reports"][0]).read_text(encoding="utf-8"))
        assert report["validation_result"]["valid_annotations"] == [
            "x=NovelLatentState"
        ]
        assert report["validation_result"]["coverage_score"] == 1.0

    def test_missing_custom_ontology_fails_closed_with_receipt(
        self, tmp_path: Path
    ) -> None:
        from ontology import process_ontology

        target = tmp_path / "input"
        target.mkdir()
        (target / "model.md").write_text(
            "## ActInfOntologyAnnotation\ns=HiddenState\n", encoding="utf-8"
        )
        output = tmp_path / "output"

        assert (
            process_ontology(
                target,
                output,
                strict_validation=True,
                ontology_terms_file=tmp_path / "missing.json",
            )
            is False
        )
        receipt = json.loads(
            (output / "ontology_results.json").read_text(encoding="utf-8")
        )
        assert receipt["success"] is False
        assert receipt["errors"][0]["error_type"] == "ontology_terms_load_error"

    def test_recursive_duplicate_stems_emit_distinct_reports(
        self, tmp_path: Path
    ) -> None:
        from ontology import process_ontology

        target = tmp_path / "input"
        for branch, term in (("a", "HiddenState"), ("b", "Observation")):
            nested = target / branch
            nested.mkdir(parents=True)
            (nested / "model.md").write_text(
                f"## ActInfOntologyAnnotation\nx={term}\n", encoding="utf-8"
            )
        output = tmp_path / "output"

        assert process_ontology(target, output, strict_validation=True)
        receipt = json.loads(
            (output / "ontology_results.json").read_text(encoding="utf-8")
        )

        relative_reports = [
            Path(report).relative_to(output).as_posix() for report in receipt["reports"]
        ]
        assert relative_reports == [
            "a/model_ontology_report.json",
            "b/model_ontology_report.json",
        ]
        assert all(Path(report).is_file() for report in receipt["reports"])

    def test_strict_processing_is_deterministic_on_committed_model(
        self, tmp_path: Path
    ) -> None:
        from ontology import process_ontology

        repo_root = Path(__file__).parents[3]
        target = repo_root / "input/gnn_files/pomdp_gridworld"
        first = tmp_path / "first"
        second = tmp_path / "second"

        assert process_ontology(target, first, strict_validation=True, recursive=False)
        assert process_ontology(target, second, strict_validation=True, recursive=False)

        first_receipt = json.loads(
            (first / "ontology_results.json").read_text(encoding="utf-8")
        )
        second_receipt = json.loads(
            (second / "ontology_results.json").read_text(encoding="utf-8")
        )
        first_reports = first_receipt.pop("reports")
        second_reports = second_receipt.pop("reports")
        assert first_receipt == second_receipt
        assert len(first_reports) == len(second_reports) > 0
        for first_report, second_report in zip(
            first_reports, second_reports, strict=True
        ):
            assert Path(first_report).read_bytes() == Path(second_report).read_bytes()


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

    def test_validate_annotations_rejects_incomplete_mapping(self) -> None:
        from ontology import validate_annotations

        result = validate_annotations(["=HiddenState"])

        assert result["valid_annotations"] == []
        assert result["invalid_annotations"] == ["=HiddenState"]
        assert result["invalid_details"][0]["reason"] == (
            "mapping annotations require a key and a value"
        )

    def test_validate_annotations_rejects_conflicting_key_mapping(self) -> None:
        from ontology import validate_annotations

        result = validate_annotations(["s=HiddenState", "s=Observation"])

        assert result["valid_annotations"] == ["s=HiddenState"]
        assert result["invalid_annotations"] == ["s=Observation"]
        assert result["invalid_details"][0]["reason"] == (
            "annotation key maps to multiple ontology terms"
        )

    def test_validate_ontology_terms_fails_closed_on_invalid_input(self) -> None:
        from ontology import validate_ontology_terms

        assert validate_ontology_terms(42) is False  # type: ignore[arg-type]

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
