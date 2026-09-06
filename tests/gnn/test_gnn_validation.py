#!/usr/bin/env python3
"""
Test Gnn Validation Tests

This file contains tests migrated from test_gnn_core_modules.py.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Migrated from test_gnn_core_modules.py
class TestGNNValidation:
    """Test gnn.validation module."""

    @pytest.mark.unit
    def test_validation_imports(self) -> Any:
        """Test that validation module can be imported."""
        from gnn import validate_gnn_structure

        assert callable(validate_gnn_structure)

    @pytest.mark.unit
    def test_validate_gnn_structure_basic(self, sample_gnn_files: Any) -> Any:
        """Test GNN structure validation."""
        from gnn import validate_gnn_structure

        # Get sample file content
        sample_file = list(sample_gnn_files.values())[0]
        content = sample_file.read_text()

        result = validate_gnn_structure(content)

        # Verify validation result structure
        assert isinstance(result, dict)


# Migrated from test_gnn_core_modules.py
class TestGNNSimpleValidator:
    """Test gnn.simple_validator module."""

    @pytest.mark.unit
    def test_simple_validator_imports(self) -> Any:
        """Test that simple validator can be imported."""
        from gnn import simple_validator

        assert hasattr(simple_validator, "SimpleValidator")

    @pytest.mark.unit
    def test_simple_validator_instantiation(self) -> Any:
        """Test SimpleValidator instantiation."""
        from gnn.simple_validator import SimpleValidator

        validator = SimpleValidator()

        # Verify validator has expected methods
        assert hasattr(validator, "validate_file")
        assert hasattr(validator, "validate_directory")

    @pytest.mark.unit
    def test_simple_validation(self, sample_gnn_files: Any) -> Any:
        """Test simple validation functionality."""
        from gnn.simple_validator import SimpleValidator

        validator = SimpleValidator()

        # Get sample file content
        sample_file = list(sample_gnn_files.values())[0]

        result = validator.validate_file(sample_file)

        # Verify validation result
        assert isinstance(result, (dict, bool))


BRIDGE_DOC_TEMPLATE = """## GNNSection
FepLeanProbe continuous

## GNNVersionAndFlags
GNN v1

## ModelName
Bridge Provenance Probe

## ModelAnnotation
Bridge provenance strict-mode probe document exercising the FepLean
Signature provenance-key requirements end to end.

## StateSpaceBlock
x[1,1,type=float]
y[1,1,type=float]

## Connections
x-y

## InitialParameterization
x={{(1.0)}}
y={{(1.0)}}

## Time
Time=t
Dynamic
Continuous
ModelTimeHorizon=1

## Footer
Probe footer

## Signature
{signature_block}"""


FULL_SIGNATURE = (
    "source_repository: fep_lean\n"
    "source_commit: 0123456789abcdef\n"
    "lean_module: lean/FepSketches/probe.lean\n"
    "projection_tool: probe_emitter.py\n"
    "target_syntax: GNN v1"
)


class TestBridgeProvenanceStrict:
    """Bridge-contract provenance strictness (bridge contract section 4)."""

    @pytest.fixture
    def validator(self) -> Any:
        from gnn.schema_validator import GNNValidator

        return GNNValidator()

    def _provenance_errors(self, result: Any) -> list:
        return [e for e in result.errors if "provenance key" in e]

    @pytest.mark.unit
    def test_bridge_doc_missing_keys_fails_strict(
        self, validator: Any, tmp_path: Any
    ) -> None:
        """A FepLean document missing provenance keys fails strict
        validation, one error per missing key."""
        from gnn.schema_validator import ValidationLevel

        doc_file = tmp_path / "missing.md"
        doc_file.write_text(
            BRIDGE_DOC_TEMPLATE.format(signature_block="source_repository: fep_lean")
        )
        result = validator.validate_file(
            doc_file, validation_level=ValidationLevel.STRICT
        )
        errors = self._provenance_errors(result)
        assert len(errors) == 4
        for key in (
            "source_commit",
            "lean_module",
            "projection_tool",
            "target_syntax",
        ):
            assert any(f"'{key}'" in e for e in errors), key
        assert not result.is_valid

    @pytest.mark.unit
    def test_bridge_doc_complete_keys_pass_strict(
        self, validator: Any, tmp_path: Any
    ) -> None:
        """A FepLean document with all five keys passes strict
        validation."""
        from gnn.schema_validator import ValidationLevel

        doc_file = tmp_path / "complete.md"
        doc_file.write_text(BRIDGE_DOC_TEMPLATE.format(signature_block=FULL_SIGNATURE))
        result = validator.validate_file(
            doc_file, validation_level=ValidationLevel.STRICT
        )
        assert self._provenance_errors(result) == []

    @pytest.mark.unit
    def test_non_bridge_doc_unaffected(self, validator: Any, tmp_path: Any) -> None:
        """A non-bridge document without provenance keys is not
        flagged."""
        from gnn.schema_validator import ValidationLevel

        doc_file = tmp_path / "plain.md"
        doc_file.write_text(
            BRIDGE_DOC_TEMPLATE.replace(
                "FepLeanProbe continuous", "PlainProbe continuous"
            ).format(signature_block="note: no provenance here")
        )
        result = validator.validate_file(
            doc_file, validation_level=ValidationLevel.STRICT
        )
        assert self._provenance_errors(result) == []

    @pytest.mark.unit
    def test_standard_level_not_enforced(self, validator: Any, tmp_path: Any) -> None:
        """The provenance requirement is strict-mode only."""
        from gnn.schema_validator import ValidationLevel

        doc_file = tmp_path / "standard.md"
        doc_file.write_text(
            BRIDGE_DOC_TEMPLATE.format(signature_block="source_repository: fep_lean")
        )
        result = validator.validate_file(
            doc_file, validation_level=ValidationLevel.STANDARD
        )
        assert self._provenance_errors(result) == []
