#!/usr/bin/env python3
"""
Regression tests for strict validation-level resolution.

``GNNValidator._level_rank`` must resolve string validation levels to real
semantic validation instead of silently defaulting to rank 0:

- ``ValidationLevel`` members rank by identity (fast path).
- Strings equal to an enum value (``"strict"``) resolve to that member.
- Strings equal to an enum name (``"STANDARD"``, case-insensitive) resolve
  via ``ValidationLevel[name]`` and run semantic validation.
- Anything else raises ``ValueError`` listing the accepted forms, loudly at
  the public ``validate_file`` API (not swallowed by its broad handler).
"""

from pathlib import Path
from typing import Any

import pytest

from gnn.schema_validator.validator import GNNValidator
from gnn.types import ValidationLevel

EXEMPLAR = (
    Path(__file__).resolve().parents[2]
    / "input/gnn_files/discrete/time_varying_dynamics.md"
)

RANKS = [
    (ValidationLevel.BASIC, 10),
    (ValidationLevel.STANDARD, 20),
    (ValidationLevel.STRICT, 30),
    (ValidationLevel.RESEARCH, 40),
    (ValidationLevel.ROUND_TRIP, 50),
]


class TestLevelRankResolution:
    """_level_rank must accept enum members, value strings, and name strings."""

    def setup_method(self) -> None:
        self.validator = GNNValidator()

    @pytest.mark.parametrize("level,expected", RANKS)
    def test_enum_member_ranks_by_identity(
        self, level: ValidationLevel, expected: int
    ) -> None:
        assert self.validator._level_rank(level) == expected

    @pytest.mark.parametrize("level,expected", RANKS)
    def test_value_string_ranks_as_member(
        self, level: ValidationLevel, expected: int
    ) -> None:
        assert self.validator._level_rank(level.value) == expected

    @pytest.mark.parametrize("level,expected", RANKS)
    def test_name_string_ranks_as_member(
        self, level: ValidationLevel, expected: int
    ) -> None:
        assert self.validator._level_rank(level.name) == expected

    def test_mixed_case_name_resolves_via_upper(self) -> None:
        assert self.validator._level_rank("Standard") == 20

    def test_unknown_string_raises_listing_accepted_forms(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            self.validator._level_rank("nonsense")
        message = str(excinfo.value)
        assert "nonsense" in message
        # Every accepted form (name and value) must be listed.
        for member in ValidationLevel:
            assert member.name in message
            assert member.value in message


class TestNameStringRunsSemanticValidation:
    """A name-string level must run the real pipeline, not skip it."""

    def test_name_level_validates_valid_exemplar(self) -> Any:
        validator = GNNValidator(
            validation_level=ValidationLevel.RESEARCH, enable_cross_validation=False
        )

        result = validator.validate_file(EXEMPLAR, validation_level="RESEARCH")

        assert result.validation_level == ValidationLevel.RESEARCH
        assert result.is_valid, result.errors

    def test_name_level_catches_missing_required_sections(self, tmp_path: Path) -> None:
        # Under the old silent rank-0 behavior this document passed
        # unvalidated; at "BASIC" name level it must be rejected.
        target = tmp_path / "broken.md"
        target.write_text("This is not a GNN model.", encoding="utf-8")
        validator = GNNValidator()

        result = validator.validate_file(target, validation_level="BASIC")

        assert result.validation_level == ValidationLevel.BASIC
        assert result.is_valid is False
        assert any("Required section missing" in error for error in result.errors)


class TestPublicApiRaisesOnUnknownLevel:
    """validate_file must surface the raise, not downgrade it to an error."""

    def test_validate_file_unknown_string_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "model.md"
        target.write_text("Some content.", encoding="utf-8")
        validator = GNNValidator()

        with pytest.raises(ValueError, match="Unknown validation level"):
            validator.validate_file(target, validation_level="nonsense")
