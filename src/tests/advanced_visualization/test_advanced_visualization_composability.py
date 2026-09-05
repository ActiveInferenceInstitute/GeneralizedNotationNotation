"""Tests for the composability helpers added in the refactor.

Covers:
- ``_shared.record_attempt`` aggregate bookkeeping (success/failed/skipped,
  optional-message filtering for D2-CLI absence).
- ``_shared._conn_endpoints`` connection normalization (scalar + new formats).
- ``_shared.VAR_TYPE_COLORS`` / ``VAR_TYPE_UNKNOWN_COLOR`` palette constants.
- ``_shared`` layout constants (deterministic seed + span).
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestRecordAttempt:
    def _make(self) -> Any:
        from advanced_visualization._shared import (
            AdvancedVisualizationAttempt,
            AdvancedVisualizationResults,
            record_attempt,
        )

        return (
            record_attempt,
            AdvancedVisualizationAttempt,
            AdvancedVisualizationResults,
        )

    def test_success_counts_and_extends_output_files(self) -> None:
        record, Attempt, Results = self._make()
        results = Results()
        record(results, Attempt("3d", "m", "success", output_files=["a.png", "b.png"]))
        assert results.total_attempts == 1
        assert results.successful == 1
        assert results.failed == 0
        assert results.skipped == 0
        assert results.output_files == ["a.png", "b.png"]
        assert results.errors == []
        assert results.warnings == []

    def test_failed_records_error_message(self) -> None:
        record, Attempt, Results = self._make()
        results = Results()
        record(results, Attempt("statistical", "m", "failed", error_message="boom"))
        assert results.failed == 1
        assert results.errors == ["boom"]
        assert results.output_files == []

    def test_failed_with_no_message_does_not_record_error(self) -> None:
        record, Attempt, Results = self._make()
        results = Results()
        record(results, Attempt("statistical", "m", "failed"))
        assert results.failed == 1
        assert results.errors == []

    def test_skipped_without_filter_records_warning(self) -> None:
        record, Attempt, Results = self._make()
        results = Results()
        record(results, Attempt("3d", "m", "skipped", error_message="no data"))
        assert results.skipped == 1
        assert results.warnings == ["no data"]

    def test_skipped_d2_cli_message_filtered_from_warnings(self) -> None:
        record, Attempt, Results = self._make()
        results = Results()
        record(
            results,
            Attempt("d2", "m", "skipped", error_message="D2 CLI not installed"),
            optional_message_filter="D2 CLI",
        )
        assert results.skipped == 1
        assert results.warnings == []  # filtered because D2 CLI is optional

    def test_skipped_non_d2_message_passes_through_filter(self) -> None:
        record, Attempt, Results = self._make()
        results = Results()
        record(
            results,
            Attempt("d2", "m", "skipped", error_message="real failure"),
            optional_message_filter="D2 CLI",
        )
        assert results.skipped == 1
        assert results.warnings == ["real failure"]

    def test_aggregate_across_multiple_attempts(self) -> None:
        record, Attempt, Results = self._make()
        results = Results()
        record(results, Attempt("3d", "m", "success", output_files=["x.png"]))
        record(
            results,
            Attempt("d2", "m", "skipped", error_message="D2 CLI absent"),
            optional_message_filter="D2 CLI",
        )
        record(results, Attempt("stat", "m", "failed", error_message="err"))
        record(results, Attempt("pomdp", "m", "success", output_files=["y.png"]))
        assert results.total_attempts == 4
        assert results.successful == 2
        assert results.failed == 1
        assert results.skipped == 1
        assert sorted(results.output_files) == ["x.png", "y.png"]
        assert results.errors == ["err"]
        assert results.warnings == []


class TestConnEndpoints:
    def _fn(self) -> Any:
        from advanced_visualization._shared import _conn_endpoints

        return _conn_endpoints

    def test_new_format_returns_source_target_directly(self) -> None:
        fn = self._fn()
        s, t = fn({"source_variables": ["A"], "target_variables": ["B"]})
        assert s == ["A"]
        assert t == ["B"]

    def test_legacy_format_normalized_to_lists(self) -> None:
        fn = self._fn()
        s, t = fn({"source": "X", "target": "Y", "weight": 0.5})
        assert s == ["X"]
        assert t == ["Y"]

    def test_empty_when_no_keys(self) -> None:
        fn = self._fn()
        s, t = fn({"foo": "bar"})
        assert s == []
        assert t == []

    def test_extra_keys_preserved_by_normalize(self) -> None:
        fn = self._fn()
        s, t = fn({"source": "A", "target": "B", "label": "prob"})
        assert s == ["A"]
        assert t == ["B"]


class TestSharedConstants:
    def test_var_type_colors_has_known_palette(self) -> None:
        from advanced_visualization._shared import (
            VAR_TYPE_COLORS,
            VAR_TYPE_UNKNOWN_COLOR,
        )

        assert VAR_TYPE_COLORS["hidden_state"] == "#FECA57"
        assert VAR_TYPE_COLORS["transition_matrix"] == "#4ECDC4"
        assert VAR_TYPE_UNKNOWN_COLOR == "#CCCCCC"
        # Palette covers the var_type values the network viz actually emits.
        for key in (
            "likelihood_matrix",
            "transition_matrix",
            "preference_vector",
            "prior_vector",
            "hidden_state",
            "observation",
            "policy",
            "action",
        ):
            assert key in VAR_TYPE_COLORS

    def test_layout_constants_deterministic(self) -> None:
        from advanced_visualization._shared import (
            FORCE_LAYOUT_SEED,
            LAYOUT_ITERATIONS,
            LAYOUT_SEED,
            LAYOUT_SPAN,
            LAYOUT_STEP,
        )

        assert LAYOUT_SEED == FORCE_LAYOUT_SEED == 42
        assert LAYOUT_SPAN == 10.0
        assert LAYOUT_ITERATIONS == 50
        assert LAYOUT_STEP == 0.01
