"""Contract tests for render module shared helpers and processor policy.

Pins real behavior of the pieces extracted/refactored in the 2026-09-04
render composability pass:
  - ``render.naming`` (shared output-stem sanitization + atomic writes)
  - ``render.spec_matrices`` (shared discrete A/B/C/D extraction + literals)
  - ``render.framework_registry`` lite preset
  - ``render.processor.parse_frameworks_selection`` (CLI --frameworks policy)
  - ``render.processor._render_succeeded`` (success-policy contract)
  - ``render.validate_render`` facade contract
  - ``render.mcp.render_spec_to_format_mcp`` (single-framework MCP tool)
  - POMDPRenderProcessor unknown-framework dispatch contract

All tests are deterministic, isolated (tmp_path), and network-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from render import validate_render
from render.framework_registry import LITE_FRAMEWORKS, get_lite_frameworks
from render.naming import MAX_STEM_LENGTH, atomic_write_text, safe_output_stem
from render.processor import (
    _render_succeeded,
    parse_frameworks_selection,
)
from render.spec_matrices import (
    extract_abcd_matrices,
    format_array_literal,
    parse_gnn_matrix_value,
)

SAMPLE_GNN = (
    Path(__file__).parent.parent.parent.parent
    / "input"
    / "gnn_files"
    / "discrete"
    / "actinf_pomdp_agent.md"
)


class TestSafeOutputStem:
    """Output-stem sanitization contract (shared by both processors)."""

    def test_replaces_unsafe_characters(self) -> None:
        assert safe_output_stem("a b/c") == "a_b_c"

    def test_strips_leading_and_trailing_separators(self) -> None:
        assert safe_output_stem("..model__") == "model"

    def test_empty_falls_back(self) -> None:
        assert safe_output_stem("") == "model"
        assert safe_output_stem("///") == "model"
        assert safe_output_stem("", fallback="pomdp_model") == "pomdp_model"

    def test_truncates_to_120_chars(self) -> None:
        assert len(safe_output_stem("x" * 500)) == MAX_STEM_LENGTH

    def test_keeps_safe_characters(self) -> None:
        assert safe_output_stem("Model-1.2_3") == "Model-1.2_3"


class TestAtomicWriteText:
    def test_writes_content_and_creates_nested_parents(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "out.py"
        result = atomic_write_text(target, "print('hi')\n")
        assert result == target
        assert target.read_text() == "print('hi')\n"

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        atomic_write_text(target, "first")
        atomic_write_text(target, "second")
        assert target.read_text() == "second"

    def test_leaves_no_temp_files_behind(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        atomic_write_text(target, "x")
        leftovers = [p for p in tmp_path.iterdir() if p != target]
        assert leftovers == []


class TestExtractAbcdMatrices:
    def test_empty_spec_gets_neutral_defaults(self) -> None:
        a, b, c, d = extract_abcd_matrices({})
        assert a.shape == (2, 2)
        assert np.allclose(np.diag(a), 1.0)
        assert b.shape == (2, 2)
        assert c.tolist() == [1.0, 0.0]
        assert np.allclose(d, [0.5, 0.5])

    def test_state_space_parameters_take_precedence(self) -> None:
        spec = {
            "stateSpace": {"size": 3, "parameters": {"A": [[1.0, 2.0, 3.0]] * 3}},
            "initialparameterization": {"A": [[9.0]]},
        }
        a, _, _, _ = extract_abcd_matrices(spec)
        assert a.shape == (3, 3)
        assert np.allclose(a, 1.0 / 3.0)  # columns normalized from [3, 6, 9] sums

    def test_initialparameterization_used_when_state_space_empty(self) -> None:
        spec = {"initialparameterization": {"A": [[1.0, 1.0], [1.0, 1.0]]}}
        a, _, _, _ = extract_abcd_matrices(spec)
        assert np.allclose(a, 0.5)

    def test_d_is_normalized_to_probability_vector(self) -> None:
        _, _, _, d = extract_abcd_matrices({"parameters": {"D": [2.0, 2.0, 4.0]}})
        assert np.allclose(d, [0.25, 0.25, 0.5])

    def test_string_matrix_parsed_via_safe_literal(self) -> None:
        spec = {"parameters": {"A": "[[0.6, 0.25], [0.4, 0.75]]"}}
        a, _, _, _ = extract_abcd_matrices(spec)
        assert np.allclose(a, [[0.6, 0.25], [0.4, 0.75]])

    def test_parse_gnn_matrix_value_invalid_string_returns_default(self) -> None:
        sentinel = np.eye(2)
        assert parse_gnn_matrix_value("not a literal", sentinel) is sentinel
        assert parse_gnn_matrix_value(None, sentinel) is sentinel


class TestFormatArrayLiteral:
    def test_one_dimensional(self) -> None:
        out = format_array_literal(np.array([0.5, 0.25]), prefix="jnp.array")
        assert out == "jnp.array([0.500000, 0.250000])"

    def test_two_dimensional_with_suffix_and_indent(self) -> None:
        out = format_array_literal(
            np.array([[1.0, 2.0]]),
            prefix="torch.tensor",
            suffix=", dtype=torch.float64",
            indent=2,
        )
        assert out == (
            "torch.tensor([\n      [1.000000, 2.000000]\n  ], dtype=torch.float64)"
        )
        assert out.startswith("torch.tensor([")
        assert out.endswith("dtype=torch.float64)")

    def test_three_dimensional_falls_back_to_repr(self) -> None:
        arr = np.ones((1, 1, 1))
        out = format_array_literal(arr, prefix="jnp.array")
        assert out == f"jnp.array({arr.tolist()})"


class TestLiteFrameworkPreset:
    def test_preset_matches_registry_constant(self) -> None:
        assert get_lite_frameworks() == list(LITE_FRAMEWORKS)

    def test_preset_contents_are_registered_frameworks(self) -> None:
        from render.framework_registry import get_supported_frameworks

        supported = get_supported_frameworks()
        for name in LITE_FRAMEWORKS:
            assert name in supported

    def test_preset_excludes_julia_backends(self) -> None:
        assert "rxinfer" not in LITE_FRAMEWORKS
        assert "activeinference_jl" not in LITE_FRAMEWORKS
        assert "stan" not in LITE_FRAMEWORKS


class TestParseFrameworksSelection:
    def test_none_means_all_frameworks(self) -> None:
        frameworks, explicit = parse_frameworks_selection(None)
        assert frameworks is None
        assert explicit is False

    def test_all_keyword_normalizes_to_none(self) -> None:
        assert parse_frameworks_selection("all") == (None, False)
        assert parse_frameworks_selection("  ALL ") == (None, False)

    def test_lite_resolves_to_registry_preset(self) -> None:
        frameworks, explicit = parse_frameworks_selection("lite")
        assert frameworks == get_lite_frameworks()
        assert explicit is False

    def test_comma_separated_string_is_explicit(self) -> None:
        frameworks, explicit = parse_frameworks_selection("pymdp, jax")
        assert frameworks == ["pymdp", "jax"]
        assert explicit is True

    def test_list_selection_is_explicit(self) -> None:
        frameworks, explicit = parse_frameworks_selection(["rxinfer"])
        assert frameworks == ["rxinfer"]
        assert explicit is True


class TestRenderSucceededPolicy:
    def test_no_files_returns_exit_code_two(self) -> None:
        assert (
            _render_succeeded(
                success_count=0,
                total_files=0,
                total_framework_successes=0,
                total_framework_attempts=0,
            )
            == 2
        )

    def test_strict_mode_requires_every_framework(self) -> None:
        assert (
            _render_succeeded(
                success_count=1,
                total_files=1,
                total_framework_successes=1,
                total_framework_attempts=2,
                strict_framework_success=True,
            )
            is False
        )
        assert (
            _render_succeeded(
                success_count=1,
                total_files=1,
                total_framework_successes=1,
                total_framework_attempts=2,
                strict_framework_success=False,
            )
            is True
        )

    def test_aggregate_policy_requires_eighty_percent_or_any_file_success(
        self,
    ) -> None:
        assert (
            _render_succeeded(
                success_count=1,
                total_files=1,
                total_framework_successes=79,
                total_framework_attempts=100,
            )
            is True
        )
        assert (
            _render_succeeded(
                success_count=1,
                total_files=1,
                total_framework_successes=0,
                total_framework_attempts=100,
            )
            is True
        )  # success_count > 0 keeps partial-success semantics
        assert (
            _render_succeeded(
                success_count=0,
                total_files=1,
                total_framework_successes=79,
                total_framework_attempts=100,
            )
            is False
        )

    def test_no_framework_attempts_falls_back_to_file_counting(self) -> None:
        assert (
            _render_succeeded(
                success_count=2,
                total_files=2,
                total_framework_successes=0,
                total_framework_attempts=0,
            )
            is True
        )
        assert (
            _render_succeeded(
                success_count=1,
                total_files=2,
                total_framework_successes=0,
                total_framework_attempts=0,
            )
            is False
        )


class TestValidateRenderContract:
    def test_none_result_raises(self) -> None:
        with pytest.raises(ValueError, match="None"):
            validate_render(None)

    def test_empty_string_result_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_render("")

    def test_any_other_result_passes(self) -> None:
        assert validate_render("code") is True
        assert validate_render(["a.py"]) is True


class TestUnknownFrameworkDispatch:
    def test_unknown_framework_message_contract(self, tmp_path: Path) -> None:
        from render.pomdp_processor import POMDPRenderProcessor

        result = POMDPRenderProcessor(tmp_path)._call_framework_renderer(
            "no_such_backend", {}, tmp_path
        )
        assert result["success"] is False
        assert result["message"] == "No renderer implemented for no_such_backend"
        assert result["artifacts"] == []


class TestRenderSpecToFormatMcp:
    def test_renders_single_framework_end_to_end(self, tmp_path: Path) -> None:
        from render.mcp import render_spec_to_format_mcp

        out_dir = tmp_path / "out"
        result = render_spec_to_format_mcp(
            str(SAMPLE_GNN), str(out_dir), framework="bnlearn"
        )
        assert result["success"] is True, result.get("error") or result.get("message")
        assert result["framework"] == "bnlearn"
        assert len(result["output_files"]) == 1
        artifact = Path(result["output_files"][0])
        assert artifact.name.endswith("_bnlearn.py")
        assert artifact.exists()

    def test_missing_file_reports_error(self, tmp_path: Path) -> None:
        from render.mcp import render_spec_to_format_mcp

        result = render_spec_to_format_mcp(
            str(tmp_path / "missing.md"), str(tmp_path / "out")
        )
        assert result["success"] is False
        assert "missing.md" in result["error"]

    def test_unsupported_target_reports_failure_not_error(self, tmp_path: Path) -> None:
        from render.mcp import render_spec_to_format_mcp

        result = render_spec_to_format_mcp(
            str(SAMPLE_GNN), str(tmp_path / "out"), framework="definitely_not_real"
        )
        assert result["success"] is False
        assert "Unsupported target" in result["message"]
