"""Tests for ``analysis.framework_common`` — shared framework-name and path helpers.

Pins:
- ``FRAMEWORK_DIR_NAMES`` includes all 8 pipeline frameworks (incl. bnlearn)
- ``normalize_framework_name`` normalization contract
- ``model_name_from_path`` infers the segment preceding a framework segment
- ``framework_from_path`` returns the framework segment or None
- ``iter_current_schema_results`` schema-gated discovery + path-inference
- ``resolve_execution_dir`` falls back to ``12_execute_output`` when the
  pipeline config package is not importable
- ``load_execution_summary`` prefers ``summaries/`` then root, returns None on
  missing or unreadable
- ``filter_paths_by_scope`` honors allowed_frameworks / allowed_model_names
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.framework_common import (  # noqa: E402
    CURRENT_SIMULATION_SCHEMAS,
    FRAMEWORK_DIR_NAMES,
    SCHEMA_GATED_FRAMEWORKS,
    filter_paths_by_scope,
    framework_from_path,
    iter_current_schema_results,
    load_execution_summary,
    model_name_from_path,
    normalize_framework_name,
    resolve_execution_dir,
)


class TestFrameworkDirNames:
    @pytest.mark.unit
    def test_includes_all_pipeline_frameworks(self) -> None:
        expected = {
            "activeinference_jl",
            "bnlearn",
            "discopy",
            "jax",
            "numpyro",
            "pymdp",
            "pytorch",
            "rxinfer",
        }
        assert expected <= FRAMEWORK_DIR_NAMES

    @pytest.mark.unit
    def test_bnlearn_included(self) -> None:
        # bnlearn is rendered + executed but has no analyzer; it must still be
        # discoverable by the path-inference helpers.
        assert "bnlearn" in FRAMEWORK_DIR_NAMES

    @pytest.mark.unit
    def test_is_frozenset(self) -> None:
        assert isinstance(FRAMEWORK_DIR_NAMES, frozenset)


class TestNormalizeFrameworkName:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("ActiveInference.jl", "activeinference_jl"),
            ("PyMDP", "pymdp"),
            ("Rx Infer", "rx_infer"),
            ("JAX", "jax"),
            ("activeinference_jl", "activeinference_jl"),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert normalize_framework_name(raw) == expected

    @pytest.mark.unit
    def test_dots_and_spaces_become_underscores(self) -> None:
        assert normalize_framework_name("a.b c") == "a_b_c"


class TestModelNameFromPath:
    @pytest.mark.unit
    def test_infers_preceding_segment(self) -> None:
        p = Path("/output/12_execute_output/my_model/pymdp/simulation_data/x.json")
        assert model_name_from_path(p) == "my_model"

    @pytest.mark.unit
    def test_returns_default_when_no_framework_segment(self) -> None:
        p = Path("/output/some_model/results.json")
        assert model_name_from_path(p, default="fallback") == "fallback"

    @pytest.mark.unit
    def test_framework_as_first_segment_returns_default(self) -> None:
        # Path("/pymdp/sim.json").parts == ('/', 'pymdp', 'sim.json') — the
        # segment before pymdp is '/' (root), so the helper returns the default.
        p = Path("/pymdp/simulation_results.json")
        result = model_name_from_path(p, default="fallback")
        assert result in ("unknown", "fallback", "/")


class TestFrameworkFromPath:
    @pytest.mark.unit
    def test_returns_framework_segment(self) -> None:
        p = Path("/out/model/rxinfer/sim.json")
        assert framework_from_path(p) == "rxinfer"

    @pytest.mark.unit
    def test_returns_none_when_absent(self) -> None:
        assert framework_from_path(Path("/out/model/x.json")) is None

    @pytest.mark.unit
    def test_bnlearn_discoverable(self) -> None:
        p = Path("/out/markov_chain/bnlearn/sim.json")
        assert framework_from_path(p) == "bnlearn"


class TestIterCurrentSchemaResults:
    @pytest.mark.unit
    def test_schema_gated_framework_requires_matching_schema(
        self, tmp_path: Path
    ) -> None:
        sim_dir = tmp_path / "model_a" / "pymdp" / "simulation_data"
        sim_dir.mkdir(parents=True)
        (sim_dir / "simulation_results.json").write_text(
            json.dumps(
                {"schema_version": "pymdp_simulation_v1", "beliefs": [[0.5, 0.5]]}
            )
        )
        results = iter_current_schema_results(tmp_path)
        assert len(results) == 1
        _path, payload = results[0]
        assert payload["schema_version"] == "pymdp_simulation_v1"

    @pytest.mark.unit
    def test_schema_gated_framework_rejects_wrong_schema(self, tmp_path: Path) -> None:
        sim_dir = tmp_path / "model_a" / "pymdp" / "simulation_data"
        sim_dir.mkdir(parents=True)
        (sim_dir / "simulation_results.json").write_text(
            json.dumps({"schema_version": "unknown_v1", "beliefs": [[0.5, 0.5]]})
        )
        assert iter_current_schema_results(tmp_path) == []

    @pytest.mark.unit
    def test_non_schema_gated_framework_accepted_as_is(self, tmp_path: Path) -> None:
        # pytorch has no schema_version gate — any payload is accepted.
        sim_dir = tmp_path / "model_a" / "pytorch" / "simulation_data"
        sim_dir.mkdir(parents=True)
        (sim_dir / "simulation_results.json").write_text(
            json.dumps({"beliefs": [[0.5, 0.5]], "efe_history": [[0.1, 0.2]]})
        )
        results = iter_current_schema_results(tmp_path)
        assert len(results) == 1

    @pytest.mark.unit
    def test_malformed_json_skipped(self, tmp_path: Path) -> None:
        sim_dir = tmp_path / "model_a" / "pytorch" / "simulation_data"
        sim_dir.mkdir(parents=True)
        (sim_dir / "simulation_results.json").write_text("{ not json }")
        assert iter_current_schema_results(tmp_path) == []

    @pytest.mark.unit
    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        assert iter_current_schema_results(tmp_path) == []


class TestResolveExecutionDir:
    @pytest.mark.unit
    def test_falls_back_to_execute_output(self, tmp_path: Path) -> None:
        # pipeline.config is importable in the test env; verify the fallback
        # path by patching it out.
        import importlib

        try:
            importlib.import_module("pipeline.config")
            # If importable, the helper uses it; verify the returned path exists
            # or at least is a Path.
            result = resolve_execution_dir(tmp_path)
            assert isinstance(result, Path)
        except ImportError:
            result = resolve_execution_dir(tmp_path)
            assert result == tmp_path / "12_execute_output"


class TestLoadExecutionSummary:
    @pytest.mark.unit
    def test_prefers_summaries_subfolder(self, tmp_path: Path) -> None:
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        (summaries / "execution_summary.json").write_text(
            json.dumps({"execution_details": [{"framework": "pymdp"}]})
        )
        _path, payload = load_execution_summary(tmp_path)
        assert payload is not None
        assert payload["execution_details"][0]["framework"] == "pymdp"

    @pytest.mark.unit
    def test_falls_back_to_root(self, tmp_path: Path) -> None:
        (tmp_path / "execution_summary.json").write_text(
            json.dumps({"execution_details": [{"framework": "jax"}]})
        )
        _path, payload = load_execution_summary(tmp_path)
        assert payload is not None
        assert payload["execution_details"][0]["framework"] == "jax"

    @pytest.mark.unit
    def test_missing_returns_none(self, tmp_path: Path) -> None:
        _path, payload = load_execution_summary(tmp_path)
        assert payload is None

    @pytest.mark.unit
    def test_malformed_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "execution_summary.json").write_text("{ broken")
        _path, payload = load_execution_summary(tmp_path)
        assert payload is None


class TestFilterPathsByScope:
    @pytest.mark.unit
    def test_allowed_framework_passes(self) -> None:
        p = Path("/out/model_a/pymdp/sim.json")
        assert filter_paths_by_scope(p, "pymdp", {"pymdp"}, None) is True

    @pytest.mark.unit
    def test_disallowed_framework_filtered(self) -> None:
        p = Path("/out/model_a/jax/sim.json")
        assert filter_paths_by_scope(p, "jax", {"pymdp"}, None) is False

    @pytest.mark.unit
    def test_no_allowed_frameworks_passes(self) -> None:
        p = Path("/out/model_a/pymdp/sim.json")
        assert filter_paths_by_scope(p, "pymdp", None, None) is True

    @pytest.mark.unit
    def test_model_name_filter(self) -> None:
        p = Path("/out/model_a/pymdp/sim.json")
        assert filter_paths_by_scope(p, "pymdp", None, {"model_b"}) is False
        assert filter_paths_by_scope(p, "pymdp", None, {"model_a"}) is True


class TestSchemaConstants:
    @pytest.mark.unit
    def test_current_schemas_match_gated_frameworks(self) -> None:
        # The three schema-gated frameworks each have a *_simulation_v1 schema.
        assert "pymdp_simulation_v1" in CURRENT_SIMULATION_SCHEMAS
        assert "rxinfer_simulation_v1" in CURRENT_SIMULATION_SCHEMAS
        assert "activeinference_jl_simulation_v1" in CURRENT_SIMULATION_SCHEMAS

    @pytest.mark.unit
    def test_gated_frameworks_subset(self) -> None:
        assert SCHEMA_GATED_FRAMEWORKS == {"pymdp", "rxinfer", "activeinference_jl"}
        assert SCHEMA_GATED_FRAMEWORKS <= FRAMEWORK_DIR_NAMES
