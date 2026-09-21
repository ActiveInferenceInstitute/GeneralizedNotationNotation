"""Tests for ``analysis.viz_schema`` — schema ids, flat fallbacks, attribution."""

from __future__ import annotations

from pathlib import Path

import pytest

from gnn.analysis.viz_schema import (
    CURRENT_VISUALIZATION_SCHEMAS,
    VISUALIZATION_FRAMEWORK_DIRS,
    _current_schema_visualization_data,
    _framework_from_path_or_payload,
)


class TestCurrentSchemaVisualizationData:
    @pytest.mark.unit
    def test_flat_pytorch_schema_payload(self) -> None:
        """A flat pytorch schema payload feeds beliefs/actions/observations."""
        data = _current_schema_visualization_data(
            {
                "schema_version": "pytorch_simulation_v1",
                "beliefs": [[0.7, 0.3]],
                "actions": [1],
                "observations": [0],
            }
        )
        assert data["schema_version"] == "pytorch_simulation_v1"
        assert data["beliefs"] == [[0.7, 0.3]]
        assert data["actions"] == [1]
        assert data["observations"] == [0]

    @pytest.mark.unit
    def test_flat_numpyro_schema_payload(self) -> None:
        """A flat numpyro schema payload feeds the same fields."""
        data = _current_schema_visualization_data(
            {
                "schema_version": "numpyro_simulation_v1",
                "beliefs": [[0.2, 0.8]],
                "actions": [0],
                "observations": [1],
            }
        )
        assert data["schema_version"] == "numpyro_simulation_v1"
        assert data["beliefs"] == [[0.2, 0.8]]
        assert data["actions"] == [0]
        assert data["observations"] == [1]

    @pytest.mark.unit
    def test_by_factor_maps_take_precedence(self) -> None:
        """The by-factor maps remain the primary source for pymdp payloads."""
        data = _current_schema_visualization_data(
            {
                "schema_version": "pymdp_simulation_v1",
                "beliefs_by_factor": {"joint_state": [0.9, 0.1]},
                "actions_by_control_factor": {"joint_action": [0]},
                "observations_by_modality": {"joint_observation": [1, 0]},
            }
        )
        assert data["beliefs"] == [0.9, 0.1]
        assert data["actions"] == [0]
        assert data["observations"] == [1, 0]

    @pytest.mark.unit
    def test_unregistered_schema_returns_empty(self) -> None:
        data = _current_schema_visualization_data({"schema_version": "other_v9"})
        assert data == {}


class TestVisualizationSchemaRegistry:
    @pytest.mark.unit
    def test_new_backend_schema_ids_registered(self) -> None:
        assert "pytorch_simulation_v1" in CURRENT_VISUALIZATION_SCHEMAS
        assert "numpyro_simulation_v1" in CURRENT_VISUALIZATION_SCHEMAS

    @pytest.mark.unit
    def test_stan_in_visualization_dirs(self) -> None:
        assert "stan" in VISUALIZATION_FRAMEWORK_DIRS


class TestStanAttribution:
    @pytest.mark.unit
    def test_stan_path_attributed_not_unknown(self) -> None:
        fw = _framework_from_path_or_payload(
            Path("/out/model_a/stan/simulation_results.json"),
            {},
        )
        assert fw == "stan"
