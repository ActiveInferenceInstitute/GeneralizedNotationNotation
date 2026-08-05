"""Tests for the TypedDict contracts and ModelKind enum in pomdp_contract.py.

Validates the typed contracts added for POMDP render specification:
- ModelKind enum detection from GNN specs
- CanonicalPomdpSpec TypedDict structure
- InitialParameterization TypedDict structure
- RxInferSimulationV1 TypedDict structure
- detect_model_kind() function behavior
"""

from __future__ import annotations

from pathlib import Path

import pytest

from render.pomdp_contract import (
    CANONICAL_B_ORDER,
    CanonicalPomdpSpec,
    InitialParameterization,
    ModelKind,
    RxInferSimulationV1,
    build_canonical_pomdp_spec,
    detect_model_kind,
)


def test_model_kind_enum_values() -> None:
    """ModelKind enum has all expected members."""
    assert ModelKind.FLAT.value == "flat"
    assert ModelKind.FACTORED.value == "factored"
    assert ModelKind.HIERARCHICAL.value == "hierarchical"
    assert ModelKind.MULTI_AGENT.value == "multi_agent"
    assert ModelKind.CONTINUOUS.value == "continuous"
    assert ModelKind.LEARNING.value == "learning"


def test_model_kind_enum_count() -> None:
    """ModelKind has exactly 6 members."""
    assert len(list(ModelKind)) == 6


def test_detect_model_kind_flat_default() -> None:
    """A simple GNN spec with no special markers returns FLAT."""
    spec = {
        "model_name": "Test",
        "model_parameters": {"num_hidden_states": 4, "num_obs": 4, "num_actions": 4},
        "initialparameterization": {
            "A": [[1, 0], [0, 1]],
            "B": [[1, 0], [0, 1]],
            "C": [0, 1],
            "D": [0.5, 0.5],
        },
    }
    assert detect_model_kind(spec) == ModelKind.FLAT


def test_detect_model_kind_multi_agent() -> None:
    """GNN spec with agent keys returns MULTI_AGENT."""
    spec = {
        "initialparameterization": {"nr_agents": 2, "agent1_id": "a1"},
    }
    assert detect_model_kind(spec) == ModelKind.MULTI_AGENT


def test_detect_model_kind_hierarchical() -> None:
    """GNN spec with hierarchical section returns HIERARCHICAL."""
    spec = {
        "gnn_section": "hierarchical",
        "initialparameterization": {},
    }
    assert detect_model_kind(spec) == ModelKind.HIERARCHICAL


def test_detect_model_kind_continuous() -> None:
    """GNN spec with continuous section returns CONTINUOUS."""
    spec = {
        "gnn_section": "continuous",
        "initialparameterization": {},
    }
    assert detect_model_kind(spec) == ModelKind.CONTINUOUS


def test_detect_model_kind_learning() -> None:
    """GNN spec with Dirichlet priors returns LEARNING."""
    spec = {
        "initialparameterization": {"dirichlet_A": [1.0, 1.0]},
    }
    assert detect_model_kind(spec) == ModelKind.LEARNING


def test_canonical_b_order_constant() -> None:
    """CANONICAL_B_ORDER is the expected string."""
    assert CANONICAL_B_ORDER == "next_state_previous_state_action"


def test_initialparameterization_typeddict_is_dict() -> None:
    """InitialParameterization TypedDict instances behave as dicts."""
    ip: InitialParameterization = {
        "A": [[1.0, 0.0], [0.0, 1.0]],
        "B": [[[1.0, 0.0], [0.0, 1.0]]],
        "C": [0.0, 1.0],
        "D": [0.5, 0.5],
    }
    assert isinstance(ip, dict)
    assert "A" in ip
    assert ip["A"] == [[1.0, 0.0], [0.0, 1.0]]


def test_canonical_pomdp_spec_typeddict_is_dict() -> None:
    """CanonicalPomdpSpec TypedDict instances behave as dicts."""
    spec: CanonicalPomdpSpec = {
        "model_name": "Test",
        "canonical_pomdp_schema": "canonical_pomdp_v1",
    }
    assert isinstance(spec, dict)
    assert spec["model_name"] == "Test"


def test_rxinfer_simulation_v1_typeddict_is_dict() -> None:
    """RxInferSimulationV1 TypedDict instances behave as dicts."""
    result: RxInferSimulationV1 = {
        "schema_version": "rxinfer_simulation_v1",
        "success": True,
        "variational_free_energy": [6.11, 6.11, 6.11],
        "vfe_per_iteration": [6.11, 6.11, 6.11],
    }
    assert isinstance(result, dict)
    assert result["schema_version"] == "rxinfer_simulation_v1"
    assert len(result["variational_free_energy"]) == 3


def test_build_canonical_pomdp_spec_returns_dict() -> None:
    """build_canonical_pomdp_spec returns a dict with canonical_pomdp_schema."""
    gnn_spec = {
        "model_name": "Test",
        "model_parameters": {
            "num_hidden_states": 2,
            "num_obs": 2,
            "num_actions": 2,
            "b_tensor_order": "next_state_previous_state_action",
        },
        "initialparameterization": {
            "A": [[1, 0], [0, 1]],
            "B": [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]],
            "C": [0, 1],
            "D": [0.5, 0.5],
        },
    }
    result = build_canonical_pomdp_spec(gnn_spec)
    assert isinstance(result, dict)
    assert result["canonical_pomdp_schema"] == "canonical_pomdp_v1"
    assert "initialparameterization" in result
    assert "A" in result["initialparameterization"]
    assert "B" in result["initialparameterization"]
    assert result["model_parameters"]["b_tensor_order"] == CANONICAL_B_ORDER
