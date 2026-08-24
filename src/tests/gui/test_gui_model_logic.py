"""Regression tests for headless GUI 2/3 model-building logic."""

from __future__ import annotations

import math
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _editor_tables(state: dict[str, object]) -> tuple[object, ...]:
    a_matrix = state["A"]
    b_matrix = state["B"]
    c_vector = state["C"]
    d_vector = state["D"]
    assert isinstance(a_matrix, dict)
    assert isinstance(b_matrix, dict)
    assert isinstance(c_vector, dict)
    assert isinstance(d_vector, dict)
    slice_index = b_matrix["current_slice"]
    assert isinstance(slice_index, int)
    b_values = b_matrix["values"]
    assert isinstance(b_values, list)
    return (
        a_matrix["values"],
        b_values[slice_index],
        [[value] for value in c_vector["values"]],
        [[value] for value in d_vector["values"]],
        slice_index,
    )


def test_gui2_loads_real_values_and_noncubic_transition_depth() -> None:
    from gui.gui_2.matrix_editor import create_matrix_from_gnn

    model = (
        REPOSITORY_ROOT / "input/gnn_files/continuous/stochastic_dynamics.md"
    ).read_text(encoding="utf-8")
    visual_data = create_matrix_from_gnn(model)
    matrices = visual_data["visual_matrices"]

    assert matrices["A"]["values"][0] == [0.9, 0.1, 0.5]
    assert matrices["B"]["rows"] == 3
    assert matrices["B"]["cols"] == 3
    assert matrices["B"]["depth"] == 2
    assert len(matrices["B"]["values"]) == 2
    assert matrices["C"]["values"] == [0.0, 0.0]
    assert matrices["D"]["values"] == [0.8, 0.1, 0.1]


def test_gui2_real_pomdp_round_trip_preserves_unedited_parameters() -> None:
    from gui.gui_2.matrix_editor import create_matrix_from_gnn
    from gui.gui_2.ui import _generate_editor_gnn, _initial_matrix_state

    model_path = REPOSITORY_ROOT / "input/gnn_files/discrete/actinf_pomdp_agent.md"
    model = model_path.read_text(encoding="utf-8")
    state = _initial_matrix_state(model)
    a_data, b_data, c_data, d_data, b_slice = _editor_tables(state)
    assert isinstance(a_data, list)
    assert isinstance(b_data, list)
    a_data[0][0] = 0.8
    b_data[0][0] = 0.75

    _, exported = _generate_editor_gnn(
        state, a_data, b_data, c_data, d_data, b_slice, model
    )

    assert "E={(0.33333, 0.33333, 0.33333)}" in exported
    assert "## Equations" in exported
    assert "A[3,3,type=float]" in exported
    assert "(0.8, 0.05, 0.05)" in exported
    reparsed = create_matrix_from_gnn(exported)["visual_matrices"]
    assert reparsed["A"]["values"][0][0] == 0.8
    assert reparsed["B"]["values"][0][0][0] == 0.75


def test_gui2_passive_hmm_does_not_gain_undeclared_preference_vector() -> None:
    from gui.gui_2.ui import _generate_editor_gnn, _initial_matrix_state

    model = (REPOSITORY_ROOT / "input/gnn_files/discrete/hmm_baseline.md").read_text(
        encoding="utf-8"
    )
    state = _initial_matrix_state(model)
    assert state["B"]["depth"] == 1
    assert state["B"]["source_type"] == "matrix"
    assert state["C"]["declared"] is False

    a_data, b_data, c_data, d_data, b_slice = _editor_tables(state)
    _, exported = _generate_editor_gnn(state, a_data, b_data, c_data, d_data, b_slice, model)

    state_space = exported.split("## StateSpaceBlock", 1)[1].split("## Connections", 1)[
        0
    ]
    parameters = exported.split("## InitialParameterization", 1)[1].split(
        "## Equations", 1
    )[0]
    assert "\nC[" not in state_space
    assert "\nC={" not in parameters
    assert "B[4,4,type=float]" in state_space


def test_gui2_callbacks_are_total_for_empty_stale_and_nonfinite_payloads() -> None:
    from gui.gui_2.ui import (
        _coerce_editor_state,
        _initial_matrix_state,
        _select_b_slice,
        _validate_editor_tables,
    )

    fallback = _initial_matrix_state("")
    assert fallback["A"]["rows"] == 3
    assert fallback["B"]["depth"] == 3

    status = _validate_editor_tables(None, [], [], [], [], math.nan)
    assert status.startswith("❌")
    assert "cannot be empty" in status

    state, selected = _select_b_slice({"stale": True}, [], math.inf)
    assert state["B"]["current_slice"] == 0
    assert selected

    malformed_state: dict[str, object] = {name: {} for name in ("A", "B", "C", "D")}
    coerced = _coerce_editor_state(malformed_state)
    assert coerced["A"]["rows"] == 3


def test_gui2_slice_switch_preserves_independent_transition_edits() -> None:
    from gui.gui_2.ui import _initial_matrix_state, _select_b_slice

    model = (
        REPOSITORY_ROOT / "input/gnn_files/discrete/actinf_pomdp_agent.md"
    ).read_text(encoding="utf-8")
    state = _initial_matrix_state(model)
    first_slice = state["B"]["values"][0]
    first_slice[0][0] = 0.25

    state, second_slice = _select_b_slice(state, first_slice, 1)
    second_slice[0][0] = 0.75
    state, restored_first = _select_b_slice(state, second_slice, 0)

    assert restored_first[0][0] == 0.25
    assert state["B"]["values"][1][0][0] == 0.75


def test_gui3_canonical_model_round_trips_inline_ontology_comments() -> None:
    from gui.gui_3.ui_designer import (
        _generate_gnn_from_design,
        _parse_gnn_for_design,
    )

    model = (
        REPOSITORY_ROOT / "input/gnn_files/discrete/actinf_pomdp_agent.md"
    ).read_text(encoding="utf-8")
    parsed = _parse_gnn_for_design(model)
    ontology = [[name, term, ""] for name, term in parsed["ontology"].items()]

    exported = _generate_gnn_from_design(
        parsed["state_spaces"],
        ontology,
        parsed["connections_text"],
        parsed["parameters"]["num_hidden_states"],
        parsed["parameters"]["num_obs"],
        parsed["parameters"]["num_actions"],
        1,
        "Unbounded",
    )

    assert parsed["ontology"]["π"] == "PolicyVector"
    assert "π=PolicyVector" in exported
    assert "# Distribution over actions" not in exported
    assert "π>u" in exported


def test_gui3_default_and_malformed_models_are_safe_to_build() -> None:
    from gui.gui_3.processor import _get_default_pomdp_template
    from gui.gui_3.ui_designer import (
        _bounded_int,
        _generate_gnn_from_design,
        _parse_gnn_for_design,
    )

    default = _parse_gnn_for_design(_get_default_pomdp_template())
    default_export = _generate_gnn_from_design(
        default["state_spaces"],
        [],
        default["connections_text"],
        3,
        3,
        3,
        1,
        "Unbounded",
    )
    assert "D>s" in default_export
    assert "u[1,type=float]" in default_export

    malformed = _parse_gnn_for_design(
        "## StateSpaceBlock\nA[,type=float]\ns[2,type=float]\n## Connections\ns>s\n"
    )
    assert malformed["state_spaces"] == [["s", "2", ""]]
    assert malformed["parse_errors"]
    assert _bounded_int(math.inf, 3, 1, 10) == 3
