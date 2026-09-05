"""Strict GNN/GEO artifact producer and opt-in registry conformance."""

from pathlib import Path
import hashlib
import json

import pytest
from export.geo_infer import build_geo_infer_artifact, export_to_geo_infer
from export.processor import export_model

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md"


def test_gridworld_values_and_source_digest(tmp_path):
    text = SOURCE.read_text()
    artifact = build_geo_infer_artifact(text, step_seconds=60)
    assert artifact["dimensions"] == dict(states=9, observations=9, actions=5)
    assert artifact["matrices"]["E"] == [0.2] * 5
    assert (
        artifact["provenance"]["source_sha256"]
        == hashlib.sha256(text.encode()).hexdigest()
    )
    assert artifact["space"]["state_ids"] == list(map(str, range(9)))
    assert artifact["time"] == dict(step_seconds=60)
    result = export_model(
        dict(raw_content=text, geo_infer=dict(step_seconds=60)),
        tmp_path,
        formats=["geo_infer"],
    )
    assert result["success"], result
    assert json.loads((tmp_path / "model.geo-infer.json").read_text()) == artifact


def test_time_is_mandatory_and_failure_leaves_no_artifact(tmp_path):
    path = tmp_path / "model.json"
    with pytest.raises(ValueError, match="explicit"):
        export_to_geo_infer(dict(raw_content=SOURCE.read_text()), path)
    assert not path.exists()


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), True])
def test_invalid_time(seconds):
    with pytest.raises(ValueError):
        build_geo_infer_artifact(SOURCE.read_text(), step_seconds=seconds)


def test_missing_matrices_are_not_fabricated():
    text = SOURCE.read_text().replace("E={(0.2, 0.2, 0.2, 0.2, 0.2)}", "")
    with pytest.raises(ValueError):
        build_geo_infer_artifact(text, step_seconds=1)


def test_continuous_model_requires_a_different_contract():
    text = (ROOT / "input/gnn_files/continuous/continuous_navigation.md").read_text()
    with pytest.raises(ValueError):
        build_geo_infer_artifact(text, step_seconds=1)


def test_state_order_is_not_sorted_or_invented():
    labels = [f"cell-{i}" for i in reversed(range(9))]
    artifact = build_geo_infer_artifact(
        SOURCE.read_text(), step_seconds=1, state_ids=labels
    )
    assert artifact["space"]["state_ids"] == labels
    with pytest.raises(ValueError):
        build_geo_infer_artifact(
            SOURCE.read_text(), step_seconds=1, state_ids=["same"] * 9
        )
    with pytest.raises(ValueError):
        build_geo_infer_artifact(SOURCE.read_text(), step_seconds=1, space_kind="h3")


def test_rectangular_action_axis_canonicalization_and_diagnostics():
    import numpy as np
    from gnn.pomdp_extractor import POMDPStateSpace, POMDPExtractor, canonicalize_pomdp

    stored = np.array(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]], [[0.5, 0.5], [0.5, 0.5]]]
    )
    model = POMDPStateSpace(
        num_states=2,
        num_observations=2,
        num_actions=3,
        B_matrix=stored.tolist(),
        matrix_provenance={
            "B": {"detected_order": ["action", "previous_state", "next_state"]}
        },
    )
    assert POMDPExtractor()._validate_pomdp_structure(model)["valid"]
    canonical = canonicalize_pomdp(model)
    np.testing.assert_array_equal(canonical.B_matrix, stored.transpose(2, 1, 0))
    np.testing.assert_array_equal(model.B_matrix, stored)
    np.testing.assert_array_equal(
        canonicalize_pomdp(canonical).B_matrix, canonical.B_matrix
    )
    assert POMDPExtractor()._validate_pomdp_structure(canonical)["valid"]
    canonical.B_matrix[0][0][0] = 99
    np.testing.assert_array_equal(model.B_matrix, stored)


def test_canonical_gridworld_has_no_spurious_dimension_warning(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        build_geo_infer_artifact(SOURCE.read_text(), step_seconds=60)
    assert "B matrix dimensions" not in caplog.text
