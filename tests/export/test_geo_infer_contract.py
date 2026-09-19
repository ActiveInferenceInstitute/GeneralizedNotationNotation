"""Strict GNN/GEO artifact producer and opt-in registry conformance."""

import hashlib
import json
from pathlib import Path

import pytest

from gnn.export.geo_infer import build_geo_infer_artifact, export_to_geo_infer
from gnn.export.processor import export_model

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md"


def test_gridworld_values_and_source_digest(tmp_path: Path) -> None:
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


def test_time_is_mandatory_and_failure_leaves_no_artifact(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    with pytest.raises(ValueError, match="explicit"):
        export_to_geo_infer(dict(raw_content=SOURCE.read_text()), path)
    assert not path.exists()


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), True])
def test_invalid_time(seconds: float) -> None:
    with pytest.raises(ValueError):
        build_geo_infer_artifact(SOURCE.read_text(), step_seconds=seconds)


def test_missing_matrices_are_not_fabricated() -> None:
    text = SOURCE.read_text().replace("E={(0.2, 0.2, 0.2, 0.2, 0.2)}", "")
    with pytest.raises(ValueError):
        build_geo_infer_artifact(text, step_seconds=1)


def test_continuous_model_requires_a_different_contract() -> None:
    text = (ROOT / "input/gnn_files/continuous/continuous_navigation.md").read_text()
    with pytest.raises(ValueError):
        build_geo_infer_artifact(text, step_seconds=1)


def test_state_order_is_not_sorted_or_invented() -> None:
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


def test_rectangular_action_axis_canonicalization_and_diagnostics() -> None:
    import numpy as np

    from gnn.extract.pomdp_extractor import (
        POMDPExtractor,
        POMDPStateSpace,
        canonicalize_pomdp,
    )

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
    assert canonical.B_matrix is not None
    canonical.B_matrix[0][0][0] = 99
    np.testing.assert_array_equal(model.B_matrix, stored)


def test_canonical_gridworld_has_no_spurious_dimension_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        build_geo_infer_artifact(SOURCE.read_text(), step_seconds=60)
    assert "B matrix dimensions" not in caplog.text


_DOUBLY_STOCHASTIC_SLICES: list[list[list[float]]] = [
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
]
_ROW_STOCHASTIC_SLICES: list[list[list[float]]] = [
    [[0.8, 0.2, 0.0], [0.7, 0.3, 0.0], [0.0, 0.6, 0.4]],
    [[0.1, 0.9, 0.0], [0.0, 0.4, 0.6], [0.3, 0.3, 0.4]],
    [[0.6, 0.4, 0.0], [0.2, 0.2, 0.6], [0.5, 0.0, 0.5]],
]
_ACTINF_B_LITERAL = (
    "B={\n"
    "  ( (1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0) ),\n"
    "  ( (0.0,1.0,0.0), (1.0,0.0,0.0), (0.0,0.0,1.0) ),\n"
    "  ( (0.0,0.0,1.0), (0.0,1.0,0.0), (1.0,0.0,0.0) )\n"
    "}"
)


def _b_literal(slices: list[list[list[float]]]) -> str:
    body = ",\n".join(
        "  ( "
        + ", ".join("(" + ", ".join(f"{v:.1f}" for v in row) + ")" for row in slc)
        + " )"
        for slc in slices
    )
    return "B={\n" + body + "\n}"


def _b_source(
    tmp_path: Path,
    name: str,
    slices: list[list[list[float]]],
    *,
    declared_comment: str,
) -> Path:
    """actinf exemplar copy with test-owned B comments and literal."""
    content = (
        ROOT / "input" / "gnn_files" / "discrete" / "actinf_pomdp_agent.md"
    ).read_text()
    lines = content.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        if line.strip().startswith("# Transition matrix: B["):
            out.append(declared_comment + "\n")
            continue
        out.append(line)
    content = "".join(out).replace(_ACTINF_B_LITERAL, _b_literal(slices))
    assert _b_literal(slices) in content, "B literal swap failed"
    path = tmp_path / name
    path.write_text(content)
    return path


def test_ambiguous_b_refused_even_when_canonical_order_is_declared(
    tmp_path: Path,
) -> None:
    """Doubly-stochastic data is undecidable; prose cannot substitute."""
    path = _b_source(
        tmp_path,
        "ambiguous_declared_canonical.md",
        _DOUBLY_STOCHASTIC_SLICES,
        declared_comment="# Transition matrix: B[next_state, previous_state, actions]",
    )
    with pytest.raises(ValueError, match="not decisive"):
        build_geo_infer_artifact(path.read_text(), step_seconds=60)


def test_noncanonical_storage_refused_despite_matching_declaration(
    tmp_path: Path,
) -> None:
    """Consistent action-outer declaration is still not exportable: never reordered."""
    path = _b_source(
        tmp_path,
        "action_outer_declared.md",
        _ROW_STOCHASTIC_SLICES,
        declared_comment="# Transition matrix: B[action, previous_state, next_state]",
    )
    with pytest.raises(
        ValueError, match=r"detected \['action', 'previous_state', 'next_state'\]"
    ):
        build_geo_infer_artifact(path.read_text(), step_seconds=60)
