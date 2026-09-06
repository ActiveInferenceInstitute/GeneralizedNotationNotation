#!/usr/bin/env python3
"""Pure-function tests for ml_integration feature extraction and helpers.

No sklearn, no network: these exercise the deterministic GNN parsing and
numeric-projection helpers only.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FULL_BLOCK_GNN = """\
## GNNSection
pomdp
## StateSpaceBlock
A[3,2]
B[2,3,4]
s[2]
o[3]
## Connections
s > o
s > o
s - o
## ActInfOntologyAnnotation
state: environment state
## InitialParameterization
precision: 1.0
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


# --- extract_gnn_features: full block file ---------------------------------


def test_extract_gnn_features_full_block_dimensions(tmp_path: Path) -> None:
    from gnn.ml_integration import extract_gnn_features

    features = extract_gnn_features(_write(tmp_path, "model.md", FULL_BLOCK_GNN))
    assert features["num_states"] == 2
    assert features["num_observations"] == 3
    assert features["num_actions"] == 4
    assert features["num_variables"] == 4
    # 3*2 + 2*3*4 + 2 + 3 = 6 + 24 + 2 + 3
    assert features["total_parameters"] == 35
    assert features["max_dimension"] == 4


def test_extract_gnn_features_connectivity_ratio(tmp_path: Path) -> None:
    from gnn.ml_integration import extract_gnn_features

    features = extract_gnn_features(_write(tmp_path, "model.md", FULL_BLOCK_GNN))
    # 2 directed + 1 undirected over n*(n-1) = 4*3
    assert features["directed_connections"] == 2
    assert features["undirected_connections"] == 1
    assert features["connectivity_ratio"] == 3 / (4 * 3)


def test_extract_gnn_features_ontology_and_parameterization_flags(
    tmp_path: Path,
) -> None:
    from gnn.ml_integration import extract_gnn_features

    features = extract_gnn_features(_write(tmp_path, "model.md", FULL_BLOCK_GNN))
    assert features["has_ontology"] is True
    assert features["has_parameterization"] is True


def test_extract_gnn_features_state_observation_fallback(tmp_path: Path) -> None:
    from gnn.ml_integration import extract_gnn_features

    content = "## StateSpaceBlock\ns[5]\no[6]\n"
    features = extract_gnn_features(_write(tmp_path, "fallback.md", content))
    assert features["num_states"] == 5
    assert features["num_observations"] == 6


def test_extract_gnn_features_A_block_wins_over_state(tmp_path: Path) -> None:
    from gnn.ml_integration import extract_gnn_features

    content = "## StateSpaceBlock\nA[3,2]\ns[9]\n"
    features = extract_gnn_features(_write(tmp_path, "a_wins.md", content))
    assert features["num_states"] == 2
    assert features["num_observations"] == 3


def test_extract_gnn_features_missing_file_returns_empty(tmp_path: Path) -> None:
    from gnn.ml_integration import extract_gnn_features

    assert extract_gnn_features(tmp_path / "nope.md") == {}


# --- qualitative flags ------------------------------------------------------


def test_extract_gnn_features_has_learning_keywords(tmp_path: Path) -> None:
    from gnn.ml_integration import extract_gnn_features

    for keyword in ("dirichlet", "concentration"):
        content = f"## InitialParameterization\n{keyword} parameters\n"
        features = extract_gnn_features(_write(tmp_path, "learn.md", content))
        assert features["has_learning"] is True, keyword


def test_extract_gnn_features_has_precision_requires_dimension_line(
    tmp_path: Path,
) -> None:
    from gnn.ml_integration import extract_gnn_features

    word_only = "## InitialParameterization\nprecision: 1.0\n"
    features = extract_gnn_features(_write(tmp_path, "word_only.md", word_only))
    assert features["has_precision"] is False

    full = "## InitialParameterization\nprecision: 1.0\nω[2,2]\n"
    features = extract_gnn_features(_write(tmp_path, "full.md", full))
    assert features["has_precision"] is True


# --- _detect_model_family ---------------------------------------------------


def test_detect_model_family_section_variants() -> None:
    from gnn.ml_integration.processor import _detect_model_family

    variants = {
        "hmm": "hmm",
        "pomdp": "pomdp",
        "hierarchical": "hierarchical",
        "continuous": "continuous",
        "multiagent": "multi_agent",
        "multi_agent": "multi_agent",
        "factor_graph": "factor_graph",
    }
    for token, expected in variants.items():
        content = f"## GNNSection\n\n{token}\n"
        assert _detect_model_family(content) == expected, token


def test_detect_model_family_fallbacks() -> None:
    from gnn.ml_integration.processor import _detect_model_family

    assert _detect_model_family("B[2,3,4]\nπ[3]\n") == "pomdp"
    assert _detect_model_family("B[d,d,d]\n") == "hmm"
    assert _detect_model_family("A[3,2]\n") == "unknown"


# --- _extract_dimensions ----------------------------------------------------


def test_extract_dimensions_skips_and_stops_at_header() -> None:
    from gnn.ml_integration.processor import _extract_dimensions

    content = "## StateSpaceBlock\nA[3, type=discrete]\nB[2, x, 4]\n## Other\nC[7,7]\n"
    dims = _extract_dimensions(content)
    assert dims == {"A": [3], "B": [2, 4]}


# --- _count_connections -----------------------------------------------------


def test_count_connections_directed_vs_undirected() -> None:
    from gnn.ml_integration.processor import _count_connections

    assert _count_connections("## Connections\ns > o\n") == {
        "directed": 1,
        "undirected": 0,
    }
    assert _count_connections("## Connections\ns - o\n") == {
        "directed": 0,
        "undirected": 1,
    }
    assert _count_connections("## Connections\ns -> o\n") == {
        "directed": 1,
        "undirected": 0,
    }


def test_count_connections_scope_and_comments() -> None:
    from gnn.ml_integration.processor import _count_connections

    content = "s > o\n## Connections\ns > o\n# s > o (comment)\n## Notes\ns - o\n"
    assert _count_connections(content) == {"directed": 1, "undirected": 0}


# --- _extract_planning_horizon / _extract_time_type -------------------------


def test_extract_planning_horizon_variants() -> None:
    from gnn.ml_integration.processor import _extract_planning_horizon

    assert _extract_planning_horizon("PLANNING_HORIZON: 4\n") == 4
    assert _extract_planning_horizon("planning_horizon: 4\n") == 4
    assert _extract_planning_horizon("ModelTimeHorizon = 7\n") == 7
    assert _extract_planning_horizon("nothing relevant\n") == 1


def test_extract_time_type_bare_lines() -> None:
    from gnn.ml_integration.processor import _extract_time_type

    assert _extract_time_type("Continuous\n") == "continuous"
    assert _extract_time_type("Discrete\n") == "discrete"
    assert _extract_time_type("time: Continuous\n") == "unknown"
    assert _extract_time_type("") == "unknown"


# --- feature_vector ---------------------------------------------------------


def test_feature_vector_order_matches_numeric_feature_names() -> None:
    from gnn.ml_integration import NUMERIC_FEATURE_NAMES, feature_vector

    values = [float(i) for i in range(len(NUMERIC_FEATURE_NAMES))]
    mapping = dict(zip(NUMERIC_FEATURE_NAMES, values))
    assert feature_vector(mapping) == values


def test_feature_vector_defaults() -> None:
    from gnn.ml_integration import NUMERIC_FEATURE_NAMES, feature_vector

    vector = feature_vector({})
    assert len(vector) == len(NUMERIC_FEATURE_NAMES)
    for name, value in zip(NUMERIC_FEATURE_NAMES, vector):
        expected = 1.0 if name == "planning_horizon" else 0.0
        assert value == expected, name


def test_feature_vector_booleans_and_type_coercion() -> None:
    from gnn.ml_integration import NUMERIC_FEATURE_NAMES, feature_vector

    vector = feature_vector({"has_precision": True, "has_learning": False})
    assert vector[NUMERIC_FEATURE_NAMES.index("has_precision")] == 1.0
    assert vector[NUMERIC_FEATURE_NAMES.index("has_learning")] == 0.0

    full = feature_vector({"num_states": 2, "connectivity_ratio": 0.25})
    assert full[NUMERIC_FEATURE_NAMES.index("num_states")] == 2.0
    assert full[NUMERIC_FEATURE_NAMES.index("connectivity_ratio")] == 0.25
    assert all(isinstance(value, float) for value in full)


# --- complexity_label -------------------------------------------------------


def test_complexity_label_boundaries() -> None:
    from gnn.ml_integration import complexity_label

    assert complexity_label(0) == "small"
    assert complexity_label(99) == "small"
    assert complexity_label(100) == "medium"
    assert complexity_label(999) == "medium"
    assert complexity_label(1000) == "large"
    assert complexity_label(10_000) == "large"


# --- summarize_features -----------------------------------------------------


def test_summarize_features_min_max_mean() -> None:
    from gnn.ml_integration import summarize_features

    stats = summarize_features(
        [
            {"num_states": 2, "num_observations": 3, "total_parameters": 10},
            {"num_states": 6, "num_observations": 3, "total_parameters": 30},
            {"num_states": 4, "num_observations": 3, "total_parameters": 20},
        ]
    )
    assert stats["num_states"] == {"min": 2, "max": 6, "mean": 4.0}
    assert stats["num_observations"] == {"min": 3, "max": 3, "mean": 3.0}
    assert stats["total_parameters"] == {"min": 10, "max": 30, "mean": 20.0}


def test_summarize_features_skips_non_numeric_and_missing() -> None:
    from gnn.ml_integration import summarize_features

    stats = summarize_features(
        [
            {"num_states": 2, "model_family": "hmm", "total_parameters": 10.0},
            {"num_states": 4, "model_family": "pomdp", "total_parameters": 20.0},
        ]
    )
    assert stats["num_states"] == {"min": 2, "max": 4, "mean": 3.0}
    assert stats["total_parameters"] == {"min": 10.0, "max": 20.0, "mean": 15.0}
    assert "model_family" not in stats

    assert summarize_features([{"num_states": "many"}]) == {}
    assert summarize_features([]) == {}


# --- _structural_analysis_entries ------------------------------------------


def test_structural_analysis_entries_shape_and_note() -> None:
    from gnn.ml_integration.processor import _structural_analysis_entries

    all_features = [
        {
            "file_name": "a.md",
            "model_family": "pomdp",
            "num_states": 2,
            "total_parameters": 35,
        },
        {
            "file_name": "b.md",
            "model_family": "hmm",
            "num_states": 3,
            "total_parameters": 12,
        },
    ]
    entries = _structural_analysis_entries(all_features, "my note")
    assert len(entries) == 2
    for entry, source in zip(entries, all_features):
        assert entry["source"] == source["file_name"]
        assert entry["type"] == "structural_analysis"
        assert entry["framework"] == "internal_stats"
        assert entry["validation_status"] == "not_applicable"
        assert entry["note"] == "my note"
        assert entry["model_family"] == source["model_family"]
        assert entry["num_states"] == source["num_states"]
        assert entry["total_parameters"] == source["total_parameters"]


def test_structural_analysis_entries_get_defaults() -> None:
    from gnn.ml_integration.processor import _structural_analysis_entries

    entries = _structural_analysis_entries([{"file_name": "c.md"}], "fallback note")
    assert entries == [
        {
            "source": "c.md",
            "type": "structural_analysis",
            "framework": "internal_stats",
            "validation_status": "not_applicable",
            "note": "fallback note",
            "model_family": None,
            "num_states": 0,
            "total_parameters": 0,
        }
    ]
