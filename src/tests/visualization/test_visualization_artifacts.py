#!/usr/bin/env python3
"""Tests for step-8 sidecars: network stats orientation, ontology legend, viz manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visualization.core.process import process_single_gnn_file
from visualization.graph.network_visualizations import generate_network_visualizations


@pytest.mark.unit
def test_network_stats_gnn_edge_orientation_and_ontology_legend(tmp_path: Path) -> None:
    out = tmp_path / "viz"
    out.mkdir()
    parsed = {
        "variables": [
            {"name": "a", "var_type": "hidden_state"},
            {"name": "b", "var_type": "hidden_state"},
            {"name": "c", "var_type": "observation"},
        ],
        "connections": [
            {
                "source_variables": ["a"],
                "target_variables": ["b"],
                "connection_type": "directed",
            },
            {
                "source_variables": ["b"],
                "target_variables": ["c"],
                "connection_type": "undirected",
            },
        ],
        "ontology_labels": {"a": "HiddenState", "c": "Observation"},
    }
    paths = generate_network_visualizations(parsed, out, "mini")
    if not paths:
        pytest.skip("networkx/matplotlib not available for graph generation")

    stats_path = out / "mini_network_stats.json"
    assert stats_path.is_file()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert "gnn_edge_orientation" in stats
    orient = stats["gnn_edge_orientation"]
    assert orient["directed_variable_pairs"] == 1
    assert orient["undirected_variable_pairs"] == 1

    leg = out / "mini_ontology_legend.txt"
    assert leg.is_file()
    text = leg.read_text(encoding="utf-8")
    assert "variable\tontology_term" in text
    assert "a\tHiddenState" in text
    assert "c\tObservation" in text


@pytest.mark.unit
def test_viz_manifest_json_after_process_single_gnn_file(tmp_path: Path) -> None:
    base = tmp_path
    gnn_in = base / "in"
    gnn_in.mkdir()
    gnn_file = gnn_in / "tiny.md"
    gnn_file.write_text("# tiny\n", encoding="utf-8")

    step3_model = base / "3_gnn_output" / "tiny"
    step3_model.mkdir(parents=True)
    parsed = {
        "model_name": "tiny",
        "variables": [
            {"name": "x", "var_type": "hidden_state", "dimensions": [2]},
            {"name": "y", "var_type": "observation", "dimensions": [2]},
        ],
        "connections": [
            {
                "source_variables": ["x"],
                "target_variables": ["y"],
                "connection_type": "directed",
            },
        ],
        "parameters": [
            {
                "name": "A",
                "type": "matrix",
                "shape": [2, 2],
                "values": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "ontology_mappings": [
            {"variable_name": "x", "ontology_term": "HiddenState"},
        ],
        "raw_sections": {},
    }
    parsed_path = step3_model / "tiny_parsed.json"
    parsed_path.write_text(json.dumps(parsed), encoding="utf-8")

    results_dir = base / "8_visualization_output"
    results_dir.mkdir(parents=True)

    paths = process_single_gnn_file(gnn_file, results_dir, verbose=False)
    assert paths, "expected at least one visualization artifact"

    model_dir = results_dir / "tiny"
    manifest_path = model_dir / "tiny_viz_manifest.json"
    assert manifest_path.is_file(), f"missing manifest; got paths: {paths[:5]}..."

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_name"] == "tiny"
    assert manifest["ontology_label_count"] == 1
    assert "viz_meta" in manifest
    assert manifest["viz_meta"].get("source") == "parsed_json"
    assert isinstance(manifest["artifacts"], list)
    assert manifest["artifact_count"] == len(manifest["artifacts"])

    leg = model_dir / "tiny_ontology_legend.txt"
    assert leg.is_file()

    stats_path = model_dir / "tiny_network_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        assert "gnn_edge_orientation" in stats
