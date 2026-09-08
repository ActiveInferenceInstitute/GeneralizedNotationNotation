"""Pins for the DisCoPy analyzer visualization pipeline (previously 12%)."""

from __future__ import annotations

import json
from pathlib import Path

from gnn.analysis.discopy.analyzer import (
    create_discopy_visualizations,
    generate_analysis_from_logs,
)

_CIRCUIT_PAYLOAD = {
    "components": ["box_a", "box_b", "box_c"],
    "analysis": {"num_components": 3, "loop_domain": "discrete"},
    "parameters": {"num_states": 2, "num_observations": 2},
}


def test_create_discopy_visualizations_renders_files(tmp_path: Path) -> None:
    generated = create_discopy_visualizations(
        dict(_CIRCUIT_PAYLOAD), tmp_path, "CircuitModel"
    )

    assert generated, "expected at least one rendered visualization"
    for path in generated:
        assert Path(path).exists()
        assert Path(path).suffix in {".png", ".svg", ".html"}


def test_create_discopy_visualizations_skips_empty_circuits(tmp_path: Path) -> None:
    generated = create_discopy_visualizations(
        {"components": [], "num_components": 0}, tmp_path, "EmptyModel"
    )

    assert generated == []


def test_walker_processes_circuit_analysis_files(tmp_path: Path) -> None:
    sim_data = tmp_path / "CircuitModel" / "discopy" / "simulation_data"
    sim_data.mkdir(parents=True)
    (sim_data / "circuit_analysis.json").write_text(
        json.dumps(_CIRCUIT_PAYLOAD), encoding="utf-8"
    )
    out = tmp_path / "viz"

    generated = generate_analysis_from_logs(tmp_path, out, verbose=False)

    assert generated
    assert out.is_dir()


def test_walker_routes_source_file_metadata(tmp_path: Path) -> None:
    sim_data = tmp_path / "MetaModel" / "discopy" / "simulation_data"
    sim_data.mkdir(parents=True)
    (sim_data / "circuit_info.json").write_text(
        json.dumps(_CIRCUIT_PAYLOAD), encoding="utf-8"
    )
    out = tmp_path / "viz"

    generated = generate_analysis_from_logs(tmp_path, out, verbose=False)

    assert generated
