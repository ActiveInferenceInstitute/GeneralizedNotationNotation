"""Tolerance contracts for the per-framework ``generate_analysis_from_logs`` walkers.

The analyzers walk Step-12 output trees and must degrade silently (warn +
continue) on malformed or missing inputs, and must always return a list.
These tests exercise the walkers without depending on matplotlib internals.
"""

from __future__ import annotations

import json
from pathlib import Path

from gnn.analysis.activeinference_jl.analyzer import (
    generate_analysis_from_logs as generate_aijl,
)
from gnn.analysis.discopy.analyzer import (
    generate_analysis_from_logs as generate_discopy,
)
from gnn.analysis.jax.analyzer import (
    generate_analysis_from_logs as generate_jax,
)


def test_discopy_walker_empty_execution_dir_returns_empty(tmp_path: Path) -> None:
    out = tmp_path / "viz"

    assert generate_discopy(tmp_path, out, verbose=False) == []


def test_discopy_walker_tolerates_malformed_circuit_json(tmp_path: Path) -> None:
    sim_data = tmp_path / "model" / "discopy" / "simulation_data"
    sim_data.mkdir(parents=True)
    (sim_data / "circuit_analysis.json").write_text("{broken", encoding="utf-8")
    out = tmp_path / "viz"

    result = generate_discopy(tmp_path, out, verbose=False)

    assert result == []


def test_aijl_walker_empty_execution_dir_creates_output_dir(tmp_path: Path) -> None:
    out = tmp_path / "viz" / "aijl"

    result = generate_aijl(tmp_path, out, verbose=False)

    assert result == []
    assert out.is_dir()


def test_aijl_walker_tolerates_malformed_results_json(tmp_path: Path) -> None:
    sim_data = tmp_path / "model" / "activeinference_jl" / "simulation_data"
    sim_data.mkdir(parents=True)
    (sim_data / "run_simulation_results.json").write_text("{broken", encoding="utf-8")
    out = tmp_path / "viz"

    result = generate_aijl(tmp_path, out, verbose=False)

    assert result == []


def test_jax_walker_empty_execution_dir_returns_empty(tmp_path: Path) -> None:
    out = tmp_path / "viz"

    assert generate_jax(tmp_path, out, verbose=False) == []


def test_jax_walker_tolerates_malformed_results_json(tmp_path: Path) -> None:
    logs = tmp_path / "model" / "jax" / "execution_logs"
    logs.mkdir(parents=True)
    (logs / "run_results.json").write_text("{broken", encoding="utf-8")
    out = tmp_path / "viz"

    result = generate_jax(tmp_path, out, verbose=False)

    assert result == []


def test_discopy_walker_skips_dirs_without_simulation_data(tmp_path: Path) -> None:
    (tmp_path / "model" / "discopy").mkdir(parents=True)
    out = tmp_path / "viz"

    assert generate_discopy(tmp_path, out, verbose=False) == []


def test_json_payload_that_is_not_a_dict_is_tolerated(tmp_path: Path) -> None:
    """A valid-JSON but non-object payload (e.g. a bare list) must not crash
    the walker — the framework extraction contract is dict-shaped."""
    sim_data = tmp_path / "model" / "discopy" / "simulation_data"
    sim_data.mkdir(parents=True)
    (sim_data / "circuit_analysis.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    out = tmp_path / "viz"

    assert generate_discopy(tmp_path, out, verbose=False) == []
