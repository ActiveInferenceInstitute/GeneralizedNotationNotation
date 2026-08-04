"""Contract tests for RxInfer.jl visualization and structured-logging blocks.

Verifies that the canonical RxInfer.jl renderer emits optional, guarded
Julia-native visualisation (Plots.jl -> PNGs) and a structured per-step
execution log (simulation.log / simulation_log.json) alongside the existing
simulation_results.json contract.

Public functions:
- test_generated_source_contains_viz_and_log_blocks
- test_rendered_script_emits_results_log_and_png
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from gnn.pomdp_extractor import extract_pomdp_from_file
from render.pomdp_processor import POMDPRenderProcessor
from render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer

REPO_ROOT = Path(__file__).resolve().parents[3]
SIMPLE_MDP = REPO_ROOT / "input" / "gnn_files" / "discrete" / "simple_mdp.md"

JULIA = shutil.which("julia")


def _render_simple_mdp(tmp_path: Path) -> Path:
    """Render the small simple_mdp exemplar to a Julia script and return it."""
    assert SIMPLE_MDP.exists(), f"missing exemplar: {SIMPLE_MDP}"
    pomdp_space = extract_pomdp_from_file(SIMPLE_MDP, strict_validation=True)
    assert pomdp_space is not None, "extract_pomdp_from_file returned None"
    gnn_spec = POMDPRenderProcessor(tmp_path)._pomdp_to_gnn_spec(pomdp_space)
    output_path = tmp_path / "simple_mdp_rxinfer.jl"
    success, message, _warnings = render_gnn_to_rxinfer(gnn_spec, output_path)
    assert success, f"render_gnn_to_rxinfer failed: {message}"
    assert output_path.exists()
    return output_path


# --- (a) generated SOURCE contains the visualization and logging blocks -------


def test_generated_source_contains_viz_and_log_blocks(tmp_path: Path) -> None:
    output_path = _render_simple_mdp(tmp_path)
    source = output_path.read_text(encoding="utf-8")

    # Structured logging block.
    assert "simulation.log" in source
    assert "simulation_log.json" in source
    assert "JSON.print" in source
    assert "write_execution_log" in source
    assert '"action" => actions[step]' in source
    assert '"expected_free_energy" => efe[step]' in source
    assert '"validation" => validation' in source

    # Guarded visualization block (Plots.jl, matplotlib-free PNGs).
    assert "using Plots" in source
    assert "write_plots" in source
    assert "savefig" in source
    assert "belief_evolution.png" in source
    assert "efe_over_time.png" in source
    assert "policy_posterior.png" in source

    # The plotting block must be guarded so a missing backend never crashes
    # the script, and the simulation_results.json contract must stay intact.
    assert "try" in source
    assert "catch e" in source
    assert "simulation_results.json" in source
    assert "JSON.print(file, results, 2)" in source
    assert '"schema_version" => SCHEMA_VERSION' in source


# --- (b) execution produces simulation_results.json + log + at least one PNG --
# Runs only when Julia is on PATH. It is further gated on RANDOM_SIMULATION_ENABLED
# (truthy) so a normal unit run stays fast; set that env var to exercise it.


@pytest.mark.skipif(JULIA is None, reason="julia is not on PATH")
@pytest.mark.skipif(
    (os.environ.get("RANDOM_SIMULATION_ENABLED") or "").strip().lower()
    not in {"1", "true", "yes", "on"},
    reason="RANDOM_SIMULATION_ENABLED not set; execution test is optional",
)
def test_rendered_script_emits_results_log_and_png(tmp_path: Path) -> None:
    output_path = _render_simple_mdp(tmp_path)

    completed = subprocess.run(
        [JULIA, "--project=/tmp/rx2", str(output_path)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert completed.returncode == 0, (
        f"julia script failed (rc={completed.returncode})\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )

    # Existing contract preserved.
    results_path = tmp_path / "simulation_results.json"
    assert results_path.exists(), "simulation_results.json was not produced"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["schema_version"] == "rxinfer_simulation_v1"
    assert results["success"] is True
    assert results["validation"]["all_valid"] is True

    # Structured per-step log emitted.
    log_path = tmp_path / "simulation.log"
    assert log_path.exists(), "simulation.log was not produced"
    log_lines = [
        line
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(log_lines) >= 2, "expected per-step + summary records"
    first = json.loads(log_lines[0])
    assert first["event"] == "step"
    assert first["step"] == 1
    assert "belief" in first
    assert "action" in first
    assert "expected_free_energy" in first
    assert "validation" in first
    summary = json.loads(log_lines[-1])
    assert summary["event"] == "summary"

    # Structured JSON sidecar present.
    assert (tmp_path / "simulation_log.json").exists()

    # At least one Julia-native PNG plot produced.
    pngs = list(tmp_path.glob("*.png"))
    assert pngs, "no PNG plots were produced"
