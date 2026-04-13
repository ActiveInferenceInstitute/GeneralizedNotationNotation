#!/usr/bin/env python3
"""
Integration tests for all discrete GNN models through the PyMDP pipeline.

Tests the full Render → Execute → Analysis flow for every model in
``input/gnn_files/discrete/``, ensuring that:

1. POMDP extraction succeeds for each file
2. PyMDP rendering produces a valid script
3. Execution via ``run_simple_pymdp_simulation`` produces:
   - Valid beliefs (all values in [0, 1])
   - Beliefs summing to 1
   - Actions within the valid range
   - No NaN values
4. The analysis extractor can consume the results
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DISCRETE_DIR = REPO_ROOT / "input" / "gnn_files" / "discrete"

# All 9 discrete GNN model files
DISCRETE_MODELS = [
    "actinf_pomdp_agent.md",
    "bnlearn_causal_model.md",
    "deep_planning_horizon.md",
    "hmm_baseline.md",
    "markov_chain.md",
    "multi_armed_bandit.md",
    "simple_mdp.md",
    "tmaze_epistemic.md",
    "two_state_bistable.md",
]


def _pymdp_importable() -> bool:
    try:
        from pymdp.agent import Agent  # noqa: F401

        return True
    except ImportError:
        return False


def _extract_and_build_spec(model_file: str, timesteps: int = 12) -> dict:
    """Extract POMDP from GNN file and build a gnn_spec dict for execution."""
    from gnn.pomdp_extractor import extract_pomdp_from_file
    from render.pomdp_processor import POMDPRenderProcessor

    path = DISCRETE_DIR / model_file
    pomdp = extract_pomdp_from_file(path, strict_validation=False)
    assert pomdp is not None, f"extract_pomdp_from_file returned None for {model_file}"
    assert pomdp.num_states > 0, f"num_states must be > 0 for {model_file}"
    assert pomdp.num_observations > 0, f"num_observations must be > 0 for {model_file}"

    proc = POMDPRenderProcessor(DISCRETE_DIR)
    gnn_spec = proc._pomdp_to_gnn_spec(pomdp, timesteps=timesteps)
    return gnn_spec


# ---------------------------------------------------------------------------
# Parametrized extraction test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_file", DISCRETE_MODELS)
def test_pomdp_extraction(model_file: str) -> None:
    """Every discrete model file should successfully extract a POMDPStateSpace."""
    from gnn.pomdp_extractor import extract_pomdp_from_file

    path = DISCRETE_DIR / model_file
    if not path.is_file():
        pytest.skip(f"Fixture missing: {path}")

    pomdp = extract_pomdp_from_file(path, strict_validation=False)
    assert pomdp is not None, f"Extraction failed for {model_file}"
    assert pomdp.num_states > 0
    assert pomdp.num_observations > 0
    assert pomdp.A_matrix is not None, f"A matrix missing for {model_file}"


# ---------------------------------------------------------------------------
# Parametrized render test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_file", DISCRETE_MODELS)
def test_pymdp_render(model_file: str, tmp_path: Path) -> None:
    """Every discrete model should successfully render a PyMDP script."""
    from gnn.pomdp_extractor import extract_pomdp_from_file
    from render.pomdp_processor import POMDPRenderProcessor

    path = DISCRETE_DIR / model_file
    if not path.is_file():
        pytest.skip(f"Fixture missing: {path}")

    pomdp = extract_pomdp_from_file(path, strict_validation=False)
    assert pomdp is not None, f"Extraction failed for {model_file}"

    proc = POMDPRenderProcessor(tmp_path)
    result = proc.process_pomdp_for_all_frameworks(pomdp, path, frameworks=["pymdp"])

    fr = result.get("framework_results", {}).get("pymdp", {})
    assert fr.get("success"), (
        f"Render failed for {model_file}: {fr.get('message', 'unknown error')}"
    )

    # Verify the output file exists and is valid Python
    output_files = fr.get("output_files", [])
    assert output_files, f"No output files for {model_file}"
    for f in output_files:
        fpath = Path(f)
        assert fpath.exists(), f"Output file missing: {f}"
        content = fpath.read_text(encoding="utf-8")
        compile(content, fpath.name, "exec")  # syntax check


# ---------------------------------------------------------------------------
# Parametrized execute test (requires pymdp)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("model_file", DISCRETE_MODELS)
def test_pymdp_execute(model_file: str, tmp_path: Path) -> None:
    """Every discrete model should execute and produce valid simulation results."""
    if not _pymdp_importable():
        pytest.skip("PyMDP (inferactively-pymdp) not installed")

    path = DISCRETE_DIR / model_file
    if not path.is_file():
        pytest.skip(f"Fixture missing: {path}")

    from execute.pymdp.simple_simulation import run_simple_pymdp_simulation

    gnn_spec = _extract_and_build_spec(model_file, timesteps=12)

    np.random.seed(42)
    ok, results = run_simple_pymdp_simulation(gnn_spec, tmp_path / model_file.replace(".md", ""))

    assert ok, f"Execution failed for {model_file}: {results.get('error', 'unknown')}"
    assert results.get("framework") == "PyMDP"

    # Validate structure
    assert isinstance(results.get("observations"), list)
    assert isinstance(results.get("actions"), list)
    assert isinstance(results.get("beliefs"), list)
    assert len(results["observations"]) == 12

    # Validate beliefs
    validation = results.get("validation", {})
    assert validation.get("all_beliefs_valid"), f"Invalid beliefs for {model_file}"
    assert validation.get("beliefs_sum_to_one"), f"Beliefs don't sum to 1 for {model_file}"
    assert validation.get("actions_in_range"), f"Actions out of range for {model_file}"

    # No NaN check
    for belief in results.get("beliefs", []):
        assert not any(
            np.isnan(v) for v in belief
        ), f"NaN in beliefs for {model_file}"


# ---------------------------------------------------------------------------
# Parametrized analysis extractor test (requires pymdp)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("model_file", DISCRETE_MODELS)
def test_pymdp_analysis_extractor(model_file: str, tmp_path: Path) -> None:
    """Analysis extractor should consume simulation results from each model."""
    if not _pymdp_importable():
        pytest.skip("PyMDP (inferactively-pymdp) not installed")

    path = DISCRETE_DIR / model_file
    if not path.is_file():
        pytest.skip(f"Fixture missing: {path}")

    from analysis.framework_extractors import extract_pymdp_data
    from execute.pymdp.simple_simulation import run_simple_pymdp_simulation

    gnn_spec = _extract_and_build_spec(model_file, timesteps=12)

    np.random.seed(42)
    ok, results = run_simple_pymdp_simulation(gnn_spec, tmp_path / "run")
    if not ok:
        pytest.skip(f"Execution failed for {model_file}: {results.get('error')}")

    # Write results to the expected location for the extractor
    model_stem = model_file.replace(".md", "")
    impl_dir = tmp_path / "impl"
    sim_data_dir = impl_dir / "simulation_data"
    sim_data_dir.mkdir(parents=True, exist_ok=True)
    payload_file = sim_data_dir / f"{model_stem}_simulation_results.json"
    payload_file.write_text(json.dumps(results, default=str), encoding="utf-8")

    extracted = extract_pymdp_data(
        {
            "framework": "pymdp",
            "implementation_directory": str(impl_dir),
            "simulation_data": {},
        }
    )

    for key in ["free_energy", "beliefs", "observations", "actions"]:
        assert key in extracted, f"Missing '{key}' in extracted data for {model_file}"
    assert len(extracted["beliefs"]) > 0, f"No beliefs extracted for {model_file}"
    assert len(extracted["observations"]) > 0, f"No observations extracted for {model_file}"


# ---------------------------------------------------------------------------
# End-to-end pipeline test (render → execute → analysis)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_all_discrete_models_e2e(tmp_path: Path) -> None:
    """
    Run the full pipeline (process_render → process_execute → process_analysis)
    for all 9 discrete models together.
    """
    import shutil

    if not _pymdp_importable():
        pytest.skip("PyMDP (inferactively-pymdp) not installed")

    from analysis.processor import process_analysis
    from execute.processor import process_execute
    from render.processor import process_render

    # Copy all discrete models to input dir
    in_dir = tmp_path / "input" / "gnn_files"
    in_dir.mkdir(parents=True)
    model_count = 0
    for model_file in DISCRETE_MODELS:
        src = DISCRETE_DIR / model_file
        if src.is_file():
            shutil.copy(src, in_dir / model_file)
            model_count += 1

    assert model_count == 9, f"Expected 9 models, found {model_count}"

    base = tmp_path / "output"
    render_out = base / "11_render_output"
    exec_out = base / "12_execute_output"
    analysis_out = base / "16_analysis_output"

    # Step 11: Render
    render_ok = process_render(
        in_dir, render_out, verbose=False, frameworks=["pymdp"], strict_validation=False
    )
    assert render_ok, "process_render should succeed for all 9 discrete models"

    pymdp_scripts = list(render_out.rglob("*pymdp.py"))
    assert len(pymdp_scripts) >= 8, (
        f"Expected at least 8 pymdp scripts, got {len(pymdp_scripts)}: "
        f"{[s.name for s in pymdp_scripts]}"
    )

    # Step 12: Execute
    exec_ok = process_execute(
        in_dir,
        exec_out,
        verbose=False,
        frameworks="pymdp",
        render_output_dir=render_out,
        timeout=300,
    )
    assert exec_ok, "process_execute should complete without fatal errors"

    # Verify simulation results
    sim_results = list(exec_out.glob("**/pymdp/simulation_data/*simulation_results.json"))
    assert sim_results, f"No simulation_results.json under {exec_out}"

    for result_file in sim_results:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        assert payload.get("framework") == "PyMDP", f"Wrong framework in {result_file.name}"
        beliefs = payload.get("beliefs") or payload.get("simulation_trace", {}).get("beliefs")
        assert beliefs, f"No beliefs in {result_file.name}"

    # Step 16: Analysis
    analysis_ok = process_analysis(in_dir, analysis_out, verbose=False)
    assert analysis_ok, "process_analysis should succeed"
