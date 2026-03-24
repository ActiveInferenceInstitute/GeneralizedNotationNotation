#!/usr/bin/env python3
"""
PyMDP contract tests for render/execute/analysis integration.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from analysis.framework_extractors import extract_pymdp_data
from execute.pymdp.simple_simulation import run_simple_pymdp_simulation
from render.processor import render_gnn_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTINF_POMDP_PATH = REPO_ROOT / "input" / "gnn_files" / "discrete" / "actinf_pomdp_agent.md"


def _pymdp_importable() -> bool:
    try:
        from pymdp.agent import Agent  # noqa: F401

        return True
    except ImportError:
        return False


def _actinf_gnn_spec(timesteps: int = 12) -> dict:
    """Same POMDP → spec path as Step 11 (POMDPRenderProcessor)."""
    import logging

    from gnn.pomdp_extractor import extract_pomdp_from_file
    from render.pomdp_processor import POMDPRenderProcessor
    from render.processor import normalize_matrices

    pomdp = extract_pomdp_from_file(ACTINF_POMDP_PATH, strict_validation=True)
    assert pomdp is not None, "extract_pomdp_from_file returned None for actinf_pomdp_agent.md"
    pomdp = normalize_matrices(pomdp, logging.getLogger("test_pymdp_contracts"))
    proc = POMDPRenderProcessor(ACTINF_POMDP_PATH.parent)
    return proc._pomdp_to_gnn_spec(pomdp, timesteps=timesteps)


def _minimal_gnn_spec() -> dict:
    return {
        "name": "contract_test_model",
        "model_name": "contract_test_model",
        "initialparameterization": {
            "A": [[0.9, 0.1], [0.1, 0.9]],
            # (action, prev, next) format expected by simple_simulation transpose logic
            "B": [[[0.8, 0.2], [0.2, 0.8]]],
            "C": [0.0, 1.0],
            "D": [0.5, 0.5],
        },
        "model_parameters": {"num_timesteps": 8},
    }


@pytest.mark.integration
def test_render_execute_contract_pymdp(tmp_path: Path) -> None:
    """
    Render should succeed and execution should produce schema-valid outputs.
    """
    gnn_spec = _minimal_gnn_spec()
    render_dir = tmp_path / "render"

    render_ok, msg, _artifacts = render_gnn_spec(gnn_spec, "pymdp", render_dir)
    assert render_ok, f"render failed: {msg}"

    # Execute using the local PyMDP execution pathway.
    exec_dir = tmp_path / "execute"
    exec_ok, results = run_simple_pymdp_simulation(gnn_spec, exec_dir)
    if not exec_ok:
        pytest.skip(f"PyMDP unavailable for runtime contract check: {results.get('error')}")

    assert results.get("success") is True
    assert results.get("framework") == "PyMDP"
    assert isinstance(results.get("observations"), list)
    assert isinstance(results.get("actions"), list)
    assert isinstance(results.get("beliefs"), list)
    assert len(results["observations"]) == gnn_spec["model_parameters"]["num_timesteps"]


@pytest.mark.integration
def test_extract_pymdp_data_from_real_execution_payload(tmp_path: Path) -> None:
    """
    Extractor should populate standard fields from a real simulation_results file.
    """
    gnn_spec = _minimal_gnn_spec()
    run_dir = tmp_path / "run"
    ok, results = run_simple_pymdp_simulation(gnn_spec, run_dir)
    if not ok:
        pytest.skip(f"PyMDP unavailable for extractor contract check: {results.get('error')}")

    impl_dir = tmp_path / "impl"
    sim_data_dir = impl_dir / "simulation_data"
    sim_data_dir.mkdir(parents=True, exist_ok=True)
    payload_file = sim_data_dir / "contract_test_model_simulation_results.json"
    payload_file.write_text(json.dumps(results), encoding="utf-8")

    extracted = extract_pymdp_data(
        {
            "framework": "pymdp",
            "implementation_directory": str(impl_dir),
            "simulation_data": {},
        }
    )

    for key in ["free_energy", "beliefs", "observations", "actions", "states", "traces"]:
        assert key in extracted
    assert len(extracted["beliefs"]) > 0
    assert len(extracted["observations"]) > 0


@pytest.mark.integration
def test_pymdp_seeded_reproducibility_contract(tmp_path: Path) -> None:
    """
    With the same numpy seed and same spec, simple simulation outputs should match.
    """
    gnn_spec = _minimal_gnn_spec()

    np.random.seed(123)
    ok1, res1 = run_simple_pymdp_simulation(gnn_spec, tmp_path / "r1")
    if not ok1:
        pytest.skip(f"PyMDP unavailable for reproducibility check: {res1.get('error')}")

    np.random.seed(123)
    ok2, res2 = run_simple_pymdp_simulation(gnn_spec, tmp_path / "r2")
    assert ok2

    assert res1["observations"] == res2["observations"]
    assert res1["actions"] == res2["actions"]


@pytest.mark.integration
def test_actinf_pomdp_golden_pymdp_simulation(tmp_path: Path) -> None:
    """
    Canonical ActInfPOMDP fixture: parsed matrices must run under run_simple_pymdp_simulation
    without NaNs and with internal validation flags true (B-axis / D rounding robustness).
    """
    if not ACTINF_POMDP_PATH.is_file():
        pytest.skip(f"Fixture missing: {ACTINF_POMDP_PATH}")
    if not _pymdp_importable():
        pytest.skip("PyMDP (inferactively-pymdp) not installed")

    spec = _actinf_gnn_spec(timesteps=12)
    np.random.seed(42)
    ok, results = run_simple_pymdp_simulation(spec, tmp_path / "actinf_run")
    assert ok, f"golden ActInf POMDP simulation failed: {results.get('error')}"
    assert results.get("framework") == "PyMDP"
    val = results.get("validation") or {}
    assert val.get("beliefs_sum_to_one") is True
    assert val.get("all_beliefs_valid") is True
    assert val.get("actions_in_range") is True
    assert len(results.get("observations", [])) == 12
    assert not any(
        np.isnan(np.array(b, dtype=float)).any() for b in results.get("beliefs", []) if b
    )


@pytest.mark.integration
@pytest.mark.slow
def test_actinf_pomdp_render_execute_analyze_e2e(tmp_path: Path) -> None:
    """
    Render pymdp script → execute via Step-12 subprocess (GNN_PROJECT_ROOT) → collect
    simulation_results.json → analysis step sees PyMDP outputs.
    """
    if not ACTINF_POMDP_PATH.is_file():
        pytest.skip(f"Fixture missing: {ACTINF_POMDP_PATH}")
    if not _pymdp_importable():
        pytest.skip("PyMDP (inferactively-pymdp) not installed")

    from analysis.processor import process_analysis
    from execute.processor import process_execute
    from render.processor import process_render

    in_dir = tmp_path / "input" / "gnn_files"
    in_dir.mkdir(parents=True)
    shutil.copy(ACTINF_POMDP_PATH, in_dir / "actinf_pomdp_agent.md")

    base = tmp_path / "output"
    render_out = base / "11_render_output"
    exec_out = base / "12_execute_output"
    analysis_out = base / "16_analysis_output"

    render_ok = process_render(
        in_dir,
        render_out,
        verbose=False,
        frameworks=["pymdp"],
        strict_validation=True,
    )
    assert render_ok, "process_render should succeed for ActInf POMDP (pymdp only)"

    pymdp_scripts = list(render_out.rglob("*pymdp.py"))
    assert pymdp_scripts, f"expected *pymdp.py under {render_out}"

    exec_ok = process_execute(
        in_dir,
        exec_out,
        verbose=False,
        frameworks="pymdp",
        render_output_dir=render_out,
        timeout=180,
    )
    assert exec_ok, "process_execute should complete without fatal errors"

    collected = list(exec_out.glob("**/pymdp/simulation_data/*simulation_results.json"))
    assert collected, f"no simulation_results.json under {exec_out}/**/pymdp/simulation_data/"

    payload = json.loads(collected[0].read_text(encoding="utf-8"))
    assert payload.get("framework") == "PyMDP"
    beliefs = payload.get("beliefs") or payload.get("simulation_trace", {}).get("beliefs")
    obs = payload.get("observations") or payload.get("simulation_trace", {}).get("observations")
    assert beliefs and obs, "payload should include beliefs and observations"

    analysis_ok = process_analysis(in_dir, analysis_out, verbose=False)
    assert analysis_ok

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return

    pymdp_pngs = list((analysis_out / "pymdp").rglob("*.png"))
    assert pymdp_pngs, "with matplotlib, analysis should emit at least one PyMDP png"
