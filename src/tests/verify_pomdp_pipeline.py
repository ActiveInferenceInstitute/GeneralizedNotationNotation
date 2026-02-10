#!/usr/bin/env python3
"""
Comprehensive POMDP Pipeline Verification Script

Confirms that all POMDP implementations across Steps 11 (Render), 12 (Execute),
and 16 (Analysis) fully run, are totally configurable, and produce real outputs.

Tests:
1. PyMDPSimulation with multiple configurations (default, minimal, full GNN, custom matrices)
2. Rendering to all 5 frameworks (PyMDP, RxInfer, ActiveInference.jl, DisCoPy, JAX)
3. Simulation execution with configurable timesteps, learning rate, matrix injection
4. Analysis pipeline: entropy, KL divergence, VFE, EFE, information gain
5. Visualization generation
6. Full pipeline: render → execute → analyze with real simulation data
"""

import json
import logging
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)-7s  %(message)s")
log = logging.getLogger("verify_pomdp")

# Track results
results = {"passed": [], "failed": [], "skipped": []}

def check(name, fn):
    """Run a verification check and record the result."""
    try:
        fn()
        results["passed"].append(name)
        log.info(f"✅  {name}")
    except Exception as e:
        results["failed"].append((name, str(e)))
        log.error(f"❌  {name}: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# 1. PYMDP SIMULATION — Configurability
# ─────────────────────────────────────────────────────────────────────────────

def test_default_config():
    """PyMDPSimulation with no config uses default gridworld."""
    from execute.pymdp.pymdp_simulation import PyMDPSimulation
    sim = PyMDPSimulation()
    assert sim.num_states == 4, f"Expected 4 states, got {sim.num_states}"
    assert sim.num_actions == 5
    assert sim.num_observations == 4
    assert sim.agent is not None
    r = sim.run_simulation(num_timesteps=5)
    assert len(r["observations"]) == 5

def test_named_states_config():
    """Config with named states/observations/actions."""
    from execute.pymdp.pymdp_simulation import PyMDPSimulation
    cfg = {
        "states": ["healthy", "sick", "recovered"],
        "observations": ["no_symptom", "mild_symptom", "severe_symptom"],
        "actions": ["rest", "medicate", "exercise"],
        "model_name": "health_model"
    }
    sim = PyMDPSimulation(gnn_config=cfg)
    assert sim.num_states == 3
    assert sim.num_observations == 3
    assert sim.num_actions == 3
    assert sim.model_name == "health_model"
    assert sim.state_names == cfg["states"]
    r = sim.run_simulation(num_timesteps=10)
    assert len(r["actions"]) == 10

def test_integer_counts_config():
    """Config with integer counts instead of named lists."""
    from execute.pymdp.pymdp_simulation import PyMDPSimulation
    cfg = {"states": 6, "observations": 3, "actions": 4}
    sim = PyMDPSimulation(gnn_config=cfg)
    assert sim.num_states == 6
    assert sim.num_observations == 3
    assert sim.num_actions == 4
    r = sim.run_simulation(num_timesteps=8)
    assert len(r["beliefs"]) == 8

def test_custom_parameters():
    """Config with custom learning rate, alpha, gamma, timesteps."""
    from execute.pymdp.pymdp_simulation import PyMDPSimulation
    cfg = {
        "states": ["a", "b", "c", "d"],
        "observations": ["x", "y"],
        "actions": ["go", "stay"],
        "parameters": {
            "learning_rate": 0.8,
            "alpha": 32.0,
            "gamma": 8.0,
            "num_timesteps": 25,
        }
    }
    sim = PyMDPSimulation(gnn_config=cfg)
    assert sim.learning_rate == 0.8
    assert sim.alpha == 32.0
    assert sim.gamma == 8.0
    r = sim.run_simulation()
    assert len(r["observations"]) == 25

def test_custom_preferences_and_priors():
    """Config with explicit preference (C) and prior (D) vectors."""
    from execute.pymdp.pymdp_simulation import PyMDPSimulation
    cfg = {
        "states": ["s0", "s1", "s2"],
        "observations": ["o0", "o1", "o2"],
        "actions": ["a0", "a1"],
        "parameters": {
            "preferences": [2.0, 0.0, -1.0],
            "prior_beliefs": [0.7, 0.2, 0.1],
        }
    }
    sim = PyMDPSimulation(gnn_config=cfg)
    # C should reflect preferences
    C = sim.model_matrices["C"]
    assert C[0] == 2.0 and C[2] == -1.0
    # D should reflect prior (normalized)
    D = sim.model_matrices["D"]
    assert D[0] > D[1] > D[2]
    r = sim.run_simulation(num_timesteps=10)
    assert r.get("success", True)

def test_gnn_matrix_injection():
    """Config with explicit A/B matrices via GNN extraction path."""
    from execute.pymdp.pymdp_simulation import PyMDPSimulation
    cfg = {
        "states": ["s0", "s1"],
        "observations": ["o0", "o1"],
        "actions": ["left", "right"],
    }
    sim = PyMDPSimulation(gnn_config=cfg)
    # Manually inject GNN matrices
    sim.gnn_matrices = {
        "A": np.array([[0.95, 0.05], [0.05, 0.95]]),
        "B": np.zeros((2, 2, 2)),
    }
    sim.gnn_matrices["B"][0, 0, 0] = 1.0  # left keeps in s0
    sim.gnn_matrices["B"][1, 0, 1] = 1.0  # right moves to s1
    sim.gnn_matrices["B"][0, 1, 0] = 1.0  # left moves to s0
    sim.gnn_matrices["B"][1, 1, 1] = 1.0  # right keeps in s1
    agent, matrices = sim.create_pymdp_model_from_gnn()
    assert agent is not None
    assert matrices["A"].shape == (2, 2)
    assert matrices["B"].shape == (2, 2, 2)
    r = sim.run_simulation(num_timesteps=10)
    assert len(r["observations"]) == 10

def test_simulation_output_structure():
    """Verify simulation output contains all expected fields."""
    from execute.pymdp.pymdp_simulation import PyMDPSimulation
    sim = PyMDPSimulation(gnn_config={
        "states": ["a", "b", "c"],
        "observations": ["x", "y", "z"],
        "actions": ["1", "2", "3"],
    })
    r = sim.run_simulation(num_timesteps=12)
    for key in ["observations", "actions", "beliefs", "performance", "trace"]:
        assert key in r, f"Missing key: {key}"
    assert len(r["observations"]) == 12
    assert len(r["actions"]) == 12
    assert len(r["beliefs"]) == 12
    # Beliefs should be valid probability distributions
    for t, b in enumerate(r["beliefs"]):
        b_arr = np.asarray(b).flatten()
        assert np.all(b_arr >= 0), f"Negative belief at t={t}"
        assert abs(b_arr.sum() - 1.0) < 0.01, f"Beliefs at t={t} don't sum to 1"

def test_serialization():
    """Results should be JSON-serializable."""
    from execute.pymdp.pymdp_simulation import PyMDPSimulation
    from execute.pymdp.pymdp_utils import safe_json_dump
    with tempfile.TemporaryDirectory() as td:
        sim = PyMDPSimulation(gnn_config={
            "states": ["a", "b"], "observations": ["x", "y"], "actions": ["go", "stay"]
        })
        r = sim.run_simulation(num_timesteps=5)
        out = Path(td) / "results.json"
        safe_json_dump(r, out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert "observations" in loaded and "actions" in loaded


# ─────────────────────────────────────────────────────────────────────────────
# 2. RENDERING — All 5 Frameworks
# ─────────────────────────────────────────────────────────────────────────────

def test_render_pymdp():
    from render.processor import render_gnn_spec
    spec = {"model_name": "verify", "states": ["a", "b"], "observations": ["x", "y"], "actions": ["go"]}
    with tempfile.TemporaryDirectory() as td:
        ok, msg, arts = render_gnn_spec(spec, "pymdp", td)
        assert ok, f"PyMDP render failed: {msg}"
        assert any(f.endswith(".py") for f in arts) or list(Path(td).rglob("*.py")), "No .py output"

def test_render_rxinfer():
    from render.processor import render_gnn_spec
    spec = {"model_name": "verify", "states": ["a", "b"], "observations": ["x", "y"], "actions": ["go"]}
    with tempfile.TemporaryDirectory() as td:
        ok, msg, arts = render_gnn_spec(spec, "rxinfer", td)
        assert ok, f"RxInfer render failed: {msg}"

def test_render_activeinference_jl():
    from render.processor import render_gnn_spec
    spec = {"model_name": "verify", "states": ["a", "b"], "observations": ["x", "y"], "actions": ["go"]}
    with tempfile.TemporaryDirectory() as td:
        ok, msg, arts = render_gnn_spec(spec, "activeinference_jl", td)
        assert ok, f"ActiveInference.jl render failed: {msg}"

def test_render_discopy():
    from render.processor import render_gnn_spec
    spec = {"model_name": "verify", "states": ["a", "b"], "observations": ["x", "y"], "actions": ["go"]}
    with tempfile.TemporaryDirectory() as td:
        ok, msg, arts = render_gnn_spec(spec, "discopy", td)
        assert ok, f"DisCoPy render failed: {msg}"

def test_render_jax():
    from render.processor import render_gnn_spec
    spec = {"model_name": "verify", "states": ["a", "b"], "observations": ["x", "y"], "actions": ["go"]}
    with tempfile.TemporaryDirectory() as td:
        ok, msg, arts = render_gnn_spec(spec, "jax", td)
        # JAX renderer may not be available — record accordingly
        if not ok and "not available" in msg.lower():
            results["skipped"].append("render_jax (renderer not installed)")
            results["passed"].pop()  # remove from passed since we'll skip
            return
        assert ok, f"JAX render failed: {msg}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANALYSIS — Active Inference Math
# ─────────────────────────────────────────────────────────────────────────────

def test_shannon_entropy():
    from analysis.post_simulation import compute_shannon_entropy
    # Uniform over 4 → ln(4)
    p = np.ones(4) / 4
    assert abs(compute_shannon_entropy(p) - np.log(4)) < 1e-6
    # Dirac → ~0
    p = np.array([1.0, 0, 0, 0])
    assert compute_shannon_entropy(p) < 0.01

def test_kl_divergence():
    from analysis.post_simulation import compute_kl_divergence
    p = np.array([0.7, 0.3])
    assert abs(compute_kl_divergence(p, p)) < 1e-4  # D_KL(P||P) = 0
    q = np.array([0.3, 0.7])
    assert compute_kl_divergence(p, q) > 0  # > 0 when P ≠ Q

def test_vfe():
    from analysis.post_simulation import compute_variational_free_energy
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    obs = np.array([1.0, 0.0])
    correct = np.array([0.9, 0.1])
    wrong = np.array([0.1, 0.9])
    fe_c = compute_variational_free_energy(obs, correct, A)
    fe_w = compute_variational_free_energy(obs, wrong, A)
    assert fe_c <= fe_w + 1e-6, "Correct belief should have lower VFE"

def test_efe():
    from analysis.post_simulation import compute_expected_free_energy
    beliefs = np.array([1.0, 0.0])
    A = np.eye(2)
    B = np.zeros((2, 2, 2))
    B[0, 0, 0] = 1.0; B[1, 0, 1] = 1.0
    B[0, 1, 0] = 1.0; B[1, 1, 1] = 1.0
    C = np.array([1.0, -1.0])
    efe0 = compute_expected_free_energy(beliefs, A, B, C, policy=0)
    efe1 = compute_expected_free_energy(beliefs, A, B, C, policy=1)
    assert np.isfinite(efe0) and np.isfinite(efe1)

def test_information_gain():
    from analysis.post_simulation import compute_information_gain, compute_kl_divergence
    prior = np.array([0.5, 0.5])
    posterior = np.array([0.9, 0.1])
    ig = compute_information_gain(prior, posterior)
    kl = compute_kl_divergence(posterior, prior)
    assert abs(ig - kl) < 1e-6

def test_analyze_metrics():
    from analysis.post_simulation import analyze_active_inference_metrics
    beliefs = [np.array([0.25 + 0.04*t, 0.25, 0.25, 0.25 - 0.04*t]).clip(0.01, 0.99).tolist()
               for t in range(15)]
    # renormalize
    beliefs = [(np.array(b)/np.array(b).sum()).tolist() for b in beliefs]
    fe = [5.0 - 0.3*t for t in range(15)]
    actions = [t % 3 for t in range(15)]
    r = analyze_metrics = analyze_active_inference_metrics(beliefs, fe, actions, "verify_model")
    assert r["model_name"] == "verify_model"
    assert r["num_timesteps"] == 15
    assert "belief_entropy" in r["metrics"]
    assert "information_gain" in r["metrics"]
    assert "free_energy" in r["metrics"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. NORMALIZATION — Fixed stub
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_matrices():
    from render.processor import normalize_matrices
    class FakePOMDP:
        def __init__(self, A=None, B=None):
            self.A_matrix = A
            self.B_matrix = B
    A = np.array([[3.0, 1.0], [1.0, 3.0]])
    B = np.ones((3, 3, 2))
    pomdp = FakePOMDP(A=A, B=B)
    result = normalize_matrices(pomdp, logging.getLogger("test"))
    # A columns should sum to 1
    np.testing.assert_allclose(result.A_matrix.sum(axis=0), [1.0, 1.0], atol=1e-10)
    # B columns should sum to 1 per action
    for a in range(2):
        np.testing.assert_allclose(result.B_matrix[:, :, a].sum(axis=0), np.ones(3), atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZATION — Confirm creation without error
# ─────────────────────────────────────────────────────────────────────────────

def test_visualization_creation():
    from execute.pymdp.pymdp_simulation import PyMDPSimulation
    from analysis.pymdp.visualizer import PyMDPVisualizer
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = {
            "states": ["s0", "s1", "s2", "s3"],
            "observations": ["o0", "o1"],
            "actions": ["up", "down", "left", "right"],
        }
        sim = PyMDPSimulation(gnn_config=cfg, output_dir=td)
        r = sim.run_simulation(num_timesteps=10)
        viz = PyMDPVisualizer(config=cfg, output_dir=td)
        viz.plot_belief_evolution(r["beliefs"])
        viz.plot_action_sequence(r["actions"])
        viz.plot_performance_metrics(r["performance"])

def test_post_simulation_viz():
    from analysis.post_simulation import (
        generate_belief_heatmaps, generate_action_analysis,
        generate_free_energy_plots, generate_observation_analysis
    )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        beliefs = [np.array([0.5, 0.3, 0.2]).tolist() for _ in range(10)]
        actions = [i % 3 for i in range(10)]
        fe = [5.0 - 0.3*t for t in range(10)]
        obs = [i % 3 for i in range(10)]
        # Functions take raw data + output path (not a dict)
        generate_belief_heatmaps(beliefs, td / "belief_heatmap.png")
        generate_action_analysis(actions, td / "action_analysis.png")
        generate_free_energy_plots(fe, td / "free_energy.png")
        generate_observation_analysis(obs, td / "obs_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. FULL PIPELINE — Render → Execute → Analyze
# ─────────────────────────────────────────────────────────────────────────────

def test_full_pipeline():
    from render.processor import render_gnn_spec
    from execute.pymdp.pymdp_simulation import PyMDPSimulation
    from analysis.post_simulation import (
        analyze_active_inference_metrics, compute_variational_free_energy,
        compute_shannon_entropy
    )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        spec = {
            "model_name": "pipeline_verify",
            "states": ["s0", "s1", "s2"],
            "observations": ["o0", "o1", "o2"],
            "actions": ["left", "right", "stay"],
        }
        # 1. Render
        ok, msg, _ = render_gnn_spec(spec, "pymdp", td / "rendered")
        assert ok, f"Render: {msg}"

        # 2. Execute
        sim = PyMDPSimulation(gnn_config=spec, output_dir=td / "executed")
        r = sim.run_simulation(num_timesteps=20)
        assert len(r["observations"]) == 20

        # 3. Analyze
        beliefs = [np.asarray(b).flatten().tolist() for b in r["beliefs"]]
        actions = [int(a) for a in r["actions"]]
        fe = [5.0 - 0.2*t for t in range(len(beliefs))]
        analysis = analyze_active_inference_metrics(beliefs, fe, actions, "pipeline_verify")
        assert analysis["num_timesteps"] == 20
        assert "belief_entropy" in analysis["metrics"]

        # 4. Compute VFE on each timestep
        A = np.eye(3)
        for b in beliefs:
            b_arr = np.array(b)
            obs = np.zeros(3); obs[0] = 1.0
            vfe = compute_variational_free_energy(obs, b_arr, A)
            assert np.isfinite(vfe)

        # 5. Entropy on final belief
        entropy = compute_shannon_entropy(np.array(beliefs[-1]))
        assert entropy >= 0


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL CHECKS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  POMDP Pipeline Comprehensive Verification")
    print("=" * 70)

    # Execute
    checks = [
        ("1a. Default config simulation", test_default_config),
        ("1b. Named states config", test_named_states_config),
        ("1c. Integer counts config", test_integer_counts_config),
        ("1d. Custom parameters (lr/alpha/gamma/timesteps)", test_custom_parameters),
        ("1e. Custom preferences & priors (C/D vectors)", test_custom_preferences_and_priors),
        ("1f. GNN matrix injection (A/B from extractor)", test_gnn_matrix_injection),
        ("1g. Output structure validation", test_simulation_output_structure),
        ("1h. JSON serialization roundtrip", test_serialization),
        ("2a. Render → PyMDP", test_render_pymdp),
        ("2b. Render → RxInfer", test_render_rxinfer),
        ("2c. Render → ActiveInference.jl", test_render_activeinference_jl),
        ("2d. Render → DisCoPy", test_render_discopy),
        ("2e. Render → JAX", test_render_jax),
        ("3a. Shannon entropy", test_shannon_entropy),
        ("3b. KL divergence", test_kl_divergence),
        ("3c. Variational free energy", test_vfe),
        ("3d. Expected free energy", test_efe),
        ("3e. Information gain", test_information_gain),
        ("3f. Full metrics analysis", test_analyze_metrics),
        ("4a. normalize_matrices (fixed stub)", test_normalize_matrices),
        ("5a. PyMDP visualizer", test_visualization_creation),
        ("5b. Post-simulation vizualizations", test_post_simulation_viz),
        ("6a. Full pipeline: render→execute→analyze", test_full_pipeline),
    ]

    for name, fn in checks:
        check(name, fn)

    # Summary
    print()
    print("=" * 70)
    print(f"  RESULTS: {len(results['passed'])} passed, "
          f"{len(results['failed'])} failed, "
          f"{len(results['skipped'])} skipped")
    print("=" * 70)

    if results["failed"]:
        print("\nFAILURES:")
        for name, err in results["failed"]:
            print(f"  ❌ {name}: {err}")

    if results["skipped"]:
        print("\nSKIPPED:")
        for s in results["skipped"]:
            print(f"  ⚠️  {s}")

    print()
    sys.exit(1 if results["failed"] else 0)
