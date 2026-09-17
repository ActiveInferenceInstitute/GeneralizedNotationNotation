# Specification: Pymdp

## Design Requirements
This module (`pymdp`) maps structural logic to the overall execution graph.
It ensures that `Pymdp` tasks resolve without runtime dependency loops.

## Components
1. **The Integration Contract** (`gnn_pymdp.md`): Defines the mathematical mapping from GNN primitives to PyMDP 1.0.0 (JAX-first) structures.
2. **Performance Benchmarking** (`run_pymdp_gnn_scaling_analysis.py`): Automated parametric sweeps (N, T) for empirical complexity analysis.
3. **Execution Safety**: Strict preflight resource gating for O(n³) dense B tensor expansion.

## Interfaces
- **Step 11 (Render)**: Generates JAX-optimized runner scripts.
- **Step 12 (Execute)**: Invokes the PyMDP 1.0.0 Agent rollout loop.
- **Model-Kind Coverage**: Discrete categorical A/B/C/D[/E] only — continuous
  linear-Gaussian models are unsupported (`supports_continuous: False` in
  `src/gnn/render/framework_registry.py`); rendered scripts execute at Step 12
  (`supports_execution: True`).
- **Scaling Orchestrator**: External driver for batch simulation and meta-analysis.
