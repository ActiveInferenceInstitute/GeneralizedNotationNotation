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
- **Scaling Orchestrator**: External driver for batch simulation and meta-analysis.
