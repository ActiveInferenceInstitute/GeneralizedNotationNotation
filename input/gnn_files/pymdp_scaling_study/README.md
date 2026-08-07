# PyMDP Scaling Study — retained exemplar subset

This directory holds a REPRESENTATIVE subset of the scaling sweep: one spec
per state-space size (N4, N8, N16, N32, N64), each at T=100.

The full sweep grid is GENERATED, not hand-authored: for every (N, T) pair in
`scripts/pymdp_scaling_config.yaml`, `scripts/run_pymdp_gnn_scaling_analysis.py`
creates a stochastic GNN spec with dense A (N×N) and B (N×N×N) matrices.
The previously committed full grid (7 N values × 3 T values, including three
21 MB N128 files — the B tensor is O(N³) text) was pruned 2026-08-07: the
T variants differ only in a runtime constant, and the sweep regenerates any
combination on demand. Run the sweep script to reproduce the full grid; do
not hand-edit the generated specs.
