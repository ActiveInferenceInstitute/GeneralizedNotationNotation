# Experiments

One-off and offline tooling, not repository gates.

- `add_module_docstrings.py` — source-mutating codemod; run deliberately, never in CI.
- `run_pymdp_gnn_scaling_analysis.py` — PyMDP scaling-study orchestrator (uses `../pymdp_scaling_config.yaml`).
- `verify_pomdp_pipeline.py` — retained as an offline verifier only; the maintained gate is `scripts/check_pomdp_gridworld_outputs.py`.