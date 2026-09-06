# Execute Tests

Pytest coverage for `src/gnn/execute/`.

This folder contains module-focused tests for execution processors, framework runners, and script safety.

## Test Files

- `test_discrete_models_pymdp.py` — PyMDP discrete-model execution behavior.
- `test_execute_benchmark_samples.py` — benchmark aggregation for repeated Step 12 executions.
- `test_execute_envelope_factories.py` — execution envelope factory construction.
- `test_execute_introspection.py` — executor introspection behavior.
- `test_execute_mcp_wiring.py` — MCP wiring of the execute module.
- `test_execute_outcome_classification.py` — execution outcome classification.
- `test_execute_overall.py` — module-level aggregate contract for the execute folder.
- `test_execute_path_collection.py` — script path collection.
- `test_execute_plan.py` — execution plan construction.
- `test_execute_pymdp_integration.py` — PyMDP integration behavior.
- `test_execute_pymdp_integration_module.py` — PyMDP integration module wiring.
- `test_execute_pymdp_package.py` — PyMDP package integration.
- `test_execute_pymdp_simulation.py` — PyMDP simulation execution.
- `test_execute_pymdp_utils.py` — PyMDP execution utilities.
- `test_execute_pymdp_visualization_module.py` — PyMDP visualization module wiring.
- `test_execute_pymdp_visualizer.py` — PyMDP visualizer behavior.
- `test_execute_result_semantics.py` — execution result semantics.
- `test_execute_script_safely.py` — sandboxed script execution.
- `test_execute_slim_detail.py` — slim report-detail behavior.
- `test_execute_stan.py` — Stan execution behavior.
- `test_executor_framework_coverage.py` — framework registry coverage.
- `test_kronecker_factorized.py` — Kronecker-factorized JAX execution.
- `test_lean_executor.py` — registration of the `lean` executor framework.
- `test_lean_runner.py` — toolchain-guarded Lean verification receipt flow.
- `test_pymdp_1_0_0_upstream_api.py` — PyMDP 1.0.0 upstream API compatibility surface.
- `test_pymdp_contracts.py` — PyMDP rendering contracts.
- `test_receipt_reliability.py` — execution receipt reliability.
- `test_run_pymdp_gnn_scaling_estimate.py` — bounds for generated GNN size estimates in `scripts/run_pymdp_gnn_scaling_analysis.py`.

Run:

```bash
uv run --extra dev python -m pytest tests/execute/ -q
```
