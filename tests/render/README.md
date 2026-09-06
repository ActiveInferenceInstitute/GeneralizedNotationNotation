# Render Tests

Pytest coverage for `src/gnn/render/`.

This folder contains module-focused tests for render processors, framework generators, MCP wiring, and renderer contracts.

## Test Files

- `test_activeinference_matrix_formatting.py` — ActiveInference.jl renderer helper functions: robust conversion of Python data structures (lists, tuples, nesting) to Julia matrix literals.
- `test_continuous_public_contract.py` — public continuous dispatch and execution output contracts.
- `test_continuous_renderers.py` — continuous (linear-Gaussian) branch of the JAX / NumPyro / PyTorch / Stan renderers.
- `test_discopy_improvements.py` — DisCoPy translator error handling and setup reporting under differing DisCoPy/JAX availability.
- `test_discopy_symmetry_contract.py` — DisCoPy matrix permutation metadata and dimension-mismatch rejection contract.
- `test_fep_bridge_render.py` — accepted FEP bridge source renders through both public structured inputs.
- `test_framework_availability.py` — framework availability gating in the canonical render registry.
- `test_generators_coverage.py` — generator coverage across the renderer backends (bnlearn, pymdp, ActiveInference.jl, and peers).
- `test_jax_factorized_pipeline.py` — Kronecker-factorized pipeline integration (MAJ-02): numbered pipeline route for factor-separable GNN specs.
- `test_jax_renderer.py` — live generated-script checks for the JAX renderer.
- `test_pomdp_contract_types.py` — TypedDict contracts and the `ModelKind` enum in `pomdp_contract.py`.
- `test_pomdp_renderer_regressions.py` — regression tests for canonical POMDP rendering edge cases.
- `test_render_cli_targets.py` — mechanical guard for the renderer CLI target choices and their dispatch branches.
- `test_render_contracts.py` — contract tests for render shared helpers and processor policy.
- `test_render_integration.py` — render integration with the pipeline: rendered output is consumable by downstream steps.
- `test_render_mcp_wiring.py` — MCP wiring: signature and return-shape parity between MCP tool adapters and the render API.
- `test_render_numpyro_stan.py` — end-to-end render tests for the NumPyro and Stan backends (GNN → render → compile check).
- `test_render_overall.py` — module-level aggregate coverage across render targets.
- `test_render_performance.py` — performance characteristics of code generation and rendering.
- `test_render_process_discovery.py` — `process_render` recursive discovery of nested exemplar GNN files.
- `test_render_pytorch_renderer.py` — focused tests for the PyTorch render backend.
- `test_render_receipt_reliability.py` — render receipts must not count retries or earlier runs twice.
- `test_render_stan.py` — tests for the Stan renderer component.
- `test_rxinfer_efe_correctness.py` — expected-free-energy formula correctness in generated RxInfer code.
- `test_rxinfer_model_strategies.py` — RxInfer `ModelKind` detection and strategy dispatch regressions.
- `test_rxinfer_multiagent_contract.py` — RxInfer multi-agent key handling and agent-count contract.
- `test_rxinfer_viz_log_contract.py` — RxInfer.jl visualization and structured-logging block contracts.
- `test_stigmergic_multi_agent.py` — native stigmergic multi-agent compilation contract (MAJ-03).
- `test_toml_matrix_parser.py` — parenthesized-tuple matrix parsers in `src/gnn/render/rxinfer/toml_generator.py`.

Run:

```bash
uv run --extra dev python -m pytest tests/render/ -q
```
