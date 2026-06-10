# RxInfer.jl Render Module

This module renders validated GNN POMDP specifications to executable RxInfer.jl scripts.

## Public Surface

- `render_gnn_to_rxinfer(gnn_spec, output_path, options=None)`
- Step 11 calls this renderer through `render.pomdp_processor.POMDPRenderProcessor`.
- `render_gnn_spec(..., "rxinfer", ...)` routes to the same canonical renderer for POMDP specs.

## Contract

The renderer consumes `canonical_pomdp_v1` data:

- `A`: observation likelihood, shape `(observation, state)`
- `B`: transition tensor, shape `(next_state, previous_state, action)`
- `C`: observation preferences
- `D`: initial hidden-state prior
- `E`: policy prior when present
- matrix provenance and runtime metadata

Generated scripts import RxInfer.jl and write `simulation_results.json` with schema `rxinfer_simulation_v1`.

## Outputs

```text
output/11_render_output/<model>/rxinfer/
├── <model>_rxinfer.jl
└── README.md
```

Step 12 collects runtime outputs into:

```text
output/12_execute_output/<model>/rxinfer/simulation_data/simulation_results.json
```

## Verification

```bash
julia --startup-file=no -e 'using RxInfer, JSON, Distributions, StatsBase'
uv run --extra dev python -m pytest src/tests/pipeline/test_pomdp_gridworld_cross_framework.py -q --tb=short
```
