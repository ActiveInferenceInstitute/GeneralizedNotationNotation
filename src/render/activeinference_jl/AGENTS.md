# ActiveInference.jl Render Agent Guide

## Purpose

Render validated GNN POMDP specs to executable ActiveInference.jl scripts.

## Ownership Boundary

- Maintain `render_gnn_to_activeinference_jl(...)` as the canonical ActiveInference.jl render surface.
- Consume `canonical_pomdp_v1` specs with explicit `A/B/C/D` and optional `E`.
- Preserve B order as `(next_state, previous_state, action)`.
- Generated artifacts belong under ignored output trees.

## Public Surfaces

- `render_gnn_to_activeinference_jl(gnn_spec, output_path, options=None)`
- `render_gnn_spec(..., "activeinference_jl", ...)`
- Step 11 via `POMDPRenderProcessor`

## Outputs

- Rendered script: `output/11_render_output/<model>/activeinference_jl/<model>_activeinference.jl`
- Runtime schema after Step 12: `activeinference_jl_simulation_v1`

## Verification

```bash
julia --startup-file=no -e 'using ActiveInference, JSON, Distributions, StatsBase'
uv run pytest src/tests/pipeline/test_pomdp_gridworld_cross_framework.py -q --tb=short
```
