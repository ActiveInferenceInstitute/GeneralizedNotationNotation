# RxInfer.jl Render Agent Guide

## Purpose

Render validated GNN POMDP specs to executable RxInfer.jl scripts.

Exemplar discovery is **recursive**: all GNN spec files under `input/gnn_files/**`
(nested folders such as `discrete/`, `continuous/`, `basics/`) are discovered and rendered.
All **45** exemplar GNN files render to and execute under RxInfer.jl (45/45 render + execute).

## Ownership Boundary

- Maintain `render_gnn_to_rxinfer(...)` as the canonical RxInfer render surface.
- Consume `canonical_pomdp_v1` specs with explicit `A/B/C/D` and optional `E`.
- Preserve B order as `(next_state, previous_state, action)`.
- Generated artifacts belong under ignored output trees.

## Public Surfaces

- `render_gnn_to_rxinfer(gnn_spec, output_path, options=None)`
- `render_gnn_spec(..., "rxinfer", ...)`
- Step 11 via `POMDPRenderProcessor`

## Outputs

- Rendered script: `output/11_render_output/<model>/rxinfer/<model>_rxinfer.jl`
- Runtime schema after Step 12: `rxinfer_simulation_v1`

## Verification

```bash
julia --startup-file=no -e 'using RxInfer, JSON, Distributions, StatsBase'
uv run --extra dev python -m pytest src/tests/pipeline/test_pomdp_gridworld_cross_framework.py -q --tb=short
```
