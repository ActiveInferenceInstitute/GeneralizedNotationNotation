## RxInfer.jl Render Agent Guide

## Purpose

Render validated GNN POMDP specs to executable RxInfer.jl scripts using genuine
``@model`` + ``infer()`` variational message-passing inference.

The canonical renderer (`rxinfer_renderer.py`) emits Julia scripts that define a
real generative POMDP model with ``@model function pomdp_model(...)`` using
``Categorical`` and ``DiscreteTransition`` nodes, then run ``infer()`` with
``free_energy=true`` to obtain posteriors over hidden states and real
variational free energy traces.

**Pipeline**: The generated script implements **offline batch inference
(Bayesian smoothing) with post-hoc EFE policy evaluation** — NOT online active
inference. The four phases are:

1. Phase 1: Forward simulation for data collection (hand-rolled EFE)
2. Phase 2: Real RxInfer ``infer()`` with ``free_energy=true`` — if this fails,
   the script crashes (no fallback, no try/catch)
3. Phase 3: Smoothed posterior extraction from ``result.posteriors[:s]``
4. Phase 4: Post-hoc EFE and policy from smoothed posteriors

The per-iteration VFE trace (``vfe_per_iteration``) is the real convergence
diagnostic. ``variational_free_energy`` is reported as the per-iteration vector
(length = INFERENCE_ITERATIONS), not a per-step constant.

Validation includes ``inference_converged``, ``vfe_present``, and
``belief_entropy_ok`` (rejects degenerate beliefs for non-identity A matrices).

Exemplar discovery is **recursive**: all GNN spec files under `input/gnn_files/**`
(nested folders such as `discrete/`, `continuous/`, `basics/`) are discovered and rendered.
All **29** exemplar GNN files render to and execute under RxInfer.jl (29/29 render + execute).

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
