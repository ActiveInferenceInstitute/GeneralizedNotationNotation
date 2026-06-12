# POMDP GridWorld Fixture Agent Guide

## Purpose

This directory contains the maintained GridWorld POMDP fixture used to verify that one GNN model can run through render, execute, analysis, and visualization for PyMDP, RxInfer.jl, and ActiveInference.jl.

## Ownership Boundary

- Keep fixtures small enough for strict local and CI runs.
- Preserve explicit `A/B/C/D/E` matrices and matrix provenance comments.
- Generated render, execute, and analysis artifacts belong under ignored output trees.

## Public Surfaces

- `pomdp_gridworld_3x3.md` is the canonical cross-framework fixture.
- The transition tensor `B` is stored as `(next_state, previous_state, action)`.
- Runtime metadata uses `num_timesteps: 15`, `random_seed: 42`, and five actions.
- Step 16 emits per-framework PNG plots, belief GIFs, 3x3 trajectory GIFs, a
  cross-framework trajectory GIF, and `gridworld_analysis_manifest.json` from the
  current execution outputs.

## Verification

```bash
uv run python src/main.py --only-steps "3,5,8,11,12,16" --target-dir input/gnn_files/pomdp_gridworld --frameworks "pymdp,rxinfer,activeinference_jl" --verbose
```
