# Step 11: Render — POMDP-Aware Code Generation

## Overview

Renders GNN specifications into executable simulation scripts for 7 computational frameworks. Includes POMDP-aware processing that extracts state spaces, normalizes probability matrices (A/B), and injects them into framework-specific code templates.

## Usage

```bash
python src/11_render.py --target-dir input/gnn_files --output-dir output --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/11_render.py` (67 lines) |
| Module | `src/render/` |
| Processor | `src/render/processor.py` |
| POMDP Processor | `src/render/pomdp_processor.py` (942 lines) |
| Module function | `process_render()` |

## POMDP Processing Pipeline

1. **Extract** POMDP state space from GNN markdown via `extract_pomdp_from_file()`
2. **Validate** dimensions (states, observations, actions) via `validate_pomdp_for_rendering()`
3. **Normalize** A and B matrices to proper probability distributions via `normalize_matrices()`
4. **Render** to all 7 frameworks via `POMDPRenderProcessor`

## Supported Frameworks

| Framework | Language | Renderer |
|-----------|----------|----------|
| PyMDP | Python | `src/render/pymdp/pymdp_renderer.py` |
| RxInfer | Julia | `src/render/rxinfer/rxinfer_renderer.py` |
| ActiveInference.jl | Julia | `src/render/activeinference_jl/activeinference_renderer.py` |
| JAX | Python | `src/render/jax/jax_renderer.py` |
| DisCoPy | Python | `src/render/discopy/` |
| PyTorch | Python | `src/render/pytorch/pytorch_renderer.py` |
| NumPyro | Python | `src/render/numpyro/numpyro_renderer.py` |

## Output

- **Directory**: `output/11_render_output/`
- Per-model directories with per-framework subdirectories containing executable scripts, README docs, and `render_processing_summary.json`

## Source

- **Script**: [src/11_render.py](#placeholder)
