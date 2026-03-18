# JAX Renderer Specification

## Overview

The `render.jax` submodule implements Step 11 (Render) backend support for generating **JAX-based Python scripts** from parsed GNN specifications.

This specification defines:

- the stable public API exported by `src/render/jax/__init__.py`,
- the input contract for `gnn_spec`,
- output artifacts and success criteria,
- dependency expectations for generated scripts.

## Public API

`src/render/jax/__init__.py` exports exactly:

- `render_gnn_to_jax(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`
- `render_gnn_to_jax_pomdp(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`
- `render_gnn_to_jax_combined(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`

No other functions are part of the supported surface unless re-exported there.

## Inputs

### `gnn_spec`

The renderer expects `gnn_spec` as a parsed dictionary, typically produced by earlier pipeline steps.

It supports multiple internal shapes for extracting \(A, B, C, D\) via `jax_renderer._extract_gnn_matrices`, including:

- POMDP-extractor style dicts with `model_parameters` and `initialparameterization`
- JSON export style dicts with `statespaceblock` and `raw_sections["InitialParameterization"]`
- older parsed dicts with `variables` and `InitialParameterization`

If extraction fails or is partial, the implementation applies defaults and recovery normalization.

### `output_path`

- Must be a file path (typically ending in `.py`).
- Parent directories are created if missing.
- The function writes exactly one file per call and returns that path in `generated_files`.

### `options`

`options` is an optional dict passed through to the generator. Callers should not rely on specific keys unless explicitly documented by the implementation.

## Outputs

Each API call writes a single Python script to `output_path`:

- **General model**: pure JAX + NumPy
- **POMDP solver**: JAX; optional Optax import with fallback behavior in generated code
- **Combined model**: JAX + Flax + Optax

Step 11 may place these scripts under:

`output/11_render_output/<model_name>/jax/` (organization handled by the parent render processor).

## Dependencies

### Module runtime (render-time)

- Python: **3.11+** (repo policy)
- `numpy`

### Generated script runtime

- `jax`, `jaxlib` (required)
- `numpy`
- `optax` (required for combined model; optional-with-fallback for POMDP solver)
- `flax` (required for combined model)

## Non-goals

- This module does not define a thin orchestrator; Step 11 orchestration is handled by `src/11_render.py` calling `render.process_render`.
- `src/render/jax/templates/` is not part of the active code-generation path unless the implementation is explicitly refactored to use it.
