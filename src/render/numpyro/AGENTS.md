# NumPyro Render Backend - Agent Scaffolding

## Module Overview

**Purpose**: Render parsed GNN/POMDP specifications to **standalone NumPyro simulation scripts**.

**Parent**: `src/render/` (Step 11: Render)

**Primary entrypoint**: `render_gnn_to_numpyro` in `numpyro_renderer.py` (re-exported by `__init__.py`).

---

## Public API

From `src/render/numpyro/__init__.py`:

- `render_gnn_to_numpyro(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`

**Contract**:
- writes exactly one Python file to `output_path`
- returns `(success, message, [output_path])` on success

---

## Implementation notes

`numpyro_renderer.py` follows the same high-level contract as other render backends:

- matrix extraction from common internal `gnn_spec` shapes (`stateSpace.parameters`, `initialparameterization`, `parameters`)
- shape sanity checks via `render.matrix_utils.validate_abcd_shapes`
- normalization via `render.matrix_utils.normalize_columns`
- standalone script generation (includes dependency checks and clear install hints)

The generated script writes `simulation_results.json` under `NUMPYRO_OUTPUT_DIR` (default: current directory).

---

## Integration points

- **Called by**: `render.pomdp_processor.POMDPRenderProcessor` when the `numpyro` framework is selected and the backend is importable.
- **Consumed by**: `src/execute/numpyro/` runner (Step 12) and `src/analysis/numpyro/` analyzer (Step 16).

---

## Testing

Preferred tests:

- formatting/syntax validation: generated script compiles (`python -m py_compile`)
- integration: Step 11 produces `numpyro/<model>_numpyro.py` when NumPyro backend is available

End-to-end execution tests should live under `src/tests/` and be resilient to optional dependency availability.

