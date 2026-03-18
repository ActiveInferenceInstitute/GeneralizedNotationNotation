# PyTorch Render Backend - Agent Scaffolding

## Module Overview

**Purpose**: Render parsed GNN/POMDP specifications to **standalone PyTorch simulation scripts**.

**Parent**: `src/render/` (Step 11: Render)

**Primary entrypoint**: `render_gnn_to_pytorch` in `pytorch_renderer.py` (re-exported by `__init__.py`).

---

## Public API

From `src/render/pytorch/__init__.py`:

- `render_gnn_to_pytorch(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`

**Contract**:
- writes exactly one Python file to `output_path`
- returns `(success, message, [output_path])` on success

---

## Implementation notes

`pytorch_renderer.py`:

- extracts \(A, B, C, D\) from common internal `gnn_spec` shapes
- validates shapes with `render.matrix_utils.validate_abcd_shapes`
- normalizes probability matrices with `render.matrix_utils.normalize_columns`
- generates a standalone script implementing an Active Inference style generative loop using `torch.tensor` operations

The generated script writes `simulation_results.json` under `PYTORCH_OUTPUT_DIR` (default: current directory).

---

## Integration points

- **Called by**: `render.pomdp_processor.POMDPRenderProcessor` when the `pytorch` framework is selected and the backend is importable.
- **Consumed by**: `src/execute/pytorch/` runner (Step 12) and `src/analysis/pytorch/` analyzer (Step 16).

---

## Testing

Preferred tests:

- formatting/syntax validation (`python -m py_compile` on the generated file)
- integration: Step 11 produces `pytorch/<model>_pytorch.py` when the backend is available

