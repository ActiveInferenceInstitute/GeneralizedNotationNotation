# PyMDP Render Submodule

This module renders parsed GNN model specs into executable PyMDP-oriented
Python scripts for Step 11 outputs.

## Public API

Exported from `src/render/pymdp/__init__.py`:

- `render_gnn_to_pymdp(gnn_spec, output_path, options=None) -> (success, message, warnings)`

No additional public function is exported by this package.

## Key Implementation

- `pymdp_renderer.py`
  - `PyMDPRenderer.render_spec(...)`
  - `PyMDPRenderer.render_file(...)`
  - `PyMDPRenderer.render_directory(...)`
  - wrapper `render_gnn_to_pymdp(...)`

## Contract

- Input is a parsed GNN dictionary expected by the render pipeline.
- Output is a Python script file at the target `output_path`.
- Return tuple structure is:
  - `success: bool`
  - `message: str`
  - `warnings: list[str]`

## Integration

- Called by the Step 11 render pipeline (`src/render/processor.py`).
- Generated scripts are consumed by Step 12 execution (`src/execute/processor.py`).
- Visualization is not handled in render.

### Step 11 → Step 12 → Step 16 layout (PyMDP)

| Step | Typical paths |
|------|----------------|
| 11 Render | `output/11_render_output/<gnn_stem>/pymdp/<ModelName>_pymdp.py` |
| 12 Execute | `output/12_execute_output/<gnn_stem>/pymdp/simulation_data/simulation_results.json` (after collection from `.../pymdp/output/pymdp_simulations/...`) |
| 16 Analysis | `output/16_analysis_output/pymdp/<model_slug>/` (plots from `generate_analysis_from_logs`) |

Generated scripts resolve the repo root via `GNN_PROJECT_ROOT` (set by Step 12 when running subprocesses) or by walking upward to a directory containing `pyproject.toml` and `src/`.

## Dependency Notes

- Runtime scripts target the PyMDP package imported as:
  - `from pymdp.agent import Agent`
- Installation guidance in this repository uses:
  - `uv pip install inferactively-pymdp`

## Verification

Relevant tests live in `src/tests/`, including render+execute integration and
PyMDP simulation pathway tests. Upstream ``Agent`` / ``utils`` methods used by
generated runners are also asserted in
`src/tests/test_pymdp_1_0_0_upstream_api.py`.
