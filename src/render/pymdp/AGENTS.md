# PyMDP Render Submodule Agent Guide

## Purpose

Translate GNN model specs into executable Python scripts for the PyMDP
execution path used by the pipeline.

## Canonical Public Surface

From `src/render/pymdp/__init__.py`:

- `render_gnn_to_pymdp(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`

If a function is not exported there, it should not be documented as public API.

## Internal Components

- `pymdp_renderer.py`
  - `parse_gnn_markdown(...)`
  - `PyMDPRenderer` class:
    - `render_file(...)`
    - `render_spec(...)`
    - `render_directory(...)`
    - `_generate_pymdp_simulation_code(...)`
    - `_get_timestamp(...)`

## Data Flow

1. Receive parsed or file-backed GNN spec.
2. Build script text with embedded state-space parameters.
3. Write script into Step 11 output structure.
4. Return `(success, message, warnings)` for pipeline orchestration.

## Integration Boundaries

- Upstream: called by render processor (`src/render/processor.py`).
- Downstream: scripts are executed by `src/execute/pymdp/`.
- This module does not run simulations and does not generate analysis plots.

## Documentation Rules For This Folder

- Keep signatures synchronized with real exports and real function parameters.
- Avoid documenting speculative helper APIs.
- Keep package naming consistent with repository setup:
  - install: `inferactively-pymdp`
  - import: `from pymdp.agent import Agent`

## Tests

Primary validation is in `src/tests/` integration coverage for render and
render→execute handoff. The generated script calls `execute_pymdp_simulation` /
`run_simple_pymdp_simulation`, whose upstream `Agent` API is locked by
`src/tests/test_pymdp_1_0_0_upstream_api.py`.
