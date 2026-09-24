# bnlearn Render Backend - Agent Scaffolding

## Module Overview

**Purpose**: Render parsed GNN/POMDP specifications to **standalone bnlearn scripts** (Bayesian-network structure/parameter learning plus exact inference).

**Parent**: `src/gnn/render/` (Step 11: Render)

**Primary entrypoint**: `generate_bnlearn_code` in `bnlearn_renderer.py` (re-exported by `__init__.py`).

---

## Public API

From `src/gnn/render/bnlearn/__init__.py`:

- `generate_bnlearn_code(model_data: Dict[str, Any], output_path: Optional[Union[str, Path]] = None) -> str`

**Contract**:
- returns the generated program text; when `output_path` is given the program is written there and also returned
- validation failure returns `""`; generation errors are logged and re-raised

---

## Implementation notes

`bnlearn_renderer.py` is codegen-only: the renderer never imports `bnlearn` — the generated script does (`import bnlearn as bn` + `bn.make_DAG` + `bn.parameter_learning.fit` + `bn.inference.fit`). Validation and literal-formatting helpers are shared with the other codegen generators via `render.generators`.

---

## Integration points

- **Called by**: `render.processor` (generator-target dispatch for `target="bnlearn"`) and the basic per-framework generation path in `_process_single_gnn_file_basic`.
- **Consumed by**: the Step-12 bnlearn script path (`src/gnn/execute/bnlearn/`, `BNLEARN_OUTPUT_DIR`). Scripts skip without the `bnlearn` extra (`uv sync --extra bnlearn`).

---

## Testing

Preferred tests:

- formatting/syntax validation: the generated script compiles (`py_compile`) — no bnlearn import happens at render or test time
- integration: Step 11 produces `bnlearn/<model>_bnlearn.py` for graph-backed specs

End-to-end execution tests should live under `tests/` and be resilient to optional dependency availability.
