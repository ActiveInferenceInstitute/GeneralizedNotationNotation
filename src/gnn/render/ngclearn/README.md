# ngc-learn Renderer

`src/gnn/render/ngclearn/` renders a parsed GNN/POMDP model into a **standalone ngc-learn** continuous linear-Gaussian (LGSSM) simulation script — the 4th Python backend of the shared `gnn.render.continuous_script` generator (KF numerics byte-identical to the jax backend). Discrete POMDP kinds are first-class unsupported (registry `continuous_only=True`).

## Usage

```python
from pathlib import Path
from gnn.render.ngclearn import render_gnn_to_ngclearn

success, msg, artifacts = render_gnn_to_ngclearn(
    gnn_spec=parsed_spec,
    output_path=Path("output/11_render_output/model/ngclearn/model_ngclearn.py"),
    options=None,
)
```

## Output

The rendered artifact is a single `.py` file. When executed, it writes `simulation_results.json` under `NGCLEARN_OUTPUT_DIR` (defaults to `.`).

## Dependencies

- render-time: none beyond the core pipeline (the renderer is codegen-only and never imports `ngclearn`)
- generated script run-time: `ngclearn` >=3.2.2, `ngcsimlib` >=3.1.1, `jax` >=0.11.1 (install with `uv sync --extra ngclearn`; the extra resolves on Python >=3.12 — `jax` 0.11.x publishes only for py3.12+)
