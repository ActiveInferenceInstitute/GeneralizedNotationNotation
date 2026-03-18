# NumPyro Renderer

`src/render/numpyro/` renders a parsed GNN/POMDP model into a **standalone NumPyro** simulation script.

## Usage

```python
from pathlib import Path
from render.numpyro import render_gnn_to_numpyro

success, msg, artifacts = render_gnn_to_numpyro(
    gnn_spec=parsed_spec,
    output_path=Path("output/11_render_output/model/numpyro/model_numpyro.py"),
    options={"num_timesteps": 10},
)
```

## Output

The rendered artifact is a single `.py` file. When executed, it writes `simulation_results.json` under `NUMPYRO_OUTPUT_DIR` (defaults to `.`).

## Dependencies

- render-time: `numpy`
- generated script run-time: `jax`, `jaxlib`, `numpyro`, `numpy`

