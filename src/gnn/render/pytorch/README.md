# PyTorch Renderer

`src/gnn/render/pytorch/` renders a parsed GNN/POMDP model into a **standalone PyTorch** simulation script.

## Usage

```python
from pathlib import Path
from gnn.render.pytorch import render_gnn_to_pytorch

success, msg, artifacts = render_gnn_to_pytorch(
    gnn_spec=parsed_spec,
    output_path=Path("output/11_render_output/model/pytorch/model_pytorch.py"),
    options={"num_timesteps": 10},
)
```

## Output

The rendered artifact is a single `.py` file. When executed, it writes `simulation_results.json` under `PYTORCH_OUTPUT_DIR` (defaults to `.`).

## Dependencies

- render-time: `numpy`
- generated script run-time: `torch`, `numpy`


## Discrete filtering contract

The first observation conditions the authored initial prior. Each later prior
is the previous posterior propagated by the previous selected action's transition
matrix, before conditioning on the new observation. The recorded beliefs are
posteriors. Zero or nonfinite predictive evidence raises an error; invalid result
validation is persisted and then raises an error, so an invalid trace cannot
complete successfully. This is categorical filtering and action scoring, not a
claim of equivalence to other backends' policy objectives.
