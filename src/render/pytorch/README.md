# PyTorch Renderer

`src/render/pytorch/` renders a parsed GNN/POMDP model into a **standalone PyTorch** simulation script.

## Usage

```python
from pathlib import Path
from render.pytorch import render_gnn_to_pytorch

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

