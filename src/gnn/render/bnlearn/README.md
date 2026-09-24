# bnlearn Renderer

`src/gnn/render/bnlearn/` renders a parsed GNN/POMDP model into a **standalone bnlearn** script for Bayesian-network structure/parameter learning and exact (junction-tree) inference.

## Usage

```python
from gnn.render.bnlearn import generate_bnlearn_code

code = generate_bnlearn_code(
    model_data=parsed_spec,
    output_path="output/11_render_output/model/bnlearn/model_bnlearn.py",
)
```

## Output

The rendered artifact is a single `.py` file (`<model>_bnlearn.py` under `bnlearn/`). When executed, it builds a DAG, simulates categorical traces, fits CPTs with MLE, runs exact inference, and prints the analysis results to stdout; Step 12 runs it with `BNLEARN_OUTPUT_DIR` pointing at the model's `simulation_data` directory.

## Dependencies

- render-time: none beyond the core pipeline (the renderer is codegen-only and never imports `bnlearn`)
- generated script run-time: `bnlearn`, `pandas`, `numpy` (install with `uv sync --extra bnlearn`)
