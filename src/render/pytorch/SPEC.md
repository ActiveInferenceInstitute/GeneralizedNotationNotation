# PyTorch Render Backend Specification

## Overview

The PyTorch render backend generates **standalone PyTorch POMDP simulation scripts** from parsed GNN specifications.

## Public API

`render.pytorch` must export:

- `render_gnn_to_pytorch(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`

## Inputs

- `gnn_spec`: parsed dict containing enough information to infer \(A, B, C, D\).
- `output_path`: file path for the generated script.
- `options`: optional dict; currently used for settings such as `num_timesteps`.

## Outputs

- exactly one `.py` file at `output_path`.
- the generated script writes `simulation_results.json` under `PYTORCH_OUTPUT_DIR`.

## Dependencies

- render-time: `numpy`
- generated script run-time: `torch`, `numpy`

## Success criteria

Rendering is successful when:

- the script file is written to `output_path`, and
- the function returns `success=True` with `artifact_paths=[str(output_path)]`.

