# NumPyro Render Backend Specification

## Overview

The NumPyro render backend generates **standalone NumPyro POMDP simulation scripts** from parsed GNN specifications.

## Public API

`render.numpyro` must export:

- `render_gnn_to_numpyro(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`

## Inputs

- `gnn_spec`: parsed dict containing at least enough information to infer \(A, B, C, D\). The implementation supports common internal shapes (`stateSpace.parameters`, `initialparameterization`, `parameters`) and applies defaults otherwise.
- `output_path`: file path for the generated script.
- `options`: optional dict; currently used for settings such as `num_timesteps`.

## Outputs

- exactly one `.py` file at `output_path`.
- the generated script writes `simulation_results.json` under `NUMPYRO_OUTPUT_DIR`.

## Dependencies

- render-time: `numpy`
- generated script run-time: `jax`, `jaxlib`, `numpyro`, `numpy`

## Success criteria

Rendering is considered successful when:

- the script file is written to `output_path`, and
- the function returns `success=True` with `artifact_paths=[str(output_path)]`.

