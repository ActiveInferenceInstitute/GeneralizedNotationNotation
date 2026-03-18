# PyTorch Executor

`src/execute/pytorch/` discovers and executes PyTorch scripts produced by Step 11 render output.

## Key functions

- `run_pytorch_scripts(rendered_simulators_dir, execution_output_dir=None, recursive_search=True, verbose=False, device=None) -> bool`

## Output

For each executed script (when `execution_output_dir` is provided), the runner writes:

- `stdout.txt`
- `stderr.txt`
- `execution_log.json`

and the executed script writes `simulation_results.json` under `PYTORCH_OUTPUT_DIR`.

