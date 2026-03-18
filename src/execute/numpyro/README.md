# NumPyro Executor

`src/execute/numpyro/` discovers and executes NumPyro scripts produced by Step 11 render output.

## Key functions

- `run_numpyro_scripts(rendered_simulators_dir, execution_output_dir=None, recursive_search=True, verbose=False) -> bool`

## Output

For each executed script (when `execution_output_dir` is provided), the runner writes:

- `stdout.txt`
- `stderr.txt`
- `execution_log.json`

and the executed script writes `simulation_results.json` under `NUMPYRO_OUTPUT_DIR`.

