# NumPyro Execution — Technical Specification

**Version**: 3.2.0

## Execution Model

- Subprocess execution via `python -c` or script file
- Pre-flight: syntax check + dependency validation
- Timeout: inherits from Step 12 timeout (default 1800s)

## Input

- NumPyro scripts from `output/11_render_output/numpyro/`

## Output

- `simulation_results.json` — MCMC samples and inference results
- Execution logs (stdout/stderr)
- Timing data

## Dependencies

- `numpyro >= 0.12.0`, `jax >= 0.4.0`
