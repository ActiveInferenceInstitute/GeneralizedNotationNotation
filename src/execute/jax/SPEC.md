# JAX Execution — Technical Specification

**Version**: 1.6.0

## Execution Model

- Python subprocess execution
- Pre-flight: JAX availability + device check (CPU/GPU/TPU)
- JIT compilation caching for repeated runs
- Timeout: inherits from Step 12 timeout (1800s default)

## Input

- JAX scripts from `output/11_render_output/jax/`

## Output

- `simulation_results.json` — Inference results and computation traces
- Execution logs (stdout/stderr)
- Device utilization metrics

## Dependencies

- `jax >= 0.4.0`, `jaxlib` (with optional GPU support)
