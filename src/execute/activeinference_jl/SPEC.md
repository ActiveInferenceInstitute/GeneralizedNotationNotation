# ActiveInference.jl Execution — Technical Specification

**Version**: 1.6.0

## Execution Model

- Julia subprocess execution via `julia` command
- Pre-flight: Julia availability check + package validation
- Timeout: inherits from Step 12 timeout (1800s default)

## Input

- Julia scripts from `output/11_render_output/activeinference_jl/`

## Output

- `simulation_results.json` — Serialized Julia inference results
- Execution logs (stdout/stderr)
- Timing and resource usage data

## Dependencies

- `julia >= 1.8` with `ActiveInference.jl` package installed
