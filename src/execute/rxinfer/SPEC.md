# RxInfer Execution — Technical Specification

**Version**: 1.6.0

## Execution Model

- Julia subprocess execution via `julia` command
- Pre-flight: Julia + RxInfer package validation
- Reactive message-passing inference execution
- Timeout: inherits from Step 12 timeout (1800s default)

## Input

- Julia/TOML scripts from `output/11_render_output/rxinfer/`

## Output

- `simulation_results.json` — Message-passing inference results
- Execution logs (stdout/stderr)
- Convergence diagnostics

## Dependencies

- `julia >= 1.8` with `RxInfer.jl` package
