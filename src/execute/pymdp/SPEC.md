# PyMDP Execution — Technical Specification

**Version**: 1.6.0

## Execution Model

- Primary engine: `ActiveInferenceAgent` for full-fidelity POMDP execution
- Fallback: `FallbackAgent` for robust degraded execution (non-mock)
- Package detection: auto-detects `pymdp` version (0.x vs 1.0.0+ API)
- Timeout: inherits from Step 12 timeout (1800s default)

## Architecture

```
pymdp/
├── __init__.py             # Package exports
├── context.py              # Execution context management
├── execute_pymdp.py        # Entry point dispatcher
├── executor.py             # High-level execution coordinator
├── package_detector.py     # PyMDP version detection
├── pymdp_runner.py         # Core simulation runner
├── pymdp_simulation.py     # Full POMDP simulation engine
├── pymdp_utils.py          # Shared utilities
├── simple_simulation.py    # Simplified simulation fallback
└── validator.py            # Input validation
```

## Input

- PyMDP scripts from `output/11_render_output/pymdp/`

## Output

- `simulation_results.json` — Belief trajectories, action sequences, EFE values
- Execution logs and timing data

## Dependencies

- `pymdp >= 0.0.7` or `pymdp >= 1.0.0` (detected automatically)
