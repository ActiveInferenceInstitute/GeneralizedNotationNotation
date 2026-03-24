# PyMDP Execute Submodule

This module runs PyMDP simulations for Step 12 and writes execution artifacts.

## Separation Of Responsibilities

- Execute step (`src/execute/pymdp/`): runs simulations and writes raw results.
- Analysis step (`src/analysis/pymdp/`): generates visualizations and analysis
  reports from execution outputs.

## Public API

Exports are defined in `src/execute/pymdp/__init__.py`. Main entry points:

- `execute_pymdp_simulation_from_gnn(...)`
- `execute_pymdp_simulation(...)`
- `validate_pymdp_environment(...)`
- `get_pymdp_health_status(...)`
- package detection helpers:
  - `detect_pymdp_installation(...)`
  - `is_correct_pymdp_package(...)`
  - `validate_pymdp_for_execution(...)`

## Core Files

- `executor.py`: execution orchestration from GNN spec / rendered scripts.
- `pymdp_simulation.py`: simulation class and simulation run loop.
- `simple_simulation.py`: lightweight simulation entry path used by tests/tools.
- `pymdp_runner.py`: script execution utility and log capture.
- `validator.py`: environment checks and health reporting.
- `package_detector.py`: detect correct `inferactively-pymdp` installation.

## Dependency Guidance

- Recommended install for this repo:
  - `uv pip install inferactively-pymdp`
- Runtime import style:
  - `from pymdp.agent import Agent`

## Output Intent

Step 12 writes execution traces, logs, and summary artifacts. Plot generation is
explicitly deferred to Step 16 analysis.

## Upstream PyMDP ``Agent`` contract tests

The simulation loop in `simple_simulation.py` depends on `inferactively-pymdp`
(`Agent.infer_states`, `infer_policies`, `sample_action`, `utils.obj_array`,
`utils.is_normalized`). Regression coverage:

```bash
uv run pytest src/tests/test_pymdp_1_0_0_upstream_api.py -v
```
