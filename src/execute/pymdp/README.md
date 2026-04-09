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

- Required install:
  ```bash
  uv pip install 'inferactively-pymdp>=1.0.0'
  ```
  pymdp 1.0.0 is the JAX-first rewrite (`equinox.Module` Agent, batched
  list-of-JAX-array models, explicit PRNG keys for action sampling).
- Runtime import style:
  ```python
  import jax.numpy as jnp
  import jax.random as jr
  from pymdp.agent import Agent
  from pymdp import utils as pymdp_utils
  ```

## Output Intent

Step 12 writes execution traces, logs, and summary artifacts. Plot generation is
explicitly deferred to Step 16 analysis.

## Upstream PyMDP ``Agent`` contract tests

The simulation loop in `simple_simulation.py` calls pymdp 1.0.0's
JAX-first `Agent` with the canonical rollout pattern:

```python
qs, info = agent.infer_states(obs, empirical_prior=prior, return_info=True)
q_pi, neg_efe = agent.infer_policies(qs)
action = agent.sample_action(q_pi, rng_key=jr.split(key, agent.batch_size + 1)[1:])
prior = agent.update_empirical_prior(action, qs)
```

Regression coverage:

```bash
uv run pytest \
    src/tests/test_pymdp_1_0_0_upstream_api.py \
    src/tests/test_pymdp_contracts.py \
    src/tests/test_execute_pymdp_integration.py \
    -v
```
