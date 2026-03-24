# GNN To PyMDP Integration Guide

This guide describes the integration contract implemented in this repository.

## What This Repository Implements

- Step 11 render writes PyMDP-targeted scripts from parsed GNN model specs.
- Step 12 execute runs those scripts and stores execution artifacts.
- Step 16 analysis reads execution artifacts and generates visual outputs.

## Core Mapping

| GNN element | PyMDP side |
|---|---|
| hidden state variables | state factors |
| observation variables | observation modalities |
| control/action variables | control factors |
| `A` | observation likelihood `P(o|s)` |
| `B` | transition dynamics `P(s'|s,u)` |
| `C` | observation preferences |
| `D` | prior over initial states |

## Minimal Local Example

```python
import numpy as np
from pymdp import utils
from pymdp.agent import Agent

A = utils.obj_array(1)
B = utils.obj_array(1)
C = utils.obj_array(1)
D = utils.obj_array(1)

A[0] = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)
B[0] = np.zeros((2, 2, 1), dtype=float)
B[0][:, :, 0] = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=float)
C[0] = np.array([0.0, 1.0], dtype=float)
D[0] = np.array([0.5, 0.5], dtype=float)

agent = Agent(A=A, B=B, C=C, D=D)
obs = [np.array([0])]
qs = agent.infer_states(obs)
q_pi, neg_efe = agent.infer_policies()
action = agent.sample_action()
```

## Pipeline artifact locations (render → execute → analysis)

Concrete directories (default `output/` base):

- **Step 11:** `11_render_output/<gnn_file_stem>/pymdp/*_pymdp.py` — generated runner; writes under that folder’s `output/pymdp_simulations/<model_name>/simulation_results.json` when executed.
- **Step 12:** `12_execute_output/<gnn_file_stem>/pymdp/simulation_data/simulation_results.json` — after `collect_execution_outputs` copies JSON from the render tree; `execution_logs/` holds subprocess logs.
- **Step 16:** `16_analysis_output/pymdp/<model_slug>/` — plots from `analysis.pymdp.analyzer.generate_analysis_from_logs`.

Step 12 sets `GNN_PROJECT_ROOT` so generated scripts can `import` from `src.execute.pymdp` regardless of render path depth.

**Tests:** `uv run pytest src/tests/test_pymdp_1_0_0_upstream_api.py src/tests/test_pymdp_contracts.py -v`

## 1.0.0 Migration Notes Used Locally

- Prefer `neg_efe` naming over older `G` naming in examples.
- Use explicit tuple unpacking for `infer_policies()`.
- Keep observation naming compatible with current upstream docs (`categorical_obs`
  terminology where relevant).

## What Is Not Claimed Here

- Full local wrapping of all JAX-first APIs (`rollout`, `infer_and_plan`,
  full PRNG-key workflow) unless implemented and tested in this repository.
- Full local support for upstream model-fitting extras (`pybefit`/NumPyro)
  unless explicitly wired into pipeline code and tests.
