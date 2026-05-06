# GNN → pymdp 1.0.0 Integration Guide

This guide describes the integration contract implemented in this repository
against the **JAX-first pymdp 1.0.0** release
([upstream](https://github.com/infer-actively/pymdp)).

## What This Repository Implements

- **Step 11 (render)** produces pymdp 1.0.0 runner scripts from parsed GNN
  model specs (`src/render/pymdp/pymdp_renderer.py`). Two templates ship:
    * **Pipeline runner** (default): a thin script that delegates to
      `src.execute.pymdp.run_simple_pymdp_simulation`.
    * **Standalone runner** (`options={"mode": "standalone"}`): a fully
      self-contained pymdp 1.0.0 script — no GNN pipeline on PYTHONPATH.
- **Step 12 (execute)** runs those scripts and stores JSON execution artifacts
  (`src/execute/pymdp/pymdp_runner.py`). The canonical rollout lives in
  `src/execute/pymdp/simple_simulation.py`, which calls real pymdp 1.0.0.
- **Step 16 (analysis)** reads execution artifacts from
  `output/12_execute_output/**/pymdp/simulation_data/simulation_results.json`
  and generates visualisations (`src/analysis/pymdp/`).

## Core Mapping

| GNN element              | pymdp 1.0.0 side                                   |
|--------------------------|----------------------------------------------------|
| hidden state variables   | state factors (one per `D[f]`, `B[f]`)             |
| observation variables    | observation modalities (one per `A[m]`, `C[m]`)    |
| control / action vars    | control factors (index listed in `control_fac_idx`)|
| `A`                      | observation likelihood `P(o│s)`                    |
| `B`                      | transition dynamics `P(s'│s, u)`                   |
| `C`                      | observation preferences (log prior over `o`)       |
| `D`                      | prior over initial states                          |
| `E` *(optional)*         | habit / policy prior                               |

## Matrix Shape Convention (pymdp 1.0.0)

Every tensor carries a **leading batch dimension** (size `batch_size`,
default 1) and is a `jax.Array` inside a `list[...]`:

| Symbol | Shape (single factor / modality, `batch_size=1`) |
|--------|--------------------------------------------------|
| `A[m]` | `(1, num_obs[m], num_states[0], num_states[1], …)` |
| `B[f]` | `(1, num_states[f], num_states[f], num_controls[f])` |
| `C[m]` | `(1, num_obs[m])`                                |
| `D[f]` | `(1, num_states[f])`                             |
| `E`    | `(1, num_policies)` (plain `Array`, not a list)  |

GNN files typically emit **unbatched numpy** matrices. The pipeline converts
them with `src/execute/pymdp/simple_simulation._to_jax_batched` — it just
prepends a batch axis and casts to `jnp.float32`.

### GNN B layout conversion

One older GNN convention stores `B` as `(action, prev_state, next_state)`.
pymdp 1.0.0 uses `(next_state, prev_state, action)`. Canonicalisation happens
inside `_canonicalise_B`:

```python
if b_raw.shape[0] == num_actions and b_raw.shape[1] == b_raw.shape[2]:
    b = b_raw.transpose(2, 1, 0)   # (next, prev, action)
else:
    b = b_raw                      # already (next, prev, action)
# + column-normalise each action slice
```

## Minimal Local Example (JAX-first)

```python
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from pymdp.agent import Agent

# 2 states, 2 observations, 2 actions
A = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
B = np.stack([np.eye(2), np.roll(np.eye(2), 1, axis=1)], axis=-1).astype(np.float32)
C = np.array([0.0, 1.0], dtype=np.float32)
D = np.array([0.5, 0.5], dtype=np.float32)

A_list = [jnp.asarray(A)[None, ...]]      # (1, 2, 2)
B_list = [jnp.asarray(B)[None, ...]]      # (1, 2, 2, 2)
C_list = [jnp.asarray(C)[None, ...]]      # (1, 2)
D_list = [jnp.asarray(D)[None, ...]]      # (1, 2)

agent = Agent(
    A=A_list, B=B_list, C=C_list, D=D_list,
    num_controls=[2],
    control_fac_idx=[0],
    policy_len=1,
    batch_size=1,
)

obs = [jnp.array([0], dtype=jnp.int32)]
qs, info = agent.infer_states(obs, empirical_prior=agent.D, return_info=True)
q_pi, neg_efe = agent.infer_policies(qs)

key = jr.PRNGKey(0)
action = agent.sample_action(q_pi, rng_key=jr.split(key, agent.batch_size + 1)[1:])
next_prior = agent.update_empirical_prior(action, qs)
```

For a **pure HMM** (only one action, e.g. a passive observer), drop
`control_fac_idx` entirely — pymdp 1.0.0 requires `num_controls[f] > 1` for any
factor listed in `control_fac_idx`.

## Pipeline artifact locations (render → execute → analysis)

Concrete directories (default `output/` base):

- **Step 11:** `11_render_output/<gnn_file_stem>/pymdp/*_pymdp.py` —
  generated runner. When executed directly, it writes
  `output/pymdp_simulations/<model_name>/simulation_results.json` (or the
  value of `$PYMDP_OUTPUT_DIR` if set).
- **Step 12:** `12_execute_output/<gnn_file_stem>/pymdp/simulation_data/simulation_results.json`
  — Step 12's `collect_execution_outputs` copies the runner's JSON into the
  canonical location; `execution_logs/` holds subprocess logs.
- **Step 16:** `16_analysis_output/pymdp/<model_slug>/` — plots from
  `analysis.pymdp.analyzer.generate_analysis_from_logs`.

Step 12 sets `GNN_PROJECT_ROOT` so the generated pipeline runner can
`import` from `src.execute.pymdp` regardless of render depth.

**Tests:**

```bash
uv run pytest \
    src/tests/test_pymdp_1_0_0_upstream_api.py \
    src/tests/test_pymdp_contracts.py \
    src/tests/test_execute_pymdp_integration.py \
    -v
```

## 1.0.0 Migration Notes Used Locally

- Import `jax.numpy as jnp` / `jax.random as jr` alongside pymdp.
- `utils.obj_array` is **gone** — use plain Python `list[jax.Array]` with a
  leading batch dim instead.
- `Agent.infer_states(obs)` no longer works; the new signature is
  `infer_states(observations, empirical_prior, *, return_info=False, ...)`.
  Returned `qs[f]` is shape `(batch, time, num_states[f])` — take the last
  time slice for current beliefs.
- `Agent.infer_policies()` now takes `qs`: `infer_policies(qs)` → `(q_pi, neg_efe)`
  where both are shape `(batch, num_policies)`.
- `Agent.sample_action()` now takes a JAX PRNG key slice:
  `sample_action(q_pi, rng_key=jr.split(key, batch_size + 1)[1:])` → shape
  `(batch, num_factors)`.
- Carry the empirical prior forward explicitly with
  `prior = agent.update_empirical_prior(action, qs)`.
- `Agent.reset()` is gone — re-seed via `empirical_prior=agent.D`.

## Basic examples

See [Minimal Local Example (JAX-first)](#minimal-local-example-jax-first) and pipeline tests under `src/tests/test_pymdp_*`.

## POMDP examples

POMDP-shaped models use the same `Agent` surface; see [Core Mapping](#core-mapping) and the doc hub [Example Gallery](../README.md#basic-examples).

## Multi-agent examples

Multi-factor and multi-control setups follow the factorial `A`/`B` list layout in [Matrix Shape Convention (pymdp 1.0.0)](#matrix-shape-convention-pymdp-100).

## Performance and Scaling

Systematic performance analysis is supported via the **Scaling Orchestrator**. This tool allows for empirical complexity sweeps across state-space dimensions (N) and time horizons (T).

- **Orchestrator**: [`scripts/run_pymdp_gnn_scaling_analysis.py`](../../scripts/run_pymdp_gnn_scaling_analysis.py)
- **Configuration**: [`scripts/pymdp_scaling_config.yaml`](../../scripts/pymdp_scaling_config.yaml)
- **Performance Guide**: See [pymdp_performance_guide.md](pymdp_performance_guide.md) for O(n³) scaling details and safety guardrails.

## Security considerations

Treat generated runners like any code that executes on your machine: use trusted GNN sources, avoid pasting secrets into model files, and review `output/` artifacts before sharing.

## What Is Not Claimed Here

- No wrapping of upstream `infer_and_plan` / `rollout` helper surfaces unless
  exercised in tests here.
- No local support for `pybefit` / NumPyro model-fitting extras unless wired
  into pipeline code **and** tests.
- No vmap/scan-based bulk rollouts via custom JAX transforms — the pipeline
  runs a plain Python loop over timesteps for readability and debuggability.
  The Agent's internal methods are still JAX-compiled.
