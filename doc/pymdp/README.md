# PyMDP Documentation (pymdp 1.0.0 / JAX-first)

**Signposts:** [AGENTS.md](AGENTS.md) · [doc/INDEX.md](../INDEX.md) · [doc/SPEC.md](../SPEC.md) (versioning)

This folder documents how this repository integrates
[pymdp 1.0.0](https://github.com/infer-actively/pymdp) in the render/execute/
analysis pipeline (Steps 11, 12, 16).

pymdp 1.0.0 is the **JAX-first rewrite** of the library. The legacy NumPy API
(`utils.obj_array`, `Agent.infer_states(obs)` with no `empirical_prior`, etc.)
is only reachable through the `pymdp.legacy` namespace, which this repository
does **not** consume.

## Start Here

- [`gnn_pymdp.md`](gnn_pymdp.md) — GNN → pymdp 1.0.0 integration contract
  (matrix shapes, rollout loop, embedded code samples).
- [`pymdp_performance_guide.md`](pymdp_performance_guide.md) — performance
  notes (JAX/JIT, batching, memory) and **Systematic Scaling Studies**.
- [`run_pymdp_gnn_scaling_analysis.py`](../../scripts/run_pymdp_gnn_scaling_analysis.py) —
  The automated orchestrator for PyMDP performance sweeps.
- [`pymdp_1_0_0_alignment_matrix.md`](pymdp_1_0_0_alignment_matrix.md) —
  upstream 1.0.0 claim mapping and local status.
- [`pymdp_pomdp/README.md`](pymdp_pomdp/README.md) — reference scripts under
  `doc/pymdp/pymdp_pomdp/` and their boundaries.

## Version Scope

Local docs track the repository-tested behaviour of
`inferactively-pymdp>=1.0.0` (pinned in `pyproject.toml`).

- <https://github.com/infer-actively/pymdp>
- <https://github.com/infer-actively/pymdp/releases/tag/v1.0.0>

Upstream features that this repository does not wire in (e.g. `infer_and_plan`,
full JAX PRNG-key learning loops, `pybefit` model-fitting) are documented as
upstream context only.

## Local Conventions

- Install: `uv pip install 'inferactively-pymdp>=1.0.0'`
- Import style:
  ```python
  import jax.numpy as jnp
  import jax.random as jr
  from pymdp.agent import Agent
  from pymdp import utils as pymdp_utils
  ```
- Agent construction uses **list-of-`jax.Array`** matrices with a leading
  batch dimension (`batch_size=1` by default):
  - `A[m].shape == (batch, num_obs[m], num_states...)`
  - `B[f].shape == (batch, num_states[f], num_states[f], num_controls[f])`
  - `C[m].shape == (batch, num_obs[m])`
  - `D[f].shape == (batch, num_states[f])`
  - `E.shape    == (batch, num_policies)` (optional)
- Rollout loop uses the 1.0.0 canonical pattern:
  ```python
  prior = agent.D
  for t in range(T):
      qs, info  = agent.infer_states(obs, empirical_prior=prior, return_info=True)
      q_pi, neg_efe = agent.infer_policies(qs)
      action    = agent.sample_action(q_pi, rng_key=jr.split(key, agent.batch_size+1)[1:])
      prior     = agent.update_empirical_prior(action, qs)
  ```
- A pure HMM (1-action / passive) must **omit** `control_fac_idx`; the 1.0.0
  Agent asserts that every indexed control factor has `num_controls[f] > 1`.

## Upstream Agent API regression tests

- `src/tests/test_pymdp_1_0_0_upstream_api.py` — asserts the installed wheel's
  `Agent` / `utils` behaviour used by `execute/pymdp/simple_simulation.py`
  (no mocks). Covers:
    - Metadata version ≥ 1.0.0
    - `Agent.update_empirical_prior` presence
    - Single-step `infer_states` → `infer_policies` → `sample_action`
    - `empirical_prior` carry-through via `update_empirical_prior`
    - Optional `E` habit vector matches policy count
- `src/tests/test_pymdp_contracts.py` — exercises the full GNN
  render → execute → analysis path against real pymdp 1.0.0.
- `src/tests/test_execute_pymdp_integration.py` — exercises the JAX-first
  Agent directly and the pipeline's `run_simple_pymdp_simulation`.
