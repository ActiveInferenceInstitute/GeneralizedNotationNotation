# pymdp 1.0.0 Compatibility Matrix

This matrix maps upstream `infer-actively/pymdp` 1.0.0 release claims to the
integration points in this repository.

## Scope

- Upstream sources reviewed:
    - <https://github.com/infer-actively/pymdp>
    - <https://github.com/infer-actively/pymdp/releases/tag/v1.0.0>
- Local surfaces reviewed:
    - `src/render/pymdp/`
    - `src/execute/pymdp/`
    - `doc/pymdp/`
    - `src/tests/test_pymdp_*` / `src/tests/test_execute_pymdp_*`

## Matrix

| Upstream 1.0.0 item | Local status | Action |
|---|---|---|
| **JAX-first Agent** (`equinox.Module`-based) | **Fully integrated** | `src/execute/pymdp/simple_simulation._build_pymdp_agent` builds a real JAX-first Agent from GNN matrices. Exercised by `test_pymdp_1_0_0_upstream_api.py` and `test_pymdp_contracts.py`. |
| `Agent(A, B, C, D, E, num_controls, control_fac_idx, policy_len, batch_size, …)` | **Fully integrated** | Used verbatim by `_build_pymdp_agent`. Passive factors (num_controls[f] == 1) omit `control_fac_idx` as required by `Agent._validate`. |
| `infer_states(observations, empirical_prior, *, return_info=False)` → `qs [, info]` | **Fully integrated** | `simple_simulation.run_simple_pymdp_simulation` always calls with `empirical_prior=…, return_info=True` and extracts `info["vfe"]`. |
| `infer_policies(qs)` → `(q_pi, neg_efe)` | **Fully integrated** | The return tuple is unpacked as `(q_pi, neg_efe)` (upstream docstring calls the second value `G` = negative EFE per policy). |
| `sample_action(q_pi, rng_key=…)` with JAX PRNG keys | **Fully integrated** | Actions are drawn with `jr.split(key, batch_size + 1)[1:]`, matching the upstream quickstart. |
| `Agent.update_empirical_prior(action, qs)` — stateful rollout closure | **Fully integrated** | Called once per step in both `simple_simulation` and `PyMDPSimulation`. Presence of this method is how this repo detects pymdp 1.0.0+. |
| Batched list-of-array models (leading batch dim on A/B/C/D/E) | **Fully integrated** | `_to_jax_batched` prepends a batch axis of size 1 (or broadcasts for `batch_size>1`). |
| `utils.random_A_array` / `random_B_array` / `list_array_uniform` | **Available, not used for GNN models** | GNN-provided numeric matrices take precedence over random generators. These utils are asserted present by `test_utils_public_surface_exists`. |
| `utils.norm_dist` (JAX-array normalisation helper) | **Available (informational)** | Pipeline uses its own numpy-based `_normalise_columns` / `_normalise_prob_vector` because GNN matrices enter as numpy. |
| `utils.obj_array` (legacy 0.x helper) | **Removed upstream** | Gone from `pymdp.utils` in 1.0.0. Removed from this repo's code paths and docs. Any legacy references are in `pymdp.legacy` only. |
| `categorical_obs` naming standardisation | **Partial** | The pipeline always passes integer observation indices in 1-element `jnp.int32` arrays. `categorical_obs=True` mode (probability-vector observations) is supported by pymdp but not exercised in local tests. |
| Explicit PRNG key flow in JAX workflows | **Integrated for rollout** | `jax.random.PRNGKey` + `split` used for every `sample_action` call. Upstream learning / fitting flows (Dirichlet updates, JAX-native) are not locally wrapped. |
| `rollout()` / `infer_and_plan()` first-class APIs | **Not directly integrated** | The pipeline runs a plain Python loop for readability. Upstream helpers may be adopted later via a dedicated wrapper. |
| Legacy NumPy path preserved under `pymdp.legacy` | **Asserted importable** | `test_pymdp_legacy_namespace_still_shipped` probes for `import pymdp.legacy`; repository code does not depend on it. |
| Model fitting via `pybefit` / NumPyro workflows | **Not integrated** | Upstream capability; not wrapped in pipeline code or tests. |
| `equinox` and `multimethod` runtime deps | **Added to `pyproject.toml`** | Pinned alongside `inferactively-pymdp>=1.0.0`. |
| Tiered notebook testing gates | **Not mirrored** | Upstream reference only. |

## Local Contract Clarifications

- **Render-side public API**
  (`src/render/pymdp/__init__.py`): `render_gnn_to_pymdp(gnn_spec, output_path, options=…)`.
  `options={"mode": "pipeline"}` (default) emits a pipeline runner; `options={"mode": "standalone"}`
  emits a fully self-contained pymdp 1.0.0 script.

- **Execute-side public API**
  (`src/execute/pymdp/__init__.py`): `execute_pymdp_simulation(gnn_spec, output_dir, correlation_id)` — canonical entry.
  Also exposes `PyMDPSimulation` (the GNN-driven wrapper class).

- **Visualisation** belongs to Step 16 (`src/analysis/pymdp/`), not Step 12.

## Locally regression-tested Agent API

Run:

```bash
uv run pytest \
    src/tests/test_pymdp_1_0_0_upstream_api.py \
    src/tests/test_pymdp_contracts.py \
    src/tests/test_execute_pymdp_integration.py \
    -v
```

Covered surfaces (real package, no mocks):

- `importlib.metadata.version("inferactively-pymdp") >= 1.0.0`
- `from pymdp.agent import Agent` / `from pymdp import utils`
- Presence of `Agent.update_empirical_prior` (the 1.0.0 signal)
- `utils.random_A_array`, `random_B_array`, `list_array_uniform`,
  `norm_dist`, `list_array_norm_dist`
- `Agent.infer_states(obs, empirical_prior=…, return_info=True)` return shapes
- `Agent.infer_policies(qs)` return tuple `(q_pi, neg_efe)`
- `Agent.sample_action(q_pi, rng_key=…)` batched action shape
- Multi-step rollout closed via `Agent.update_empirical_prior`
- Optional `E` habit vector length aligned with the policy count
- Import of `pymdp.control` and `pymdp.inference` (used internally by Agent)
- Import of `pymdp.legacy` (namespace probe; not consumed)

Pipeline-level integration is covered by
`src/tests/test_pymdp_contracts.py::test_actinf_pomdp_render_execute_analyze_e2e`,
which renders real GNN POMDP input → runs pymdp 1.0.0 via Step 12 → collects
the JSON result → analyses via Step 16.

## Validation Policy Used For This Alignment

- If a behaviour is not implemented and tested locally, docs label it as
  upstream context rather than a local guarantee.
- Examples in local docs use the canonical pymdp 1.0.0 import style:
  - `import jax.numpy as jnp`
  - `import jax.random as jr`
  - `from pymdp.agent import Agent`
  - `from pymdp import utils as pymdp_utils`
- Installation guidance uses the `inferactively-pymdp` PyPI name and pins
  the version floor to `1.0.0`.
