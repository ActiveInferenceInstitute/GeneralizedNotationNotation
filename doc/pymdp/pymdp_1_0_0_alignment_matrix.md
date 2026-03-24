# PyMDP 1.0.0 Compatibility Matrix

This matrix maps upstream `infer-actively/pymdp` 1.0.0 release claims to
current integration points in this repository.

## Scope

- Upstream sources reviewed:
  - <https://github.com/infer-actively/pymdp>
  - <https://github.com/infer-actively/pymdp/releases/tag/v1.0.0>
- Local surfaces reviewed:
  - `src/render/pymdp/`
  - `src/execute/pymdp/`
  - `doc/pymdp/`
  - related tests in `src/tests/`

## Matrix

| Upstream 1.0.0 item | Local status | Action |
|---|---|---|
| JAX-first positioning | Partial | Keep this repo's execution path framework-agnostic; avoid claiming full JAX-first API parity in PyMDP docs unless tested locally. |
| `infer_policies()` returns `(q_pi, neg_efe)` | Tested locally | Upstream names the second return `G` (negative EFE per policy). Code uses `neg_efe`; see `src/tests/test_pymdp_1_0_0_upstream_api.py`. |
| `categorical_obs` naming standardization | Partial | Remove legacy-only wording and ensure examples use current observation naming conventions. |
| Explicit PRNG key flow in JAX workflows | Not directly implemented | Document as upstream behavior; avoid implying this repository wraps full JAX-native runtime semantics for PyMDP. |
| `rollout()` / `infer_and_plan()` first-class APIs | Not directly integrated | Do not claim support via local wrapper modules unless exercised in local code/tests. |
| Legacy NumPy path preserved under `pymdp.legacy` | Informational | Mention in docs as migration context only, not as a local integration contract. |
| Model fitting via `pybefit`/NumPyro workflows | Not directly integrated | Document as optional upstream capability; do not present as locally validated pipeline feature. |
| Tiered notebook testing gates | Not locally mirrored | Keep as upstream reference only. |

## Local Contract Clarifications

- Render-side public API is the package export in `src/render/pymdp/__init__.py`:
  - `render_gnn_to_pymdp(...)`
- Execute-side public API is the package export list in `src/execute/pymdp/__init__.py`.
- Local execute path supports real package detection and guarded fallback behavior.
- Visualization belongs to analysis step (`src/analysis`), not execute step.

## Locally regression-tested Agent API (inferactively-pymdp)

Run:

```bash
uv run pytest src/tests/test_pymdp_1_0_0_upstream_api.py -v
```

Covered surfaces (real package, no mocks):

- `importlib.metadata.version("inferactively-pymdp")` readable
- `from pymdp.agent import Agent` / `from pymdp import utils`
- `utils.obj_array`, `utils.is_normalized` on A/B payloads matching the pipeline
- `Agent.reset`, `Agent.infer_states`, `Agent.infer_policies`, `Agent.sample_action` (single- and multi-step)
- Optional `E` habit vector length aligned with policy count
- `import pymdp.control`, `import pymdp.inference` (modules used internally by `Agent`)
- `pymdp.legacy`: import attempted; skipped if absent in the installed wheel

Pipeline integration beyond the raw Agent API remains covered by
`src/tests/test_pymdp_contracts.py` and `src/tests/test_execute_pymdp_*.py`.

The PyPI distribution version (e.g. from `importlib.metadata.version("inferactively-pymdp")`)
may read below `1.0.0` while still exposing the same `Agent` methods; the tests above
validate behavior, not the marketing version number.

## Validation Policy Used For This Alignment

- If a behavior is not implemented and tested locally, docs label it as upstream
  context rather than local guarantee.
- Examples in local docs use import style consistent with this repository:
  - `from pymdp.agent import Agent`
- Installation guidance uses `inferactively-pymdp` naming.
