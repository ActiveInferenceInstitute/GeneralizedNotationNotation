# `gnn/utils/system_env/` — agent contract

Concern-package home (S2-33 Step 7, family 1/3; design:
[`docs/development/utils_split_design.md`](../../../../docs/development/utils_split_design.md)).
The old top-level module paths under `gnn/utils/` remain as DeprecationWarning
facades for the deprecation window — new code imports from this package.

## Invariants

- `_EXPORT_MAP` keys in `gnn/utils/__init__.py` are frozen (113 keys,
  golden-asserted by `tests/tests/test_infrastructure_exports.py`); this
  package's leaf modules are the *values* for the `get_system_info` and
  `get_venv_python` keys. `matplotlib_setup` has no facade key — it was never
  exported through the `gnn.utils` facade; its consumers import the leaf
  directly.
- The family `__init__` re-exports public names eagerly as real objects
  (design §4.3.1), but `import gnn.utils` must stay light — the top-level
  facade never imports this package statically
  (`tests/tests/test_light_import.py` is the gate).
- Optional heavy dependencies stay guarded at leaf scope: `system_utils`
  probes psutil in a try/except and publishes `PSUTIL_AVAILABLE`;
  `matplotlib_setup` imports matplotlib only inside
  `apply_env_backend_if_set`. Preserve both probes exactly when editing
  leaves.
- Cross-family imports go through leaf modules, never through any facade (I5).
- Zero behavior changes: this package is a mechanical reorganization; the
  per-module contracts live in each leaf's docstring.

## Tests

`tests/tests/test_infrastructure_exports.py` (frozen surface),
`tests/tests/test_light_import.py` (lazy facade), and the per-leaf pins
`tests/utils/test_system_utils.py` / `tests/utils/test_venv_utils.py`.