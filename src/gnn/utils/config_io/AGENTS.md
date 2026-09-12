# `gnn/utils/config_io/` — agent contract

Concern-package home (S2-33 split, Step 7 family 2/3; design:
`docs/development/utils_split_design.md`). The old top-level module paths
under `gnn/utils/` remain as DeprecationWarning facades for the deprecation
window — new code imports from this package.

## Invariants

- `_EXPORT_MAP` keys in `gnn/utils/__init__.py` are frozen (113 keys,
  golden-asserted by `tests/tests/test_infrastructure_exports.py`); this
  package's leaf modules are the *values* for the 14 `config_loader` names.
- Eager re-exports are safe here (§4.3.1): all four leaves are stdlib-only at
  module scope — the one third-party import (PyYAML in `config_loader`) is
  guarded by `try/except ImportError` and degrades to a
  `YAML_AVAILABLE = False` sentinel. Importing `gnn.utils` never touches this
  package (`tests/tests/test_light_import.py` is the gate).
- Cross-family imports go through leaf modules, never through any facade.
- Zero behavior changes: this package is a mechanical reorganization; the
  per-module contracts live in each leaf's docstring.

## Tests

`tests/tests/test_infrastructure_exports.py` (frozen surface),
`tests/tests/test_light_import.py` (lazy top-level facade), and the per-module
suites `tests/utils/test_io_utils.py`, `tests/utils/test_code_metrics.py`,
`tests/utils/test_utils_core.py`, `tests/utils/test_shared_helpers.py`.
