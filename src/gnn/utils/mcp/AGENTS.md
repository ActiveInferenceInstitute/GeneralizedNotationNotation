# `gnn/utils/mcp/` — agent contract

Concern-package home (S2-33 split; design: `docs/development/utils_split_design.md`).
The old top-level module paths under `gnn/utils/` remain as DeprecationWarning
facades for the deprecation window — new code imports from this package.

## Invariants

- `_EXPORT_MAP` keys in `gnn/utils/__init__.py` are frozen (113 keys,
  golden-asserted by `tests/tests/test_infrastructure_exports.py`); this
  package's leaf modules are the *values*.
- No import-time side effects at package level: leaves load on first use
  (`tests/tests/test_light_import.py` is the gate).
- Cross-family imports go through leaf modules, never through any facade.
- Zero behavior changes: this package is a mechanical reorganization; the
  per-module contracts live in each leaf's docstring.

## Tests

`tests/tests/test_infrastructure_exports.py` (frozen surface),
`tests/tests/test_light_import.py` (lazy facade), and the per-family suites
under `tests/`.
