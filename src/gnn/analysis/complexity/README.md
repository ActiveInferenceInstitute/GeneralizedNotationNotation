# Complexity Subpackage

Static complexity estimation for GNN models: per-backend bounds, complexity-class
labels, and a deterministic schema-versioned receipt — no framework imports, no
execution required.

## Public API

From `gnn.analysis` (re-exported) or `gnn.analysis.complexity`:

- `estimate_model_complexity(model_or_path)` — parse (or accept) a
  `GNNInternalRepresentation`, classify model kinds, and return the
  `gnn.complexity_estimate/v1` receipt dict: `model` (name/sha256/path),
  `structure` (variable/edge counts, factor arities, state-space dims, time block),
  `model_kinds`, `per_backend` (one bound row per registry backend, fixed
  `BACKEND_ORDER`, `applicable` flags), `estimator_version`.
- `to_json_text(receipt)` — stable JSON serialization (sorted keys, fixed indent).
- `ESTIMATOR_VERSION` — receipt schema version.

## Module structure

- `bounds.py` — `BACKEND_BOUNDS` / `BACKEND_ORDER` / `BackendBound`: per-backend
  formula registry (family, asymptotic, complexity class, drivers) keyed by the
  10 executor `_RUNNER_LOADERS` frameworks + bnlearn.
- `estimator.py` — structure extraction, kind gating, per-backend row emission.
- `AGENTS.md` — maintenance contract, receipt schema, honesty rules.

Estimates are labeled BOUNDS: a class label without a fabricated number whenever
the horizon is `Unbounded`; measured performance belongs to the benchmark harness
(wave 8 W8-CD), not to this subpackage.
