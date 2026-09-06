# gnn.execute.lean

Lean 4 execution backend for the GNN pipeline: verifies emitted documents
against the `FEP.GnnDocument` typed surface owned by the sibling `fep_lean`
checkout (bridge contract v0.5).

## What this module does

`GNNExecutor` registers `lean` as the ninth execution family. The runner
resolves the fep_lean checkout (`FEP_LEAN_ROOT` env overrides the
`../fep_lean` default), discovers `*.lean` files and emitted GNN `*.md`
documents under the rendered target directory, and drives
`fep-lean bridge verify-document` for each — writing per-document receipts
into the framework output directory.

## Public API

- `FEP_LEAN_ROOT_ENV` — env var naming the fep_lean checkout.
- `resolve_fep_lean_root()` — locate + validate the checkout (or `None`).
- `lean_toolchain_available()` — availability probe used by the executor.
- `verify_document(document, receipt=None, *, model, fail_on_warnings,
  gnn_root, timeout)` — verify one document; returns a record with
  `success`, the parsed receipt, or a fail-closed `error`.
- `run_lean_scripts(rendered_simulators_dir, execution_output_dir=None,
  recursive_search=True, verbose=False)` — per-framework runner (same
  signature contract as the other executors); `True` only when every
  document verifies (or there is nothing to verify).

## Scope boundary

`verify-document` proves syntax and `WellFormed` only. It does NOT
auto-prove `DiscreteConforms` / `ContinuousConforms` — that is
research-grade and out of scope (bridge contract v0.5 §13). When the
fep_lean checkout is unavailable the backend is reported skipped, never
silently successful.
