# AGENTS.md — `gnn.execute.lean`

Lean 4 execution backend: document verification against the `FEP.GnnDocument`
typed surface owned by the sibling `fep_lean` checkout (bridge contract v0.5,
`docs/design/gnn-bridge/bridge-contract.md` in that repo).

## Files

- `lean_runner.py` — `resolve_fep_lean_root` (env `FEP_LEAN_ROOT` overrides the
  `../fep_lean` default), `lean_toolchain_available`, `verify_document`
  (one document → bridge `verify-document` invocation + parsed receipt),
  `run_lean_scripts` (per-framework runner: discovers `*.lean` and GNN `*.md`
  documents under the rendered target directory and verifies each; returns
  `True` only when every document is well-formed or nothing was found).
- `__init__.py` — curated re-export surface (`__all__`).

## Public API

`FEP_LEAN_ROOT_ENV`, `resolve_fep_lean_root`, `lean_toolchain_available`,
`verify_document`, `run_lean_scripts`.

## Contract

- The runner mirrors the per-framework runner signature
  (`rendered_simulators_dir`, `execution_output_dir`, `recursive_search`,
  `verbose`) so `gnn.execute.executor`'s `ExecutorFrameworkSpec` registry can
  drive it like any other backend (`lean_executions` result key).
- Unavailable checkout ⇒ skip with an info log and a `False` return from
  `run_lean_scripts`; `verify_document` returns
  `{"success": False, "error": "fep_lean unavailable"}`.
- Verification is fail-closed: warnings fail under `--fail-on-warnings`, and a
  non-zero bridge exit records the CLI error text in the result record.
- Scope boundary: `verify-document` proves syntax + `WellFormed` only. It does
  NOT auto-prove `DiscreteConforms` / `ContinuousConforms` (out of scope, see
  bridge contract v0.5 §13).
