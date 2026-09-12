# `gnn/utils/errors/` — agent contract

Concern-package home (S2-33 Step 7, family 3/3; design:
[`../../../../docs/development/utils_split_design.md`](../../../../docs/development/utils_split_design.md)).
The old top-level module paths under `gnn/utils/` remain as DeprecationWarning
facades for the deprecation window — new code imports from this package.

## Leaves

- `error_handling.py` — `PipelineErrorHandler`/`PipelineError`, severity /
  category / recovery-strategy enums, `ExitCode` contract, and the step
  exit-code helpers (`coerce_step_exit_code`, `status_from_exit_code`,
  `pipeline_exit_code`, `is_critical_pipeline_step`, ...).
- `error_recovery.py` — `ErrorRecoveryManager`/`ErrorReporter`,
  `ErrorContext`/`ErrorRecord` structures, `ErrorCodeRegistry`, and the
  formatting helpers (`format_error_message`, `format_and_log_error`,
  `get_recovery_manager`).

## Invariants

- `_EXPORT_MAP` keys in `gnn/utils/__init__.py` are frozen (113 keys,
  golden-asserted by `tests/tests/test_infrastructure_exports.py`); this
  package's leaf modules are the *values*.
- Eager re-export (design §4.3.1): both leaves are stdlib-only, so this
  package's `__init__` re-exports public names as real objects — no PEP 562
  indirection. Importing `gnn.utils` never touches this package (the
  top-level facade stays lazy; `tests/tests/test_light_import.py` is the
  gate).
- Cross-family imports go through leaf modules, never through any facade.
- Zero behavior changes: this package is a mechanical reorganization; the
  per-module contracts live in each leaf's docstring.

## Tests

`tests/tests/test_infrastructure_exports.py` (frozen surface),
`tests/tests/test_light_import.py` (lazy facade), and the per-family suites
`tests/utils/test_error_recovery_framework.py`,
`tests/utils/test_pipeline_template_exit_codes.py`,
`tests/utils/test_argument_and_dependency_contracts.py`, and the pipeline
suites `tests/pipeline/test_main_orchestrator.py`,
`tests/pipeline/test_step_executor.py`, `tests/pipeline/test_pipeline_recovery.py`.
