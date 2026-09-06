# Utils Tests

Pytest coverage for `src/gnn/utils/`.

This folder contains module-focused tests for shared helpers, pipeline templates, schemas, and framework probes.

## Test Files

- `test_argument_and_dependency_contracts.py` — interface guardrails for step arguments and dependency closure, including the exit-code contract wording pinned across maintained docs.
- `test_error_recovery_framework.py` — error message formatting, recovery suggestions, and error-handling behavior.
- `test_framework_availability.py` — framework availability probes verified against the real interpreter's `importlib.util.find_spec` results.
- `test_io_utils.py` — edge cases for the shared file I/O helpers: text vs bytes vs serialized writes, atomic temp-file replacement, missing-input handling.
- `test_new_utils.py` — `gnn/utils/step_logging.py` and `gnn/utils/base_processor.py` behavior.
- `test_path_conversion.py` — string-to-`Path` coercion, `None` handling for critical path arguments, and config validation entry points.
- `test_pipeline_config_merge.py` — merging `input/config.yaml` defaults into pipeline arguments and step-command construction.
- `test_pipeline_template_exit_codes.py` — the widened exit-code contract (`0=success, 1=error, 2=success with warnings/skipped`).
- `test_pipeline_template_factory.py` — the `create_standardized_pipeline_script` factory: argument parsing, logging, output-directory resolution, exit-code coercion.
- `test_pipeline_warnings_fix.py` — pipeline prerequisite checking and output-directory management (nested-directory detection, warning generation).
- `test_safe_eval.py` — the bounded `literal_eval` wrapper.
- `test_shared_helpers.py` — the consolidated shared helpers: writable-probe, memory probe, resource limits, and the fallback-default argument table.
- `test_utils_core.py` — `GNNPipelineConfig` defaults, validation, and core utility behavior.
- `test_validation_schemas.py` — validation schemas exercised with real tempdirs.

Run:

```bash
uv run --extra dev python -m pytest tests/utils/ -q
```
