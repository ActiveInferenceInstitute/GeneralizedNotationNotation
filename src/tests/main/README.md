# src/tests/main

Test package for the src-root entry surfaces (`src/main.py`,
`src/manuscript_variables.py`, `src/__init__.py`) owned by the srcroot
fleet worker.

## What is covered

- `test_main_step_selection.py`: the pure step-selection core
  (`select_pipeline_steps`, `StepSelection`), strict vs. lenient step-list
  parsing, `step_number_from_script_name`, and the fail-fast +
  log-preserving behavior of `_resolve_steps_to_execute`.
- `test_manuscript_variables_api.py`: `save_variables` → `load_variables`
  round-trip, validation errors, and `token_checksum` stability against the
  real producer output.

## Run

```bash
uv run --extra dev python -m pytest src/tests/main/ -q
```

Tests are deterministic, offline, and import `main` /
`manuscript_variables` via the `src/`-on-`sys.path` convention shared by the
suite (see `src/tests/test_manuscript_variables.py`).
