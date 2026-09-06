# src/tests/main — Agent Scaffolding

## Overview

Tests for the src-root entry surfaces: the composable step-selection core in
`src/main.py` (`select_pipeline_steps`, `parse_step_list_strict`,
`step_number_from_script_name`, the `_resolve_steps_to_execute` adapter) and
the manuscript-variable round-trip API in `src/manuscript_variables.py`
(`load_variables`, `token_checksum`).

## Running

```bash
uv run --extra dev python -m pytest src/tests/main/ -q
```

## Files

- `test_main_step_selection.py` — pure selection contract, lenient
  `parse_step_list` back-compat, fail-fast error paths, log-line preservation.
- `test_manuscript_variables_api.py` — `save_variables` → `load_variables`
  round-trip, validation errors, `token_checksum` stability.
