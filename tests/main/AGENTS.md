# Main Tests — Agent Scaffolding

## Overview

Tests for the `gnn` entry surfaces: the composable step-selection core in
`src/gnn/main.py` (`select_pipeline_steps`, `parse_step_list_strict`,
`step_number_from_script_name`, the `_resolve_steps_to_execute` adapter) and
the manuscript-variable round-trip API in `src/gnn/manuscript/variables.py`
(`load_variables`, `token_checksum`).

## Running

```bash
uv run --extra dev python -m pytest tests/main/ -q
```

## Files

- `test_main_step_selection.py` — pure selection contract, lenient
  `parse_step_list` acceptance, fail-fast error paths, log-line preservation.
- `test_manuscript_build_figures.py` — figure-build registry contract for
  `scripts/manuscript_build_figures.py` (registration, registry JSON,
  manuscript references, alt text).
- `test_manuscript_variables.py` — producer-drift regression suite for
  `gnn.manuscript.generate_variables` against the live repository.
- `test_manuscript_variables_api.py` — `save_variables` → `load_variables`
  round-trip, validation errors, `token_checksum` stability.
