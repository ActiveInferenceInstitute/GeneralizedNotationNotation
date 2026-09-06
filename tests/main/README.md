# Main Tests

Pytest coverage for the `gnn` entry surfaces (`src/gnn/main.py`,
`src/gnn/manuscript_variables.py`) owned by the main-suite fleet worker.

## What is covered

- `test_main_step_selection.py`: the pure step-selection core
  (`select_pipeline_steps`, `StepSelection`), strict vs. lenient step-list
  parsing, `step_number_from_script_name`, and the fail-fast +
  log-preserving behavior of `_resolve_steps_to_execute`.
- `test_manuscript_build_figures.py`: regression contract for
  `scripts/manuscript_build_figures.py` — the
  `(label, generator, expected PNG, alt text)` table is the single source for
  figure registration, `output/figures/figure_registry.json`, manuscript
  figure references, and alt-text presence.
- `test_manuscript_variables.py`: producer-drift regression suite for
  `gnn.manuscript_variables.generate_variables` — every assertion recomputes
  the expected value from the live repository and compares it against the
  producer.
- `test_manuscript_variables_api.py`: `save_variables` → `load_variables`
  round-trip, validation errors, and `token_checksum` stability against the
  real producer output.
- `test_version_consistency.py`: single-version contract —
  `pyproject.toml` is the source of truth and every `__version__` literal
  (`gnn`, `gnn.cli`, `gnn.api`) plus the FastAPI app metadata must agree
  with it.

## Run

```bash
uv run --extra dev python -m pytest tests/main/ -q
```

Tests are deterministic, offline, and import `gnn.main` /
`gnn.manuscript_variables` through the installed `gnn` package (see
`tests/main/test_manuscript_variables.py`).
