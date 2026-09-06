# Test Helpers — Technical Specification

**Version**: 1.6.0

## Module API

- `get_test_data_dir()` → `Path` — Path to `src/tests/test_data/`
- `get_sample_gnn_model()` → `Path` — Path to the on-disk `sample_gnn_model.md` fixture
- `load_sample_gnn_spec()` → `dict` — Parses the sample model; falls back to a minimal spec dict when the file is missing
- `render_gnn_files(target_dir, output_dir)` → `dict` — Recovery-friendly bulk render used by `src/tests/pipeline/test_pipeline_recovery.py`

## Recovery Pattern

`render_gnn_files()` tolerates a patched `numpy.typing` raising `RecursionError` (bumps the recursion limit and records `recursion_limit_adjusted`), globs with string paths to dodge pathlib recursion edge cases, and writes scaffold artifacts plus a summary dict with recovery actions.

