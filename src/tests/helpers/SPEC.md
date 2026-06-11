# Test Helpers — Technical Specification

**Version**: 1.6.0

## Fixture API

- `create_temp_gnn_file(content, suffix)` → `Path` — Creates temporary GNN file for test isolation
- `get_test_output_dir(test_name)` → `Path` — Returns unique output directory per test
- `safe_render_cleanup(output_dir)` — Cleans render artifacts without affecting other tests

## Recovery Pattern

`render_recovery.py` implements try/finally cleanup to prevent test pollution from render step side effects.
