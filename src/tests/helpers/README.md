# Test Helpers

Shared test utilities and fixtures used across the GNN test suite.

## Contents

- `__init__.py` — Common fixtures: `create_temp_gnn_file()`, `get_test_output_dir()` (60 lines)
- `render_recovery.py` — Render step recovery for test isolation (57 lines)

## Usage

```python
from tests.helpers import create_temp_gnn_file
from tests.helpers.render_recovery import safe_render_cleanup
```

## See Also

- [Parent: tests/README.md](../README.md)
