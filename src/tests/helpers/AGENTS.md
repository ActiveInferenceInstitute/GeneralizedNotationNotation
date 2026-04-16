# Test Helpers Sub-module

## Overview

Shared test utilities and recovery helpers used across the GNN test suite. Provides common fixtures, assertion helpers, and render recovery logic for test stability.

## Architecture

```
helpers/
├── __init__.py            # Helper exports and shared test fixtures (60 lines)
└── render_recovery.py     # Render step recovery helpers for test isolation (57 lines)
```

## Key Exports

- **Test fixtures** — Common `pytest` fixtures for temporary directories, mock GNN files, and pipeline context.
- **`render_recovery`** — Utilities to safely recover from render step failures during integration testing, preventing cascading test failures.

## Usage

```python
from tests.helpers import create_temp_gnn_file, get_test_output_dir
from tests.helpers.render_recovery import safe_render_cleanup
```

## Parent Module

See [tests/AGENTS.md](../AGENTS.md) for the overall test architecture.

**Version**: 1.6.0
