# Test Helpers Sub-module

## Overview

Shared test utilities used across the GNN test suite: sample-model path helpers and a render-recovery helper for test isolation.

## Architecture

```
helpers/
├── __init__.py            # Path helpers for test_data/ + sample-model loader; re-exports render_gnn_files
└── render_recovery.py     # render_gnn_files(): recovery-friendly bulk render used by resilience tests
```

## Key Exports

- `get_test_data_dir()` — path to `src/tests/test_data/`
- `get_sample_gnn_model()` — path to the on-disk `sample_gnn_model.md` fixture
- `load_sample_gnn_spec()` — parse the sample model into a spec dict (falls back to a minimal dict when the file is missing)
- `render_gnn_files()` — render every GNN file in a directory, capturing per-file results for recovery tests

## Usage

```python
from tests.helpers import get_sample_gnn_model, load_sample_gnn_spec
from tests.helpers.render_recovery import render_gnn_files
```

## Parent Module

See [tests/AGENTS.md](../AGENTS.md) for the overall test architecture.

**Version**: 3.2.0
