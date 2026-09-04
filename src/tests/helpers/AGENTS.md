# Test Helpers Sub-module

## Overview

Shared, typed test utilities used across the GNN test suite: script loading,
canonical sample content, an MCP registry stub, sample-model path helpers,
and a render-recovery helper for test isolation.

## Architecture

```
helpers/
├── __init__.py            # Re-exports every public symbol below
├── script_loader.py       # load_module_from_path(): importlib loader for standalone scripts
├── gnn_samples.py         # SAMPLE_GNN_CONTENT + write_sample_gnn_markdown()
├── mcp_stubs.py           # MCPTools: in-memory MCP registry stub
├── render_recovery.py     # render_gnn_files(): recovery-friendly bulk render
└── (path helpers in __init__.py for test_data/)
```

## Key Exports

- `load_module_from_path(name, path, sys_path=None)` — load a standalone script as a module; optional sibling-directory `sys.path` injection with automatic cleanup
- `SAMPLE_GNN_CONTENT` — canonical minimal POMDP GNN markdown (single source for the conftest `sample_gnn_*` fixtures)
- `write_sample_gnn_markdown(target)` — write the ontology-annotated sample markdown (creates parents)
- `MCPTools` — in-memory MCP registry stub (`register_tool` / `register_resource` / `execute_tool`); the conftest `test_mcp_tools` fixture returns an instance
- `get_test_data_dir()` / `get_sample_gnn_model()` / `load_sample_gnn_spec()` — path helpers for `src/tests/test_data/`
- `render_gnn_files()` — render every GNN file in a directory, capturing per-file results for recovery tests

## Usage

```python
from tests.helpers import (
    MCPTools,
    SAMPLE_GNN_CONTENT,
    get_sample_gnn_model,
    load_module_from_path,
    load_sample_gnn_spec,
    write_sample_gnn_markdown,
)
from tests.helpers.render_recovery import render_gnn_files
```

## Parent Module

See [tests/AGENTS.md](../AGENTS.md) for the overall test architecture.

**Version**: 3.2.0
