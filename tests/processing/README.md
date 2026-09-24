# Processing Tests

Pytest coverage for `src/gnn/processing/`.

This folder contains module-focused tests for the processing public API,
lightweight parsing/structural checks, and file discovery. It mirrors the
`src/gnn/processing/` layout.

Run:

```bash
uv run --extra dev python -m pytest tests/processing/ -q
```

`mcp.py` exposes the MCP tool surface and is out of scope for these minimal tests.
