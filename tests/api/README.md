# API Tests

Pytest coverage for `src/gnn/api/`.

This folder contains module-focused tests for API startup, endpoints, and service contracts.
Worker-limit coverage checks strict shared admission, request/MCP schema bounds,
stored and polled settings, run-hash configuration and subprocess propagation.
It does not claim that configured limits were used by native processing.

Run:

```bash
uv run --extra dev python -m pytest tests/api/ -q
```
