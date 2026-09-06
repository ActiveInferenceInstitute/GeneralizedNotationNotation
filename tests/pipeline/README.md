# Pipeline Tests

Pytest coverage for `src/gnn/pipeline/` and `src/gnn/main.py`.

This folder contains module-focused and cross-step tests for orchestration, numbered scripts, recovery, and render-execute-analysis chains.

Run:

```bash
uv run --extra dev python -m pytest tests/pipeline/ -q
```
