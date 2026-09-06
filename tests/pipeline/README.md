# Pipeline Tests

Pytest coverage for `src/pipeline/` and `src/main.py`.

This folder contains module-focused and cross-step tests for orchestration, numbered scripts, recovery, and render-execute-analysis chains.

Run:

```bash
uv run --extra dev python -m pytest src/tests/pipeline/ -q
```
