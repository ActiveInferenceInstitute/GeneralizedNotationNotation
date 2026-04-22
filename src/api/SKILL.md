---
name: gnn-api
description: "GNN REST API server for pipeline execution and job management. Use when starting the GNN API server, triggering pipeline steps via HTTP, polling job status, or invoking GNN tools without the CLI."
---

# GNN REST API (Optional Module)

Provides a FastAPI-based REST interface for triggering pipeline steps, polling job status, and invoking individual GNN tools over HTTP.

Requires the `[api]` extra: `uv sync --extra api`

## Key Commands

```bash
# Start the API server
python -m api.server

# Or via the MCP pipeline step (also registers API tools)
python src/main.py --only-steps 21 --verbose
```

## API

```python
from api.processor import create_job, get_job, list_jobs

# Create a pipeline job
job_id = create_job(
    target_dir="input/gnn_files",
    steps=[3, 5, 11, 12],
    verbose=True,
)

# Poll job status
job = get_job(job_id)
print(job["status"])  # "pending" | "running" | "completed" | "failed"

# List all jobs
jobs = list_jobs()
```

## Recommended Workflow

```bash
# 1. Install API dependencies
uv sync --extra api

# 2. Start the server
python -m api.server

# 3. Trigger a pipeline run via HTTP
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"target_dir": "input/gnn_files", "steps": [3,5,11]}'

# 4. Poll job status
curl http://localhost:8000/jobs/<job-id>
```

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Architectural specification
