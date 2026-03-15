# GNN API Module

## Overview

FastAPI-based REST interface for the GNN processing pipeline. Enables headless pipeline execution, job management, and individual tool invocation over HTTP.

**Optional dependency**: Requires the `api` extra (`uv sync --extra api`)

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/run` | Trigger full pipeline execution with options |
| `GET` | `/api/status/{job_id}` | Poll job status (queued → running → done/failed) |
| `POST` | `/api/validate` | Validate a GNN file (upload or inline content) |
| `POST` | `/api/parse` | Parse a GNN file and return JSON AST |
| `POST` | `/api/render` | Render a GNN file to a specific framework |
| `GET` | `/api/stream/{job_id}` | SSE stream of real-time pipeline progress |

## Usage

### Start the API Server

```bash
# Via CLI
gnn serve --host 0.0.0.0 --port 8000

# Direct
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Example Requests

```bash
# Run pipeline
curl -X POST http://localhost:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"target_dir": "input/gnn_files", "skip_steps": [13]}'

# Validate a file
curl -X POST http://localhost:8000/api/validate \
  -F "file=@input/gnn_files/discrete/actinf_pomdp_agent.md"

# Stream pipeline progress
curl -N http://localhost:8000/api/stream/abc123
```

## Architecture

- **Framework**: FastAPI with async support
- **Job management**: Background tasks with unique job IDs
- **SSE streaming**: Server-Sent Events for real-time progress updates
- **Validation**: Shared validation logic from `gnn.schema`
- **Entry point**: `api.app:start_server()` (called by `gnn serve`)

## File Structure

```text
api/
├── __init__.py    # Module metadata and feature flags
├── app.py         # FastAPI application and endpoint definitions
├── AGENTS.md      # Agent documentation
├── README.md      # This file
└── SPEC.md        # Module specification
```

## References

- [AGENTS.md](AGENTS.md) — Agent documentation
- [SPEC.md](SPEC.md) — Module specification
