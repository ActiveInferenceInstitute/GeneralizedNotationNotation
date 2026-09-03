# GNN API Module

## Overview

FastAPI-based REST interface for the GNN processing pipeline. Enables headless pipeline execution, job management, and individual tool invocation over HTTP.

**Optional dependency**: Requires the `api` extra (`uv sync --extra api`)

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/run` | Trigger pipeline execution with options |
| `GET` | `/api/v1/runs` | List known runs |
| `GET` | `/api/v1/runs/{run_hash}` | Poll run status |
| `GET` | `/api/v1/runs/{run_hash}/report` | Download the Markdown report |
| `GET` | `/api/v1/runs/{run_hash}/stream` | Stream progress as SSE |
| `GET` | `/api/v1/health` | Inspect API and renderer health |

## Usage

### Start the API Server

```bash
# Via CLI
gnn serve --host 0.0.0.0 --port 8000

# Direct
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Example Requests

```bash
# Run pipeline
curl -X POST http://localhost:8000/api/v1/run \
  -H "Content-Type: application/json" \
  -d '{"target_dir": "input/gnn_files", "skip_steps": [13]}'

# Stream pipeline progress
curl -N http://localhost:8000/api/v1/runs/abc123/stream
```

## Architecture

- **Framework**: FastAPI with async support
- **Job management**: Background tasks with unique job IDs; run execution delegates
  to the real `main.py` orchestrator in a worker thread
- **SSE streaming**: Server-Sent Events for real-time progress updates
- **Validation**: Shared validation logic from `gnn.schema`
- **Entry point**: `api.app:start_server()` (called by `gnn serve`)
- **Response contract**: JSON and SSE payloads use `{status,data,error,meta}`;
  the report download intentionally remains `text/markdown`
- **Job/tool surface**: `api.server:app` provides `/api/v1/process`,
  `/api/v1/jobs`, and `/api/v1/tools` for explicit job and step management

## File Structure

```text
api/
├── __init__.py    # Module metadata and feature flags
├── app.py         # FastAPI "run" surface; `gnn serve` entry point (start_server)
├── server.py      # FastAPI "job/tool" surface (process/jobs/tools)
├── processor.py   # In-memory job manager
├── models.py      # Pydantic request/response models
├── responses.py   # Shared response envelope and exception handlers
├── auth.py        # Optional API-key authentication (GNN_API_KEY)
├── path_utils.py  # Symlink-safe repo path resolution
├── rate_limit.py  # Request rate limiting
├── mcp.py         # MCP tool registration manifest
├── AGENTS.md      # Agent documentation
├── README.md      # This file
├── SPEC.md        # Module specification
└── SKILL.md       # Capability API
```

## References

- [AGENTS.md](AGENTS.md) — Agent documentation
- [SPEC.md](SPEC.md) — Module specification
