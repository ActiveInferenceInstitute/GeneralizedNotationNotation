---
name: api
description: Architectural specification for the API module
---

# API SPEC

## Architecture
The API module is a FastAPI-driven server acting as the RESTful presentation layer for the GNN pipeline. 

### Core Components
1. **Server Initialization**: Bootstraps the FastAPI application and mounts routers.
2. **Job Management**: Provides asynchronous execution and tracking of long-running GNN pipeline tasks.
3. **Endpoints**: 
   - `/api/v1/health`: System health status.
   - `/api/v1/run` and `/api/v1/runs/{hash}`: Run submission and polling.
   - `/api/v1/process` and `/api/v1/jobs/{id}`: Explicit job submission and polling.
   - `/api/v1/tools`: Pipeline-step discovery and invocation.
4. **Response contract**: JSON responses and SSE data payloads have the exact
   top-level shape `{status, data, error, meta}`. Validation and unexpected
   failures use the same shape; the report download is native Markdown.

## Implementation Details
Requires `fastapi>=0.100.0` and `uvicorn[standard]>=0.23.0` (installed via the `api` extra).

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
