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
   - `/health`: System health status.
   - `/process`: Synchronous or asynchronous GNN file processing.
   - `/jobs/{id}`: Polling endpoint for job status.
   - `/tools`: Interface for MCP tool discovery.

## Implementation Details
Requires `fastapi>=0.100.0` and `uvicorn[standard]>=0.23.0` (installed via the `api` extra).

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
