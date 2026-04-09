# GNN API Module — Agent Scaffolding

## Module Overview

**Purpose**: FastAPI-based REST interface for programmatic pipeline invocation, job management, and tool discovery.
**Pipeline Step**: Infrastructure module (not a numbered step)
**Category**: Infrastructure / API
**Status**: ✅ Production Ready
**Version**: 1.0.0
**Last Updated**: 2026-03-24

The `api` module provides a FastAPI-based REST interface for the GNN processing pipeline.
It enables programmatic pipeline invocation, job management, and tool discovery without
requiring direct CLI access.

## Architecture

```
src/api/
  __init__.py    -- Module metadata, availability checks
  models.py      -- Pydantic request/response models (API contract)
  processor.py   -- In-memory job manager, async execution
  server.py      -- FastAPI app with routes and middleware
  mcp.py         -- MCP tool registration manifest
  AGENTS.md      -- This file
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/v1/process | Submit pipeline job |
| GET | /api/v1/jobs/{id} | Poll job status |
| DELETE | /api/v1/jobs/{id} | Cancel job |
| GET | /api/v1/jobs | List recent jobs |
| GET | /api/v1/tools | List pipeline steps |
| POST | /api/v1/tools/{step} | Invoke single step |
| GET | /api/v1/health | Health check |

## Installation

The API module requires optional dependencies:

```bash
uv sync --extra api
# then:
python -m api.server
# or:
uvicorn api.server:app --reload
```

## Design Decisions

- **No authentication**: Research tool for local use only. Document explicitly.
- **In-memory jobs**: No database required. Jobs lost on restart.
- **AsyncIO execution**: Non-blocking pipeline runs via asyncio subprocess.
- **Background tasks**: FastAPI BackgroundTasks for fire-and-forget job execution.
- **CORS**: Allows localhost origins for browser-based access.

## Integration with Pipeline

This module is NOT a numbered pipeline step. It is an optional service module
that wraps the pipeline for API access. Import it independently:

```python
from api.processor import create_job, execute_job_async
```

## Agent Guidance

When working with this module:
1. The `processor.py` contains job lifecycle logic -- extend it for persistence needs
2. The `models.py` is the API contract -- change carefully to preserve backwards compatibility
3. Add new endpoints in `server.py` following the existing pattern
4. MCP tools in `mcp.py` should mirror significant REST endpoints


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
