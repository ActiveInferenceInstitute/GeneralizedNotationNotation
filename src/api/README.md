# GNN API Module

## Overview
The `api` module provides a robust REST API for the Generalized Notation Notation (GNN) ecosystem. It exposes core pipeline functionalities, including parsing, validation, execution, and MCP tool registration, over HTTP.

## Architecture
Implemented via FastAPI, the module offers asynchronous job execution, in-memory job management, and seamless integration with the Model Context Protocol (MCP). It features 7 core endpoints (e.g., `/process`, `/jobs/{id}`, `/tools`, `/health`).

## Usage
Start the server using Uvicorn or the provided CLI commands. The API is distributed as an optional extra (`api`) in the `pyproject.toml`.

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
