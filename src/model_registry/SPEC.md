# Model Registry Module Specification

## Overview

Centralized registry for GNN models with versioning, metadata management, and lifecycle tracking. Pipeline Step 4.

## Architecture

The model registry uses a class-based design:

- **`registry.py`** — Core implementation containing three classes:
  - `ModelVersion` — Represents a specific version with SHA-256 hash, file path, and metadata
  - `ModelEntry` — Represents a model with version history, tags, and searchable metadata
  - `ModelRegistry` — Centralized registry with JSON persistence, search, and model lifecycle management
  - `process_model_registry()` — Top-level function that discovers and registers all GNN files in a directory
- **`mcp.py`** — Model Context Protocol tool registrations for registry operations
- **`__init__.py`** — Re-exports `ModelRegistry` and `process_model_registry`

## Key Features

- Model registration with automatic metadata extraction (name, version, author, tags, description)
- SHA-256 content hashing for integrity verification
- Semantic version tracking with version history
- Full-text search across names, descriptions, and tags
- JSON-based registry persistence
- MCP integration for programmatic access

## Key Exports

```python
from model_registry import ModelRegistry, process_model_registry
```

## Dependencies

- **Required**: `json`, `pathlib`, `datetime`, `hashlib`, `re`
- **No external dependencies**

## Testing

```bash
uv run python -m pytest src/tests/test_model_registry_integration.py -v
```
