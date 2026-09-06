# CLI Module — Agent Scaffolding

## Module Overview

**Purpose**: Command-line interface and dispatch functionality for the GNN pipeline.
**Pipeline Step**: Infrastructure module (not a numbered step)
**Category**: Infrastructure / Development Tools
**Status**: ✅ Production Ready
**Version**: 3.2.0
**Last Updated**: 2026-09-03

The CLI module provides the `gnn` command-line tool — a unified interface to the entire GNN pipeline. It acts as a thin dispatcher, routing 16 subcommands to their respective module APIs. Public exits are `0` for success, `1` for errors, and `2` for completed commands with warnings or degraded readiness.

## Architecture

- **Pattern**: Thin dispatcher (not a pipeline step)
- **Entry point**: `src.cli:main` (registered in `pyproject.toml [project.scripts]`)
- **Dependencies**: All pipeline modules (imported lazily per subcommand)

## Capabilities

- **Pipeline execution** via `gnn run` with skip/only-steps, log-format, and skip-llm options
- **File validation** via `gnn validate` (section, state-space, connection, dimension checks)
- **JSON/YAML parsing** via `gnn parse` with format and summary modes
- **Code generation** via `gnn render` (PyMDP, RxInfer, JAX, NumPyro, Stan, PyTorch)
- **POMDP extraction** via `gnn extract` (structured JSON of the POMDP state space, with graceful degradation when the extractor is unavailable)
- **Run reproduction** via `gnn reproduce` using content-addressable hashing
- **Environment checks** via `gnn preflight` and `gnn health`
- **Live development** via `gnn watch` (file monitoring with 250ms debounce)
- **Dependency graphs** via `gnn graph` (Mermaid/text output)
- **API server** via `gnn serve` (delegates to `api/app.py`)
- **LSP server** via `gnn lsp` (delegates to `lsp/__init__.py`)

## File Structure

```
cli/
├── __init__.py          # Main dispatcher and 16 subcommands
├── __main__.py          # `python -m cli` entry point
├── lsp.py               # GNN Language Server (stdio)
├── mcp.py               # MCP tool surface for CLI subcommands
├── templates.py         # Maintained template index and copy helpers
├── template_index.json  # Externalized template metadata
├── template_assets/     # Packaged GNN template files
├── AGENTS.md            # This file
├── README.md            # Usage guide
├── SPEC.md              # Module specification
└── SKILL.md             # Capability API
```

## References

- [README.md](README.md) — Usage guide with examples
- [SPEC.md](SPEC.md) — Specification

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
