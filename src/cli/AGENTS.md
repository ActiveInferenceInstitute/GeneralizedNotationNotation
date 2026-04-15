# CLI Module — Agent Scaffolding

## Module Overview

**Purpose**: Command-line interface and dispatch functionality for the GNN pipeline.
**Pipeline Step**: Infrastructure module (not a numbered step)
**Category**: Infrastructure / Development Tools
**Status**: ✅ Production Ready
**Version**: 1.5.0
**Last Updated**: 2026-04-15

The CLI module provides the `gnn` command-line tool — a unified interface to the entire GNN pipeline. It acts as a thin dispatcher, routing 12 subcommands to their respective module APIs.

## Architecture

- **Pattern**: Thin dispatcher (not a pipeline step)
- **Entry point**: `src.cli:main` (registered in `pyproject.toml [project.scripts]`)
- **Dependencies**: All pipeline modules (imported lazily per subcommand)

## Capabilities

- **Pipeline execution** via `gnn run` with skip/only-steps, log-format, and skip-llm options
- **File validation** via `gnn validate` (section, state-space, connection, dimension checks)
- **JSON/YAML parsing** via `gnn parse` with format and summary modes
- **Code generation** via `gnn render` (PyMDP, RxInfer, JAX, NumPyro, Stan, PyTorch)
- **Run reproduction** via `gnn reproduce` using content-addressable hashing
- **Environment checks** via `gnn preflight` and `gnn health`
- **Live development** via `gnn watch` (file monitoring with 250ms debounce)
- **Dependency graphs** via `gnn graph` (Mermaid/text output)
- **API server** via `gnn serve` (delegates to `api/app.py`)
- **LSP server** via `gnn lsp` (delegates to `lsp/__init__.py`)

## File Structure

```
cli/
├── __init__.py    # Main dispatcher (414 lines, 12 subcommands)
├── AGENTS.md      # This file
├── README.md      # Usage guide
└── SPEC.md        # Module specification
```

## References

- [README.md](README.md) — Usage guide with examples
- [SPEC.md](SPEC.md) — Specification
