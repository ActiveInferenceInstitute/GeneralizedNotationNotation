# CLI Module — Specification

## Purpose

Provide a unified `gnn` CLI entry point that dispatches to pipeline module APIs.

## Requirements

1. **Subcommand routing**: 12 subcommands (`run`, `validate`, `parse`, `render`, `report`, `reproduce`, `preflight`, `health`, `serve`, `lsp`, `watch`, `graph`)
2. **Lazy imports**: Each handler imports its target module only when invoked
3. **Standard exit codes**: 0=success, 1=error
4. **Verbose mode**: `--verbose` / `-v` flag enables DEBUG logging globally
5. **Path management**: All handlers ensure `src/` is on `sys.path`

## Interface

```python
def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint. Returns exit code."""
```

## Constraints

- No state between invocations (pure CLI tool)
- All domain logic lives in target modules, not in CLI handlers
- Entry point registered in `pyproject.toml` as `gnn = "src.cli:main"`
