# Scripts Toolkit

The `scripts/` directory is the repository's hub for standalone maintenance, linting, and developer acceleration utilities. These tools operate externally to the core `src/` orchestrator pipeline, actively protecting codebase integrity over time.

## Key Files
- `check_gnn_doc_patterns.py`: A strict RegEx-enforced documentation linter that audits `doc/` and `src/gnn/` against deprecated path aliases and import references.

## Execution
Tools are designed to be executed via `uv run` internally from the project root:

```bash
uv run python scripts/check_gnn_doc_patterns.py --strict
```
