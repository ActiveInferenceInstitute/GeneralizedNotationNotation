# API Index

This directory contains machine-readable indices for the GNN codebase.

- `api_index.json`: Generated map of modules, functions, and classes under `src/`. Created by `scripts/generate_api_index.py`.

## Generate

```bash
python scripts/generate_api_index.py
```

## Notes

- Tests and output directories are excluded from the index.
- The index is AST-derived and contains file paths, module names, function signatures, class bases, and docstrings when available. 