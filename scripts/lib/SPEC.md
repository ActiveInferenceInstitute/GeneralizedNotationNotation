# Specification: scripts/lib/

## Purpose

Shared utility module for audit/check scripts under `scripts/`. Provides common functions for repo path resolution, skip-path logic, generated-output detection, and standardized CLI flags (`--strict`).

## Dependencies

- **Runtime**: Python 3.11+ standard library (`pathlib`, `argparse`, `typing`)
- **No external dependencies** — zero install footprint

## Exports

| Symbol | Type | Description |
|---|---|---|
| `repo_root()` | `-> Path` | Repository root, derived from file location |
| `should_skip_path(path, root)` | `-> bool` | True when path matches skip/generated-output patterns |
| `is_generated_output(rel)` | `-> bool` | True when relative path is under `_output`/`_outputs` |
| `add_strict_flag(parser)` | `-> None` | Adds `--strict` to an argparse parser |
| `exit_with_findings(count, strict)` | `-> int` | 0 if no findings or non-strict, 1 if strict + findings |

## Rules

1. All functions must have type annotations.
2. All functions must have docstrings.
3. No external dependencies beyond Python stdlib.
4. Any script in `scripts/` may import from `scripts.lib.shared` using:
   ```python
   from scripts.lib.shared import repo_root, should_skip_path, exit_with_findings
   ```