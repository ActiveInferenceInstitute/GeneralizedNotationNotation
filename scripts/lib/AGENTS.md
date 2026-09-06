# scripts/lib/ — Shared Script Utilities

## Overview

`scripts/lib/shared.py` provides common functions used across multiple audit and check scripts under `scripts/`. It reduces duplication of ROOT discovery, skip-path logic, and exit-code handling.

`scripts/lib/manuscript_figure_tokens.py` loads `output/data/manuscript_variables.json` as a mapping that records which keys a figure generator actually reads. `manuscript_build_figures.py` writes those `{key: value}` pairs into `output/figures/figure_registry.json`, and `src/tests/test_manuscript_figure_freshness.py` compares them against the live token map, so a committed PNG cannot keep printing a count the producer has moved past.

## Exports

| Function | Purpose |
|---|---|
| `repo_root()` | Return the repository root (`Path`), derived from file location |
| `should_skip_path(path, root, ...)` | Check if a path matches generated-output or skip-part patterns |
| `is_generated_output(rel)` | Detect paths under `_output`/`_outputs` dirs or generated report prefixes |
| `add_strict_flag(parser)` | Add a standard `--strict` flag to an argparse parser |
| `exit_with_findings(count, strict)` | Return exit code 0 or 1 based on findings + strict flag |
| `load_tokens()` | Load the manuscript token map as a `RecordingTokens` mapping; writes consumed `{key: value}` pairs to `$GNN_FIGURE_TOKEN_PROVENANCE` at exit (`manuscript_figure_tokens.py`) |

## Usage

```python
from scripts.lib.shared import repo_root, should_skip_path, exit_with_findings

root = repo_root()
if should_skip_path(some_path, root):
    continue
```

## Adding New Shared Functions

1. Add the function to `scripts/lib/shared.py` (or a purpose-named sibling module).
2. Ensure it has a docstring and type annotations.
3. Update this file and [`README.md`](README.md).