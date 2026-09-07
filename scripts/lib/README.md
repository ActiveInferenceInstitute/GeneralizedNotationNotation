# scripts/lib/ — Shared Script Utilities

## Overview

Common utility functions used across audit, check, and manuscript-figure scripts. Reduces code duplication for repo-root discovery, skip-path detection, and exit-code handling, and supplies the token-map loader that records figure provenance.

## Contents

| File | Purpose |
|---|---|
| [`shared.py`](shared.py) | `repo_root()`, `should_skip_path()`, `is_generated_output()`, `add_strict_flag()`, `exit_with_findings()` |
| [`manuscript_figure_tokens.py`](manuscript_figure_tokens.py) | `load_tokens()` — the manuscript token map as a mapping that records every key a figure generator reads |

## Usage

```python
from scripts.lib.shared import repo_root, should_skip_path, exit_with_findings

root = repo_root()
path = root / "some" / "file.md"
if not should_skip_path(path, root):
    # process the file...
    pass

count = 0  # number of issues found
sys.exit(exit_with_findings(count, strict=True))
```

## Adding New Functions

1. Add to `shared.py` with docstring and type annotations.
2. Update [`AGENTS.md`](AGENTS.md) with the new export.