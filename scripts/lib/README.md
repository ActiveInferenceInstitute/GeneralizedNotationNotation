# scripts/lib/ — Shared Script Utilities

## Overview

Common utility functions used across audit and check scripts. Reduces code duplication for repo-root discovery, skip-path detection, and exit-code handling.

## Contents

| File | Purpose |
|---|---|
| [`shared.py`](shared.py) | `repo_root()`, `should_skip_path()`, `is_generated_output()`, `add_strict_flag()`, `exit_with_findings()` |

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