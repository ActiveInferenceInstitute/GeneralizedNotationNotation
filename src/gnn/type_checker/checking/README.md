# Type Checker Checking

Core structural checking package for Step 5.

This folder contains the rule and dimension-checking implementation used by `src/gnn/type_checker/processor.py`.

Both `Categorical` and the authored `categorical` spelling are valid state-space
types. Validation retains the source spelling; this alias does not relax shape,
matrix, or unknown-type checks.

Run:

```bash
uv run --extra dev python -m pytest tests/type_checker/ -q
```
