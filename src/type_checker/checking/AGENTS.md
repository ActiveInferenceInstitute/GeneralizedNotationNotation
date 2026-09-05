# Type Checker Checking Agent

## Overview
This directory owns the core structural type-checking rules for `src/type_checker/`.

## Purpose
- Implement dimension and rule checks used by Step 5.
- `sections.py` owns the single section-scoped content extraction shared by the checker and the estimator; `summary.py` owns the `ValidationSummary` aggregation.
- Keep public exports in `__init__.py` aligned with `core.py`, `sections.py`, `summary.py`, `dimensions.py`, and `rules.py`.
- Keep tests in `src/tests/type_checker/` focused on real checker behavior.

## Verification
Run `uv run --extra dev python -m pytest src/tests/type_checker/ -q`.
