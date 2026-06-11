# Type Checker Checking Agent

## Overview
This directory owns the core structural type-checking rules for `src/type_checker/`.

## Purpose
- Implement dimension and rule checks used by Step 5.
- Keep public exports in `__init__.py` aligned with `core.py`, `dimensions.py`, and `rules.py`.
- Keep tests in `src/tests/type_checker/` focused on real checker behavior.

## Verification
Run `uv run --extra dev python -m pytest src/tests/type_checker/ -q`.
