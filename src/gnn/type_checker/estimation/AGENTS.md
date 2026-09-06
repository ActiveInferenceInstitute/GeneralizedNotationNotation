# Type Checker Estimation Agent

## Overview
This directory owns resource-estimation logic for `src/gnn/type_checker/`.

## Purpose
- Estimate computational cost, memory pressure, and reportable resource signals for Step 5. `estimator.py` classifies the spec's `## Time` section as Static/Dynamic/Hierarchical and reuses `checking.sections` for section-scoped parsing.
- Keep Markdown and HTML reporting helpers aligned with estimator outputs.
- Keep tests in `tests/type_checker/` focused on real estimation behavior.

## Verification
Run `uv run --extra dev python -m pytest tests/type_checker/ -q`.
