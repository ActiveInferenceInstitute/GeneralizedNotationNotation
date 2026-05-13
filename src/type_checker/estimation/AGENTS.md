# Type Checker Estimation Agent

## Overview
This directory owns resource-estimation logic for `src/type_checker/`.

## Purpose
- Estimate computational cost, memory pressure, and reportable resource signals for Step 5.
- Keep Markdown and HTML reporting helpers aligned with estimator outputs.
- Keep tests in `src/tests/type_checker/` focused on real estimation behavior.

## Verification
Run `uv run pytest src/tests/type_checker/ -q`.
